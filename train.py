import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from skimage import measure
import numpy as np
import time

from dataset.sdf_dataset import SDFDataset
from model.sdf_decoder import SDFDecoder
from model.sdf_encoder import SDFEncoder
from utils.mesh import gen_mesh
from utils.file_io import *

cuda = torch.device('cuda:0')

logger = SummaryWriter('sdf_results')

if __name__ == "__main__":
    np.random.seed(int(time.time()))
    dataset = SDFDataset('../twim_data', random_pts=True)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)

    encoder = SDFEncoder()
    decoder = SDFDecoder()
    # encoder.load_state_dict(torch.load('checkpoints/epoch_encoder_%d.pth' % 0))
    # decoder.load_state_dict(torch.load('checkpoints/epoch_decoder_%d.pth' % 0))
    
    encoder.load_state_dict(torch.load('checkpoints/epoch_encoder_latest.pth'), strict=False)
    decoder.load_state_dict(torch.load('checkpoints/epoch_decoder_latest.pth'), strict=False)

    encoder.to(cuda)
    decoder.to(cuda)

    optimizer_en = torch.optim.Adam(encoder.parameters(), 1e-3)
    optimizer_de = torch.optim.Adam(decoder.parameters(), 1e-3)

    log_file = open('train_log', 'w')

    Epoch = 1000
    for epoch in range(Epoch):
        train_idx = 0
        for data in dataloader:
            sdf = data["sdf"].to(cuda)
            pts = data["pts"].to(cuda)
            pos_encoding = data["pos_encoding"].to(cuda)
            gt_pts = data["gt_pts"].to(cuda)
            scale = data["scale"]

            net_code = encoder(sdf)
            code = net_code.reshape(-1, 512)
            code = code.unsqueeze(1).repeat(1, pts.shape[1], 1)
            # code = torch.cat((code, pts), dim=2)
            # net_sdf = decoder(code.permute(0, 2, 1)).permute(0, 2, 1)
            code = torch.cat((code, pos_encoding), dim=2)
            net_sdf = decoder(code.permute(0, 2, 1), pos_encoding=True).permute(0, 2, 1)
            
            gt_pts = gt_pts.reshape(pts.shape[0], pts.shape[1], 1)
            loss = F.mse_loss(net_sdf, gt_pts)

            one_sdf = torch.sum(net_sdf[gt_pts >= 0.9]) / torch.sum((gt_pts >= 0.9))
            print(one_sdf)
            logger.add_scalar("pos_call", one_sdf, train_idx)
            logger.add_scalar("pos_loss", loss.item(), train_idx)

            optimizer_de.zero_grad()
            optimizer_en.zero_grad()

            loss.backward()

            optimizer_de.step()
            optimizer_en.step()

            print("loss:", loss.item()) 
            # print(torch.sum(net_sdf) / pts.shape[0])       
            
            if train_idx == 0:
                scale = scale[0].numpy()
                gen_mesh('obj_results/%d_test.obj' % epoch, decoder, net_code[0].reshape(-1, 512, 1), scale)
                export_one_pts('pts.obj', pts[0].detach().cpu().reshape(-1, 3).numpy() * scale, 
                    net_sdf[0].detach().cpu().reshape(-1).numpy())
                export_one_pts('gt_pts.obj', pts[0].detach().cpu().reshape(-1, 3).numpy() * scale, 
                    gt_pts[0].detach().cpu().reshape(-1).numpy())
                exit(0)
                # show_sdf = net_sdf.detach().cpu().numpy()[0].reshape((64, 64, 64))
                # print(show_sdf.shape)
                # print(np.max(show_sdf))
                # vs, fcs, vn, _ = measure.marching_cubes(show_sdf)
                # export_obj( % epoch, vs, fcs, vn)
                

            log_file.write("%f\n" % loss.item())   

            train_idx += 1
        torch.save(encoder.state_dict(), 'checkpoints/epoch_encoder_%d.pth' % epoch)
        torch.save(decoder.state_dict(), 'checkpoints/epoch_decoder_%d.pth' % epoch)
        torch.save(encoder.state_dict(), 'checkpoints/epoch_encoder_latest.pth')
        torch.save(decoder.state_dict(), 'checkpoints/epoch_decoder_latest.pth')

