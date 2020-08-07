import torch
from dataset.sdf_dataset import SDFDataset
from model.sdf_decoder import SDFDecoder
from model.sdf_encoder import SDFEncoder
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from skimage import measure
from utils.file_io import *
import numpy as np

cuda = torch.device('cuda:0')

logger = SummaryWriter('results')

if __name__ == "__main__":
    dataset = SDFDataset('../twim_data')
    dataloader = DataLoader(dataset, batch_size=1, num_workers=3)

    encoder = SDFEncoder()
    decoder = SDFDecoder()
    # encoder.load_state_dict(torch.load('checkpoints/epoch_encoder_%d.pth' % 4))
    # decoder.load_state_dict(torch.load('checkpoints/epoch_decoder_%d.pth' % 4))

    encoder.to(cuda)
    decoder.to(cuda)

    optimizer_en = torch.optim.Adam(encoder.parameters(), 1e-4)
    optimizer_de = torch.optim.Adam(decoder.parameters(), 1e-4)

    log_file = open('train_log', 'w')

    Epoch = 1000
    for epoch in range(Epoch):
        train_idx = 0
        for data in dataloader:
            sdf = data["sdf"].to(cuda)
            pts = data["pts"].to(cuda)

            net_code = encoder(sdf)
            code = net_code.reshape(-1, 512)
            code = code.unsqueeze(1).repeat(1, pts.shape[1], 1)
            code = torch.cat((code, pts), dim=2)
            net_sdf = decoder(code.permute(0, 2, 1)).permute(0, 2, 1)

            sdf = sdf.permute(0, 2, 3, 4, 1).reshape(pts.shape[0], -1, 1)
            
            loss = F.mse_loss(sdf, net_sdf)

            one_sdf = torch.sum(net_sdf[sdf >= 0.9]) / torch.sum((sdf >= 0.9))
            print(one_sdf)
            logger.add_scalar("call", one_sdf, train_idx)
            logger.add_scalar("loss", loss.item(), train_idx)

            optimizer_de.zero_grad()
            optimizer_en.zero_grad()

            loss.backward()

            optimizer_de.step()
            optimizer_en.step()

            print("loss:", loss.item()) 
            # print(torch.sum(net_sdf) / pts.shape[0])       

            # show_sdf = net_sdf.detach().cpu().numpy()[0].reshape((64, 64, 64))
            # print(show_sdf.shape)
            # print(np.max(show_sdf))
            # vs, fcs, vn, _ = measure.marching_cubes(show_sdf, 0.5)
            # export_obj('test.obj', vs, fcs, vn)
            # exit(0)

            log_file.write("%f\n" % loss.item())   

            train_idx += 1
        torch.save(encoder.state_dict(), 'checkpoints/epoch_encoder_%d.pth' % epoch)
        torch.save(decoder.state_dict(), 'checkpoints/epoch_decoder_%d.pth' % epoch)

