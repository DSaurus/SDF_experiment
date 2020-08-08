import torch
import numpy as np
from utils.position_encoding import position_encoding_xyz
from utils.file_io import export_obj
from skimage import measure

def gen_mesh(file_name, net, code, scale=np.array([1, 1, 1])):
    # code [B, C, 1]
    # net decoder
    res = 128
    x,y,z = torch.meshgrid(torch.linspace(-1, 1, res), torch.linspace(-1, 1, res), torch.linspace(-1, 1, res))
    grid = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=1)

    pts_num = int(5e5)
    sdf = np.zeros(res**3)
    for i in range(0, grid.shape[0], pts_num):
        print(i)
        query_pts = grid[i:min(grid.shape[0], i+pts_num), :].numpy()
        print(query_pts)
        query_encoding = position_encoding_xyz(query_pts, 64)
        query_encoding = torch.FloatTensor(query_encoding).unsqueeze(0).permute(0, 2, 1).to(code.device)
        t_code = code.repeat(1, 1, query_encoding.shape[2])
        query_encoding = torch.cat([t_code, query_encoding], dim=1)
        result = net(query_encoding, pos_encoding=True).detach().cpu().numpy()
        print(result.shape)
        print(np.max(result), np.min(result))
        sdf[i:min(grid.shape[0], i+pts_num)] = result[0, 0, :]
    sdf = sdf.reshape((res, res, res))
    vs, fcs, vn, _ = measure.marching_cubes(sdf, 0.5)
    vs *= scale
    export_obj(file_name, vs, fcs, vn)


