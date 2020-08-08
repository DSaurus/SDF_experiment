import torch
import numpy as np
import os
import trimesh
import trimesh.repair as rp
from torch.utils.data import Dataset
from utils.file_io import export_obj, export_pts_cloud
from utils.position_encoding import position_encoding_xyz
from skimage import measure

class SDFDataset(Dataset):
    def __init__(self, obj_path, random_pts = False):
        super(SDFDataset, self).__init__()

        self.obj_path = obj_path
        self.obj_names = os.listdir(obj_path)
        self.random_pts = random_pts
    

    def __getitem__(self, index):
        obj_name = os.path.join(os.path.join(self.obj_path, self.obj_names[index]), self.obj_names[index][:-2] + ".obj")

        mesh = trimesh.load(obj_name)
        verts = mesh.vertices
        b_min = np.min(verts, axis=0)
        b_max = np.max(verts, axis=0)

        center = mesh.center_mass
        length = np.max(np.max(verts, axis=0) - np.min(verts, axis=0))
        radius = length / 2

        # z, y, x = torch.meshgrid(torch.linspace(center[2] - radius, center[2] + radius, 64),
        #     torch.linspace(center[1] - radius, center[1] + radius, 64),
        #     torch.linspace(center[0] - radius, center[0] + radius, 64))
        z, y, x = torch.meshgrid(torch.linspace(b_min[2], b_max[2], 64),
            torch.linspace(b_min[1], b_max[1], 64),
            torch.linspace(b_min[0], b_max[0], 64))
        
        sample_pts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=1).numpy()

        # show_pts = sample_pts[mesh.contains(sample_pts)]
        # export_pts_cloud("pts.obj", show_pts)

        inside = mesh.contains(sample_pts).reshape((64, 64, 64))
        sdf = np.zeros((64, 64, 64))
        sdf[inside] = 1

        # vs, faces, vn, _ = measure.marching_cubes(sdf, 0.5)
        # export_obj('test.obj', vs, faces, vn)

        print("pts ratio:", np.sum(sdf) / (64**3))

        # z, y, x = torch.meshgrid(torch.linspace(-1, 1, 64),
        #     torch.linspace(-1, 1, 64),
        #     torch.linspace(-1, 1, 64))
        # sample_pts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=1).numpy()
        surface_pts, _ = trimesh.sample.sample_surface(mesh, 5000)

        sample_pts = surface_pts + np.random.normal(scale=0.2, size=surface_pts.shape)
        if self.random_pts:
            random_pts = np.random.rand(5000, 3) * (b_max - b_min) + b_min
            sample_pts = np.concatenate([sample_pts, random_pts], axis=0)
        inside = mesh.contains(sample_pts)

        sample_pts -= (b_max + b_min) / 2
        sample_pts /= (b_max - b_min) / 2

        if self.random_pts:
            gt_pts = np.zeros((10000))
        else:
            gt_pts = np.zeros((5000))
        gt_pts[inside] = 1

        # position encoding
        pos_encoding = position_encoding_xyz(sample_pts, 64)

        res = {
            "sdf" : torch.FloatTensor(sdf).unsqueeze(3).permute(3, 0, 1, 2),
            "pts" : torch.FloatTensor(sample_pts),
            "pos_encoding" : torch.FloatTensor(pos_encoding),
            "gt_pts" : torch.FloatTensor(gt_pts),
            "scale" : torch.FloatTensor(b_max - b_min)
        }

        return res
    
    def __len__(self):
        return self.obj_names.__len__()

