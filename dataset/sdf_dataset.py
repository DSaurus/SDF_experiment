import torch
import numpy as np
import os
import trimesh
import trimesh.repair as rp
from torch.utils.data import Dataset
from utils.file_io import export_obj, export_pts_cloud
from skimage import measure

class SDFDataset(Dataset):
    def __init__(self, obj_path):
        super(SDFDataset, self).__init__()

        self.obj_path = obj_path
        self.obj_names = os.listdir(obj_path)
    

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
        surface_pts, _ = trimesh.sample.sample_surface(mesh, 64*64*64)

        sample_pts = surface_pts + np.random.normal(scale=0.2, size=sample_pts.shape)
        inside = mesh.contains(sample_pts)

        sample_pts -= (b_max + b_min) / 2
        sample_pts /= (b_max - b_min) / 2
        gt_pts = np.zeros((64*64*64))
        gt_pts[inside] = 1

        res = {
            "sdf" : torch.FloatTensor(sdf).unsqueeze(3).permute(3, 0, 1, 2),
            "pts" : torch.FloatTensor(sample_pts),
            "gt_pts" : torch.FloatTensor(gt_pts)
        }

        return res
    
    def __len__(self):
        return self.obj_names.__len__()

