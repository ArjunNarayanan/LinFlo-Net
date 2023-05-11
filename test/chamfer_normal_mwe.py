import torch
from pytorch3d.loss import chamfer_distance


verts = torch.rand([1,1000,3])
normals = torch.rand([1,1000,3])

chd, chn = chamfer_distance(verts, verts, x_normals=normals, y_normals=normals)
# chd = 0, chn = 1e-8

chd, chn = chamfer_distance(verts, verts, x_normals=normals, y_normals=-normals)
# chd = 0, chn = 1e-8