import torch
from src.template import Template
from src.chamfer import chamfer_distance
import torch.nn.functional as F

template_fn = "data/template/highres_template.vtp"
template = Template.from_vtk(template_fn)

verts = template.verts_list()[1].unsqueeze(0)
normals = template.verts_normals_list()[1].unsqueeze(0)

# n1 = torch.tensor([[1.,1.,1.]])
# r = F.cosine_similarity(n1, -n1)

chd, chn = chamfer_distance(verts, verts, x_normals=normals, y_normals=-normals)