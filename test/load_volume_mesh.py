import vtk_utils.vtk_utils as vtu
import numpy as np
import pickle
from src.template import *
import torch

# fn = "data/template/lv_volume.vtu"
# mesh = vtu.load_vtk_mesh(fn)
#
# points = vtu.vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32)
# data = {"lv-interior": points}
# pickle.dump(data, open("data/template/lv-interior.pkl", "wb"))


template_fn = "data/template/highres_template.vtp"
point_cloud_fn = "data/template/lv-interior.pkl"

template = Template.from_vtk(template_fn)
point_cloud = pickle.load(open(point_cloud_fn, "rb"))["lv-interior"]
point_cloud = torch.from_numpy(point_cloud)

vertices = template.verts_list()
faces = template.faces_list()
data = {"vertices": vertices, "faces": faces, "point_cloud": point_cloud}

out_fn = "data/template/template_with_volume.pkl"
pickle.dump(data, open(out_fn, "wb"))