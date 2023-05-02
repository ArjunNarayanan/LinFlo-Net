import src.io_utils as io
import torch
from pytorch3d.structures import Meshes
from pytorch3d.structures import join_meshes_as_batch
import pickle
import os


class Template(Meshes):
    def __init__(self, verts, faces, faceids_name=None, faceids=None, **kwargs):
        super().__init__(verts=verts, faces=faces, **kwargs)
        self.faceids_name = faceids_name
        self.faceids = faceids

    @classmethod
    def from_vtk(cls, template_fn, device=None, faceids_name=None):
        mesh_list = io.read_mesh_list(template_fn)
        verts = [io.mesh2verts(m) for m in mesh_list]
        faces = [io.mesh2faces(m) for m in mesh_list]

        if faceids_name is not None:
            faceids = io.read_face_ids(template_fn, faceids_name)
        else:
            faceids = None

        meshes = cls(verts, faces, faceids_name=faceids_name, faceids=faceids)
        if device is not None:
            meshes = meshes.to(device)

        return meshes

    def to(self, device):
        new_mesh = super().to(device)
        new_mesh.faceids = self.faceids
        new_mesh.faceids_name = self.faceids_name
        return new_mesh

    def batch_vertex_coordinates(self, batch_size):
        verts_list = self.verts_list()
        batched_verts = [v.repeat([batch_size, 1, 1]) for v in verts_list]
        return batched_verts

    def batch_face_connectivity(self, batch_size):
        faces_list = self.faces_list()
        batched_faces = [f.repeat([batch_size, 1, 1]) for f in faces_list]
        return batched_faces

    def to_vtk_mesh(self):
        return io.pytorch3d_to_vtk(self)

    def offset_verts(self, offset):
        new_template = super().offset_verts(offset)
        new_template.faceids = self.faceids
        new_template.faceids_name = self.faceids_name
        return new_template

    def update_packed(self, new_verts_packed):
        offset = new_verts_packed - self.verts_packed()
        return self.offset_verts(offset)

    def occupancy_map(self, num_grid_points):
        vertices = self.verts_packed()
        return _occupancy_map(vertices, num_grid_points)


class TemplateWithVolume(Template):
    def __init__(self, verts, faces, faceids_name=None, faceids=None, point_cloud=None, **kwargs):
        super().__init__(verts, faces, faceids_name=faceids_name, faceids=faceids, **kwargs)
        self._INTERNAL_TENSORS.append("point_cloud")
        self.point_cloud = point_cloud

    @classmethod
    def from_pkl(cls, pickled_file, device=None):
        assert os.path.isfile(pickled_file)
        data = pickle.load(open(pickled_file, "rb"))
        vertices = data["vertices"]
        faces = data["faces"]
        point_cloud = data["point_cloud"]
        faceids_name = data.get("faceids_name", None)
        faceids = data.get("faceids", None)

        template = cls(vertices, faces, faceids_name=faceids_name, faceids=faceids, point_cloud=point_cloud)

        if device is not None:
            template = template.to(device)

        return template

    def to(self, device):
        template = super().to(device)
        if self.point_cloud is not None:
            point_cloud = self.point_cloud.to(device)
        verts_list = template.verts_list()
        faces_list = template.faces_list()

        return TemplateWithVolume(
            verts_list,
            faces_list,
            point_cloud=point_cloud,
            faceids_name=self.faceids_name,
            faceids=self.faceids
        )


def _occupancy_map(vertices, num_grid_points):
    occupancy = torch.zeros(3 * [num_grid_points], device=vertices.device)
    idx = (vertices * num_grid_points).clamp(min=0.0, max=num_grid_points - 1).long()

    occupancy[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    occupancy = occupancy.unsqueeze(0)
    return occupancy


def _verts_packed_of_batch(verts_list, batch):
    verts = [v[batch] for v in verts_list]
    verts = torch.cat(verts, dim=0)
    return verts


def batch_occupancy_map_from_padded_verts(batch_padded_verts, batch_size, num_grid_points):
    verts_packed = [_verts_packed_of_batch(batch_padded_verts, b) for b in range(batch_size)]
    occ = [_occupancy_map(v, num_grid_points) for v in verts_packed]
    occupancy = torch.stack(occ)
    return occupancy


class BatchTemplate():
    def __init__(self, meshes_list):
        assert len(meshes_list) > 0
        self.meshes_list = meshes_list
        self.num_components = len(meshes_list)
        self.batch_size = len(meshes_list[0])
        assert all([len(mesh) == self.batch_size for mesh in meshes_list])

    def __getitem__(self, item):
        return self.meshes_list[item]

    @classmethod
    def from_single_template(cls, template, batch_size):
        meshes_list = [join_meshes_as_batch([t.clone() for b in range(batch_size)]) for t in template]
        return cls(meshes_list)

    @classmethod
    def from_template_list(cls, template_list):
        batch_size = len(template_list)
        assert batch_size > 0
        num_components = len(template_list[0])
        assert all((len(t) == num_components for t in template_list))

        meshes_list = []
        for component in range(num_components):
            mesh = join_meshes_as_batch([t[component] for t in template_list])
            meshes_list.append(mesh)

        return cls(meshes_list)

    def batch_component_vertex_coordinates(self, component_id):
        return self.meshes_list[component_id].verts_padded()

    def batch_vertex_coordinates(self, detach=False):
        batched_verts = [m.verts_padded() for m in self.meshes_list]
        if detach:
            batched_verts = [bv.detach() for bv in batched_verts]

        return batched_verts

    def batch_normals(self):
        batched_normals = [m.verts_normals_padded() for m in self.meshes_list]
        return batched_normals

    def _check_batched_verts_shape(self, batched_vertices):
        assert len(batched_vertices) == self.num_components
        assert (all(bv.ndim == 3 for bv in batched_vertices))
        assert all([bv.shape[0] == self.batch_size] for bv in batched_vertices)
        assert all(bv.shape[2] == 3 for bv in batched_vertices)

    def verts_packed_of_batch(self, batch):
        verts_list = [m[batch].verts_packed() for m in self.meshes_list]
        verts = torch.cat(verts_list, dim=0)
        return verts

    def occupancy_map_of_batch(self, batch, num_grid_points):
        verts = self.verts_packed_of_batch(batch)
        occupancy = _occupancy_map(verts, num_grid_points)
        return occupancy

    def update_batched_vertices(self, new_batched_vertices, detach=False):
        self._check_batched_verts_shape(new_batched_vertices)

        for component_id in range(self.num_components):
            if detach:
                bv = new_batched_vertices[component_id].detach()
            else:
                bv = new_batched_vertices[component_id]
            self.meshes_list[component_id] = self.meshes_list[component_id].update_padded(bv)

    def join_component_meshes(self):
        mesh_list = []
        for batch in range(self.batch_size):
            mesh = join_meshes_as_batch([m[batch] for m in self.meshes_list])
            mesh_list.append(mesh)
        return mesh_list

    def occupancy_map(self, num_grid_points):
        occupancy = [self.occupancy_map_of_batch(b, num_grid_points) for b in range(self.batch_size)]
        occupancy = torch.stack(occupancy)
        return occupancy


class BatchTemplateWithVolume(BatchTemplate):
    def __init__(self, meshes_list, point_cloud):
        assert point_cloud.ndim == 3
        batch_size = point_cloud.shape[0]
        assert all([len(m) == batch_size for m in meshes_list])

        super().__init__(meshes_list)
        self.point_cloud = point_cloud

    @classmethod
    def from_pkl(cls, filename, batch_size, device):
        assert os.path.isfile(filename)
        data = pickle.load(open(filename, "rb"))
        vertices = data["vertices"]
        faces = data["faces"]
        point_cloud = data["point_cloud"]
        faceids_name = data.get("faceids_name", None)
        faceids = data.get("faceids", None)

        template = Template(vertices, faces, faceids_name=faceids_name, faceids=faceids)
        if device is not None:
            template = template.to(device)
            point_cloud = point_cloud.to(device)

        meshes_list = [join_meshes_as_batch([t.clone() for b in range(batch_size)]) for t in template]
        point_cloud = point_cloud.repeat([batch_size, 1, 1])

        return cls(meshes_list, point_cloud)
