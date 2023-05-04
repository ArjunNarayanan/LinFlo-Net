import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np


def vector_field_to_vtk(np_vector_field, output_file):
    assert np_vector_field.ndim == 4
    assert np_vector_field.shape[-1] == 3
    assert all(s == np_vector_field.shape[0] for s in np_vector_field.shape[:-1])

    grid_size = np_vector_field.shape[0]
    assert grid_size > 1
    spacing = 3 * [1.0 / (grid_size - 1)]

    image = vtk.vtkImageData()
    image.SetOrigin(0., 0., 0.)
    image.SetSpacing(spacing)
    image.SetDimensions(grid_size, grid_size, grid_size)

    vtk_data_array = numpy_to_vtk(num_array=np_vector_field.ravel())
    vtk_data_array.SetNumberOfComponents(3)
    vtk_data_array.SetName("vector-field")
    image.GetPointData().SetScalars(vtk_data_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(image)
    writer.SetFileName(output_file)
    print("Writing vector field to : ", output_file)
    writer.Write()


grid_size = 4
nx = ny = nz = grid_size

total_npts = nx * ny * nz

# ax = np.tile(np.linspace(1, grid_size, grid_size), [grid_size, grid_size, 1])
# ax = np.tile(np.linspace(1,grid_size,grid_size).reshape([grid_size,1,1]), [1,grid_size,grid_size])
ax = np.tile(np.linspace(1,grid_size,grid_size).reshape([1,grid_size,1]), [grid_size,1,grid_size])
ay = np.zeros(total_npts).reshape(3 * [grid_size])
az = np.zeros(total_npts).reshape(3 * [grid_size])

grid_vector = np.array([ax, ay, az]).transpose([1, 2, 3, 0])
# grid_vector = np.roll(grid_vector, 2, axis=3)

output_file = "test/output/vector-field.vti"
vector_field_to_vtk(grid_vector, output_file)
