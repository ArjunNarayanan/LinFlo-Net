import os
import sys
import glob
import numpy as np
import SimpleITK as sitk

if __name__ == '__main__':
    '''This script resample the segmentation to have consistent spacing, origin and dimension with the image data
    and updates the label ids. The filenames in the segmentation and image folders are assumed to match.
    '''
    image_dir = "/Users/arjun/Documents/Research/SimCardio/Datasets/WholeHeartDataWithInlets/train/mr/image"
    input_segmentation_dir = "/Users/arjun/Documents/Research/SimCardio/Datasets/WholeHeartDataWithInlets/train/mr/raw_labels"
    output_segmentation_dir = "/Users/arjun/Documents/Research/SimCardio/Datasets/WholeHeartDataWithInlets/train/mr/label"
    extension = ".gz"

    if not os.path.isdir(output_segmentation_dir):
        os.makedirs(output_segmentation_dir)

    segmentation_files = sorted(glob.glob(os.path.join(input_segmentation_dir, "*" + extension)))

    # origianl label ids
    ids_o = [0, 1, 2, 3, 4, 5, 6, 7]
    # new label ids
    ids = [0, 205, 420, 500, 550, 600, 820, 850]

    for file_name in segmentation_files:
        sample_name = os.path.basename(file_name)
        image_file = os.path.join(image_dir, sample_name)
        assert os.path.isfile(image_file)
        print("Processing file ", image_file)

        image = sitk.ReadImage(image_file)
        current_segmentation = sitk.ReadImage(file_name)

        new_segmentation = sitk.Resample(current_segmentation, image.GetSize(),
                                         sitk.Transform(),
                                         sitk.sitkNearestNeighbor,
                                         image.GetOrigin(),
                                         image.GetSpacing(),
                                         image.GetDirection(),
                                         0,
                                         current_segmentation.GetPixelID())

        py_im = sitk.GetArrayFromImage(new_segmentation)
        u_ids = np.unique(py_im)

        assert len(u_ids) == len(ids)
        if not all([u == i] for (u,i) in zip(u_ids,ids)):
            print("\tReassigning labels")
            for i in range(len(ids)):
                py_im[py_im==u_ids[i]] = ids[i]
                py_im[py_im==i] = ids[i]
                py_im[py_im==ids_o[i]] = ids[i]

        out_im = sitk.GetImageFromArray(py_im)
        out_im.SetOrigin(new_segmentation.GetOrigin())
        out_im.SetSpacing(new_segmentation.GetSpacing())
        out_im.SetDirection(new_segmentation.GetDirection())
        out_file = os.path.join(output_segmentation_dir, sample_name)
        sitk.WriteImage(out_im, out_file)

