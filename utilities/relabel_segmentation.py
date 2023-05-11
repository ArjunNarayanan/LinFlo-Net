import SimpleITK as sitk
import os
import glob


def relabel_segmentation_array(segmentation_array, old2new_labels):
    for old_label, new_label in old2new_labels.items():
        segmentation_array[segmentation_array == old_label] = new_label


def relabel_segmentation_sitk(segmentation, old2new_labels, dtype=None):
    segmentation_array = sitk.GetArrayFromImage(segmentation)
    relabel_segmentation_array(segmentation_array, old2new_labels)

    if dtype is not None:
        segmentation_array = segmentation_array.astype(dtype)

    new_segmentation = sitk.GetImageFromArray(segmentation_array)
    new_segmentation.CopyInformation(segmentation)
    return new_segmentation


def relabel_segmentation_file(seg_fn, old2new_labels, dtype, output_dir):
    filename = os.path.basename(seg_fn)
    seg = sitk.ReadImage(seg_fn)
    new_seg = relabel_segmentation_sitk(seg, old2new_labels, dtype)
    out_fn = os.path.join(output_dir, filename)

    print("Writing file ", out_fn)
    sitk.WriteImage(new_seg, out_fn)


def relabel_dir(input_dir, old2new_labels, dtype, output_dir, extension=".nii.gz"):
    filenames = glob.glob(os.path.join(input_dir, "*" + extension))
    for fn in filenames:
        relabel_segmentation_file(fn, old2new_labels, dtype, output_dir)
    print("COMPLETED RELABELING ", len(filenames), " SEGMENTATIONS")


input_dir = "/Users/arjun/Documents/Research/SimCardio/HeartFlow/output/WholeHeartData/trained_models/ct/combined/flow/model-1/evaluation/test/ct/segmentation"
old2new_labels = {0: 0, 1: 205, 2: 420, 3: 500, 4: 550, 5: 600, 6: 820, 7: 850}
dtype = "uint16"
output_dir = "/Users/arjun/Documents/Research/SimCardio/HeartFlow/output/WholeHeartData/trained_models/ct/combined/flow/model-1/evaluation/test/ct/relabeled-segmentation"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
relabel_dir(input_dir, old2new_labels, dtype, output_dir)
