import os
import SimpleITK as sitk

seg_dir = "/Users/arjunnarayanan/Documents/Research/Simcardio/HeartFlow/output/WholeHeartData/ct-mr-cropped/LT-flow/clip-0075-div-005/CV-auto-crop/mr/segmentation"
img_dir = "/Users/arjunnarayanan/Documents/Research/Simcardio/Datasets/Cardiovascular/mr"
outdir = "translated-segmentation"
output_dir = os.path.join(os.path.dirname(seg_dir), outdir)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

filename = "mr_test_2003_image.nii.gz"

seg_file = os.path.join(seg_dir, filename)
img_file = os.path.join(img_dir, filename)

seg = sitk.ReadImage(seg_file)
img = sitk.ReadImage(img_file)
seg.SetOrigin(img.GetOrigin())

outfile = os.path.join(output_dir, filename)
sitk.WriteImage(seg, outfile)
