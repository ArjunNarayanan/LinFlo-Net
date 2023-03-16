import SimpleITK as sitk

image_fn = "/Users/arjunnarayanan/Documents/Research/Simcardio/Datasets/HeartData-samples/test/mr/image/mr_test_2031_image.nii.gz"
# seg_fn = "/Users/arjunnarayanan/Documents/Research/Simcardio/Datasets/HeartData-samples/train/mr/label/la_001.nii.gz"

image = sitk.ReadImage(image_fn)
# seg = sitk.ReadImage(seg_fn)

cropped_image = image[150:350, 150:500, :]
# cropped_seg = seg[0:180, 130:305, 105:310]

out_im_fn = "output/test/mr_test_2031_image_cropped.nii.gz"
# out_seg_fn = "output/test/crop_la_001_seg.nii.gz"

sitk.WriteImage(cropped_image, out_im_fn)
# sitk.WriteImage(cropped_seg, out_seg_fn)