import SimpleITK as sitk
import os
import pandas as pd

root_dir = "/Users/arjun/Documents/Research/SimCardio/Datasets/WholeHeartDataWithInlets/train/mr"
extension = ".nii.gz"

index_fn = os.path.join(root_dir, "index.csv")
index = pd.read_csv(index_fn)

for idx in range(len(index)):
    fn = index.loc[idx,"file_name"]

    img_fn = os.path.join(root_dir, "image", fn + extension)
    seg_fn = os.path.join(root_dir, "label", fn + extension)

    img = sitk.ReadImage(img_fn)
    seg = sitk.ReadImage(seg_fn)

    if img.GetSize() == seg.GetSize():
        print("File ", fn, " checked")
    else:
        print("WARNING: Dimension mismatch between image and label for file ", fn)
