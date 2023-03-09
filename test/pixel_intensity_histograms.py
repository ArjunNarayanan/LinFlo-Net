import SimpleITK as sitk
import matplotlib.pyplot as plt
import os


def plot_file(fn, ax):
    img = sitk.ReadImage(fn)
    img_arr = sitk.GetArrayFromImage(img).ravel()
    label = os.path.basename(fn)
    label = label.split(".")[0]
    ax.hist(img_arr, label=label, alpha=0.2, density=True)




# fig, ax = plt.subplots()
base_dir = "/Users/arjun/Documents/Research/SimCardio/Datasets/HeartDataSegmentation/test/mr/image"
filename = "mr_test_2004_image.nii.gz"
img_fn = os.path.join(base_dir, filename)
img = sitk.ReadImage(img_fn)
img_arr = sitk.GetArrayFromImage(img).ravel()

# m = img_arr.mean()
# s = img_arr.std()
# img_arr = (img_arr - m)/s
label = filename.split(".")[0]
ax.hist(img_arr, label=label, alpha=0.3, density=True)
ax.legend()
# fig.savefig("output/test/ct_hist.png")