![image](figures/flow-deformation-no-encoder.png)

# LinFlo-Net

This is a deep learning package to automatically generate simulation ready 3D meshes of the human heart from biomedical images.

## Setting up environment on Savio

The following instructions can set up a conda environment on the Berkeley Research Computing Savio system. But a similar approach can be used on an SLURM based high-performance computing cluster.

Since Savio provides limited space in your home directory, we install all conda packages to our scratch folder.

```commandline
module load cuda/10.2
module load gcc/5.4.0
ENVDIR=/global/scratch/users/<your_username>/environments/linflonet
rm -rf $ENVDIR
export CONDA_PKGS_DIRS=/global/scratch/users/<your_username>/tmp/.conda
conda create --prefix $ENVDIR
```

Press `y` when prompted to create your conda environment and then activate your environment,

```commandline
source activate $ENVDIR
```

Next install `pytorch`. Savio does not have the version of `cuda` required for the latest pytorch version, so we will install `pytorch 1.12.1`.

```commandline
conda install pytorch==1.12.1 cudatoolkit=10.2 -c pytorch
```

Press `y` when prompted to start the installation.

Next we will install [pytorch3d](https://pytorch3d.org/) which provides several useful routines for dealing with 3D data and mesh data-structures in conjunction with `pytorch`,

```commandline
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

You can find all the other dependencies in the system generated `requirements.txt` in the repository. You should be able to install these directly with `pip` after installing the above packages.

### Test that everything works

First request a brief interactive session with a GPU,

```commandline
srun --pty -A <account_name> -p savio3_gpu --nodes=1 --gres=gpu:GTX2080TI:1 --ntasks=1 --cpus-per-task=2 -t 00:30:00 bash -i
```

Once your resources are allocated, load your conda environment and launch python

```commandline
source activate /global/scratch/users/<your_username>/environments/linflonet
python
```

Now type the following into your python session,

```python
import torch
from pytorch3d.loss import chamfer_distance
device = torch.device("cuda")

a = torch.rand([5,10000,3]).to(device)
b = torch.rand([5,10000,3]).to(device)
loss = chamfer_distance(a, b)
```

If everything runs without error, you are all set!

## Dataset Creation

We use the multi-modality whole heart segmentation challenge (MMWHS) [dataset](https://zmiclab.github.io/zxh/0/mmwhs/). Download and unzip the data. You should have the following folders,

   - CT : 2 folders each with 10 images and corresponding segmentations
   - MR : 1 folder with 20 images and corresponding segmentations

You can split the data into train and validation as you find appropriate. We chose to use the first 16 samples as training and the remaining 4 samples as validation. Split the data appropriately and place them in separate folders. Make sure to keep the CT and MR data separately as we will need to normalize / scale them differently. We will perform data augmentation on the training data.

### Data augmentation

We will use the data augmentation procedure available in the [MeshDeformNet](https://github.com/fkong7/MeshDeformNet) package. Clone this package to your system and run `pip install -r requirements.txt` to install package dependencies. (You may want to create a virtual environment first.)

To perform augmentation, modify the command below and execute it

```commandline
mpirun -np 16 python ~/path/to/MeshDeformNet/data/data_augmentation.py \
    --im_dir /path/to/image/directory \
    --seg_dir /path/to/segmentation/directory \
    --out_dir /path/to/output/directory \
    --modality ct or mr \
    --mode train \
    --num number_of_augmentations
```

The output folder will contain two subfolders `modality_train` with the augmented images and `modality_train_seg` with the augmented segmentations where modality is either `ct` or `mr`.

### Creating ground-truth meshes

We generate ground-truth meshes using marching cubes on the ground-truth segmentations. We can do this using `prepare_data.py` which is in the base directory of this package. We pass directory information using a yaml config file. (All example config files are in the `config/` sub-folder.)

```yaml
im_folder: /path/to/image/folder # for training data use the augmented images
seg_folder: /path/to/segmentation/folder # for training data use the augmented segmentations
out_folder: /path/to/output/folder
modality: ct # ct or mr
extension: .nii.gz # file extension of image and segmentation data
```

The output folder is going to have 3 subfolders : `seg`, `vtk_image`, `vtk_mesh`. `vtk_image` will be the input to our neural network, and `vtk_mesh` will be the corresponding ground truth meshes. From this point onward, we assume that the folder with the relevant data has the `vtk_image` and `vtk_mesh` subfolders.

### Final steps

The data set is reasonably large, and we will have to load it from memory. It is useful to store the images as pytorch tensors and the meshes as pytorch3d data structures in pickled files. To do this, we first build a csv index of all the files.

```commandline
python utilities/prepare_csv.py -f /path/to/data
```

This will create an `index.csv` in the data folder with the names of all files. Next,

```commandline
python utilities/pickle_pytorch3d_dataset.py -f /path/to/data -o /path/to/output/folder
```

The output folder will now contain `.pkl` files which contain the combined image and meshes in a dictionary. This can be used by our dataloader to load the appropriate files during training.


## Using pre-trained model

The pre-trained model takes as input a CT image in NIFTI format, a template mesh in VTP format and outputs a deformed mesh in VTP format.

First, place your image data in a folder named `image`. Let the path to this folder be `/path/to/folder/image`. Make sure that the images have extension `.nii.gz` or `.nii`. Next, run the following command to build an index of the image dataset,

```
python utilities/prepare_test_data_csv.py -f /path/to/folder
```

Note that the argument to `-f` is the path to the **parent** directory of the `image` directory.

After generating the index, it's time to execute the model.

Use one of the config files for example `config/WH/ct/flow/combined-4/predict_test_meshes_ct.yml`. Modify the path to the model, path to the image dataset, and the path to your output directory. Next, run the prediction script,

```
python utilities/predict_test_meshes.py -config /path/to/config/file
```

The script will generate output meshes and segmentations for each input image file.