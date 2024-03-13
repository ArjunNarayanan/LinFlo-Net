# LinFlo-Net

A deep learning package to automatically generate simulation ready 3D meshes of the human heart from biomedical images. [Link to paper](https://asmedigitalcollection.asme.org/biomechanical/article/doi/10.1115/1.4064527/1194613).

![image](figures/flow-deformation-no-encoder.png)


## Setting up environment on Savio

The following instructions can set up a conda environment on the Berkeley Research Computing Savio system. But a similar approach can be used on any SLURM based high-performance computing cluster.

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

To perform augmentation, modify the command below and execute it. The script below launches 16 jobs in parallel (`-np 16`). You can modify that depending on the capacity of the system you are using.

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

We generate ground-truth meshes using marching cubes on the ground-truth segmentations. We can do this using `workflows/prepare_data.py`.

```commandline
python workflows/prepare_data \
    --image /path/to/image/folder \
    --segmentation /path/to/segmentation/folder \
    --output /path/to/output/folder \
    --modality ct # can be either ct or mr
    --ext .nii.gz # input files extension
```

The output folder is going to have 3 subfolders : `seg`, `vtk_image`, `vtk_mesh`. `vtk_image` will be the input to our neural network, and `vtk_mesh` will be the corresponding ground truth meshes. From this point onward, we assume that the folder with the relevant data has the `vtk_image` and `vtk_mesh` subfolders.

### Final steps

The data set is reasonably large, and we will have to load it from memory. It is useful to store the images as pytorch tensors and the meshes as pytorch3d data structures in pickled files. To do this, we first build a csv index of all the files.

```commandline
python utilities/prepare_train_dataset_csv.py -f /path/to/data/folder
```

Make sure to provide the path to the parent directory containing `vtk_image` and `vtk_mesh` sub-directories. This will create an `index.csv` in the parent folder with the names of all files. Next,

```commandline
python utilities/pickle_image_segmentation_mesh_dataset.py -config /path/to/config/file
```

Look at `config/pickle_dataset.yml` for an example config file. Note that `seg_label` in the config file follows the labelling convention of the MMWHS dataset.

The output folder will now contain `.pkl` files which contain the combined image, segmentations, and meshes in a dictionary. This can be used by our dataloader to load the appropriate files during training.


## Training the model

Before training, make sure to activate the conda environment that we created earlier. Request a GPU session if you would like to use a GPU for training. Alternatively, submit the below commands as part of a batch job with `sbatch` on a SLURM system. The training workflow will save the best performing model as a checkpoint in the output directory specified in the config file.

### Training Linear Transformation module

Take a look at the example config file in `config/linear_transform.yml`. Make a copy, and modify it appropriately.

Then run the command,

```commandline
python workflows/train_linear_transform.py -config /path/to/config/file
```

### Training the Flow Deformation module

Take a look at the example config file in `config/flow_deformation.yml`. Make a copy, and modify it appropriately. In particular, make sure you provide the path to the linear transformation module trained in the previous step.

Then run the command,

```commandline
python workflows/train_flow_with_udf.py -config /path/to/config/file
```

## Using trained models on new data

The pre-trained model takes as input a CT image in NIFTI format, a template mesh in VTP format and outputs a deformed mesh in VTP format.

First, place your image data in a folder named `image`. Let the path to this folder be `/path/to/folder/image`. Make sure that the images have extension `.nii.gz` or `.nii`. Next, run the following command to build an index of the image dataset,

```
python utilities/prepare_test_data_csv.py -f /path/to/folder
```

Note that the argument to `-f` is the path to the **parent** directory of the `image` directory.

After generating the index, it's time to execute the model.

Take a look at the example config file `config/predict_test_meshes_ct.yml`. Modify the path to the model, path to the image dataset, and the path to your output directory. Next, run the prediction script,

```
python utilities/predict_udf_test_meshes.py -config /path/to/config/file
```

Use the script `utilities/predict_test_meshes.py` if you want to evaluate the Linear Transform as a standalone module.

The script will generate output meshes and segmentations for each input image file.