# HeartFlow

This is a deep learning package to automatically generate simulation ready 3D meshes of the human heart from biomedical images.

## Setting up environment on Savio

The following instructions can set up a conda environment on the Berkeley Research Computing Savio system.

Since Savio provides limited space in your home directory, we install all conda packages to our scratch folder.

```commandline
module load cuda/10.2
module load gcc/5.4.0
ENVDIR=/global/scratch/users/<your_username>/environments/heartflow
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
source activate /global/scratch/users/<your_username>/environments/heartflow
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