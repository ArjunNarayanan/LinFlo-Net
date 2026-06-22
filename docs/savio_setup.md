# Setting up environment on Savio

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

## Test that everything works

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

For prediction and PyPI install, see the [Quick start guide](quick_start.md). For the full project overview, see the [main README](../README.md).
