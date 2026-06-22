# LinFlo-Net

**New to LinFlo-Net?** See the [Quick start guide](docs/quick_start.md) for install and prediction in a few minutes.

**Install:** `pip install linflonet` ([PyPI](https://pypi.org/project/linflonet/))

**Pre-trained weights:** [Zenodo](https://zenodo.org/records/20802633) ([DOI: 10.5281/zenodo.20802633](https://doi.org/10.5281/zenodo.20802633))

A deep learning package to automatically generate simulation ready 3D meshes of the human heart from biomedical images. [Link to paper](https://asmedigitalcollection.asme.org/biomechanical/article/doi/10.1115/1.4064527/1194613).

![image](figures/flow-deformation-no-encoder.png)


For SLURM-based clusters (e.g. Berkeley Savio), see [Setting up environment on Savio](docs/savio_setup.md).

## Install from PyPI

For prediction only (Python 3.10+), install from [PyPI](https://pypi.org/project/linflonet/):

```commandline
pip install linflonet
```

`pytorch3d` is required but not listed as a pip dependency because it must be
built from source on most platforms. Install `torch` first, then:

```commandline
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## Pre-trained model weights

Pre-trained PyTorch weights for inference are available on [Zenodo](https://zenodo.org/records/20802633) ([DOI: 10.5281/zenodo.20802633](https://doi.org/10.5281/zenodo.20802633)).

Download and extract the archive (~395 MB):

```commandline
curl -L -o LinFlo-Net_weights.zip \
    "https://zenodo.org/records/20802633/files/LinFlo-Net_weights.zip?download=1"
unzip LinFlo-Net_weights.zip
```

This provides `best_model.pth`, the best validation checkpoint from training the full LinFlo-Net architecture (linear transform + flow deformation with signed-distance supervision). The same checkpoint is used for **CT** and **MR** inputs; set the modality at inference time with `--modality ct` or `--modality mr`.

## CLI usage

The `linflonet` command generates heart meshes (`.vtp`) and segmentations for
CT or MR NIfTI images.

**Single image:**

```commandline
linflonet predict \
    --image /path/to/scan.nii.gz \
    --model /path/to/best_model.pth \
    --modality ct \
    --output /path/to/output
```

**Folder of images** (flat folder or `image/` subdirectory):

```commandline
linflonet predict \
    --folder /path/to/images \
    --model /path/to/best_model.pth \
    --modality mr \
    --output /path/to/output
```

Template mesh and distance map default to files bundled with the package. Override
with `--template` and `--template-distance-map` if needed. For linear-transform-only
models, pass `--linear-transform`.

**Using a YAML config** (same format as `config/predict_single_ct.yml`):

```commandline
linflonet predict --config config/predict_single_ct.yml --image /path/to/scan.nii.gz -o /path/to/output
```

Outputs are written to `<output>/meshes/` and `<output>/segmentation/`.

You can also run `python -m linflonet predict ...` or install in editable mode
from a git checkout:

```commandline
pip install -e .
```

## Setting up a local environment with pip (Python 3.12)

If you only need to run prediction (not training), you can set up a lightweight
environment on Python 3.12 using `requirements-py312.txt`.

First, initialize the `vtk_utils` submodule and create a virtual environment,

```commandline
git submodule update --init
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-py312.txt
```

`pytorch3d` does not ship prebuilt wheels for most platforms and must be built
from source *after* `torch` is installed. Its `setup.py` imports `torch` at build
time, so you must disable pip's build isolation (otherwise you get
`ModuleNotFoundError: No module named 'torch'`):

```commandline
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

On macOS, make sure the Xcode command-line tools are installed first
(`xcode-select --install`) so the C++ extension can compile.

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

Download the pre-trained weights from [Zenodo](https://zenodo.org/records/20802633) (see [Pre-trained model weights](#pre-trained-model-weights) above), then run prediction with the `linflonet` CLI. The model takes a CT or MR image in NIfTI format and outputs a deformed heart mesh (`.vtp`) and a segmentation rasterized to image space. Template mesh and distance map are bundled with the package.

**Single image:**

```commandline
linflonet predict \
    --image /path/to/scan.nii.gz \
    --model /path/to/best_model.pth \
    --modality ct \
    --output /path/to/output
```

**Folder of images** (flat folder or `image/` subdirectory):

```commandline
linflonet predict \
    --folder /path/to/images \
    --model /path/to/best_model.pth \
    --modality mr \
    --output /path/to/output
```

See the [Quick start guide](docs/quick_start.md) for full install and usage details.

### Legacy prediction scripts

The repository also includes YAML-driven workflows used during development. Place images in a folder named `image`, build an index with `utilities/prepare_test_data_csv.py`, then run `utilities/predict_udf_test_meshes.py` with a config such as `config/predict_test_meshes_ct.yml`. Use `utilities/predict_test_meshes.py` to evaluate the linear transform module alone.