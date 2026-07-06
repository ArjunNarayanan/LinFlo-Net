# Quick start

Run heart mesh prediction from CT or MR images using the `linflonet` CLI (linear transform + flow deformation).

## Prerequisites

- Python 3.10–3.12
- Pre-trained weights (`best_model.pth`) from [Zenodo](https://zenodo.org/records/20802633) ([DOI: 10.5281/zenodo.20802633](https://doi.org/10.5281/zenodo.20802633))
- Input images (e.g. `.mha`, `.nii`, or `.nii.gz`)
- A GPU is recommended but not required (CPU fallback is supported)

## Install

Install from [PyPI](https://pypi.org/project/linflonet/) (Python 3.10+):

```commandline
python3 -m venv .venv
source .venv/bin/activate
pip install linflonet
```

Install `pytorch3d` after `torch` (required for model loading; no prebuilt wheels on most platforms):

```commandline
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

On macOS, install Xcode command-line tools first if needed: `xcode-select --install`.

Verify the install:

```commandline
linflonet --version
```

### Development install

To work on the source code, clone the repository and install in editable mode:

```commandline
git clone https://github.com/ArjunNarayanan/LinFlo-Net.git
cd LinFlo-Net
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

Alternatively, install pinned dependencies from the repo before `pytorch3d`:

```commandline
pip install -r requirements.txt
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -e .
```

## Download pre-trained weights

Pre-trained weights for inference are hosted on [Zenodo](https://zenodo.org/records/20802633):

```commandline
curl -L -o LinFlo-Net_weights.zip \
    "https://zenodo.org/records/20802633/files/LinFlo-Net_weights.zip?download=1"
unzip LinFlo-Net_weights.zip
mv best_model.pth model/best_model.pth
```

This is the combined-4 LT+flow checkpoint (~395 MB archive). The same weights work for **CT** and **MR** scans; set the modality at inference time with `--modality ct` or `--modality mr`.

## Batch prediction (recommended)

Place images in `data_coro/` (flat folder or `data_coro/image/`) and run:

```commandline
linflonet predict --config config/WH/ct/flow/combined-4/predict_test_meshes_ct.yml
```

This reads `root_dir`, `extension`, `model`, and `output_dir` from the config. Outputs:

- `output/predict/data_coro/meshes/<stem>.vtp` — deformed heart mesh
- `output/predict/data_coro/segmentation/<stem>.nii.gz` — mesh rasterized to image space

Override paths from the command line if needed:

```commandline
linflonet predict \
    --config config/WH/ct/flow/combined-4/predict_test_meshes_ct.yml \
    --folder /path/to/images \
    -e .mha \
    -o /path/to/output
```

**Legacy equivalent:**

```commandline
python utilities/prepare_test_data_csv.py -f data_coro -e .mha
python utilities/predict_test_meshes.py -config config/WH/ct/flow/combined-4/predict_test_meshes_ct.yml
```

The CLI discovers images directly and does not require `index.csv`.

## Predict a single image

```commandline
linflonet predict \
    --image /path/to/scan.nii.gz \
    --model model/best_model.pth \
    --modality ct \
    --output /path/to/output
```

Or with a config file:

```commandline
linflonet predict \
    --config config/predict_single_ct.yml \
    --image /path/to/scan.nii.gz \
    --output /path/to/output
```

## Predict a folder of images

Works with a flat folder of images or a folder containing an `image/` subdirectory:

```commandline
linflonet predict \
    --folder /path/to/images \
    --model model/best_model.pth \
    --modality mr \
    --output /path/to/output
```

Limit to the first *N* files with `-n N` (default: all).

## Python API

```python
from linflonet.predict import PredictionConfig, predict_images

config = PredictionConfig(
    model="model/best_model.pth",
    template="data/template/whole_heart_with_ao.vtp",
    modality="ct",
)
predict_images(config, ["/path/to/scan.nii.gz"], "/path/to/output")
```

## Troubleshooting

| Problem | Likely fix |
|---------|------------|
| `ModuleNotFoundError: No module named 'pytorch3d'` | Install `pytorch3d` after `torch` (see above) |
| `ModuleNotFoundError: No module named 'torch'` during pytorch3d install | Use `--no-build-isolation` |
| Template file not found | Use `data/template/whole_heart_with_ao.vtp` or pass an absolute path with `--template` |
| Missing `--model` / `--modality` | Download weights from [Zenodo](https://zenodo.org/records/20802633), place at `model/best_model.pth`, then pass both flags or use `--config` |
| `Provide --folder/--image or set files.root_dir in --config` | Add `--folder` / `--image`, or set `files.root_dir` in the YAML config |

For training, dataset preparation, and HPC setup, see the [main README](../README.md).
