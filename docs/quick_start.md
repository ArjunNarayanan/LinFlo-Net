# Quick start

Run heart mesh prediction from a CT or MR NIfTI scan using the `linflonet` CLI.

## Prerequisites

- Python 3.10–3.12
- A trained model checkpoint (`best_model.pth`)
- An input image (`.nii` or `.nii.gz`)
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

## Predict a single image

Provide your own model checkpoint and set `--modality` to match how the model was trained (`ct` or `mr`).

```commandline
linflonet predict \
    --image /path/to/scan.nii.gz \
    --model /path/to/best_model.pth \
    --modality ct \
    --output /path/to/output
```

Outputs:

- `<output>/meshes/<stem>.vtp` — deformed heart mesh
- `<output>/segmentation/<stem>.nii.gz` — mesh rasterized to image space

Template mesh and distance map default to files bundled with the package. Override with `--template` and `--template-distance-map` if needed.

For **linear-transform-only** models (no flow/UDF stage), add `--linear-transform`.

## Predict a folder of images

Works with a flat folder of NIfTI files or a folder containing an `image/` subdirectory:

```commandline
linflonet predict \
    --folder /path/to/images \
    --model /path/to/best_model.pth \
    --modality mr \
    --output /path/to/output
```

Limit to the first *N* files with `-n N` (default: all).

## Using a YAML config

Example configs: `config/predict_single_ct.yml`, `config/predict_single_mr.yml`.

```commandline
linflonet predict \
    --config config/predict_single_ct.yml \
    --image /path/to/scan.nii.gz \
    --output /path/to/output
```

CLI flags override values in the config file (`--model`, `--modality`, `--template`, etc.).

## Python API

```python
from linflonet.predict import PredictionConfig, predict_images

config = PredictionConfig(
    model="/path/to/best_model.pth",
    template="whole_heart_with_ao.vtp",  # bundled template by basename
    modality="ct",
    template_distance_map="highres_template_distance.vtk",
)
predict_images(config, ["/path/to/scan.nii.gz"], "/path/to/output")
```

## Troubleshooting

| Problem | Likely fix |
|---------|------------|
| `ModuleNotFoundError: No module named 'pytorch3d'` | Install `pytorch3d` after `torch` (see above) |
| `ModuleNotFoundError: No module named 'torch'` during pytorch3d install | Use `--no-build-isolation` |
| Template file not found | Use a bundled name (`whole_heart_with_ao.vtp`) or pass an absolute path with `--template` |
| Missing `--model` / `--modality` | Pass both flags, or use `--config` |

For training, dataset preparation, and HPC setup, see the [main README](../README.md).
