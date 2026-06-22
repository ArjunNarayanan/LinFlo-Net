"""Paths to bundled package resources."""

import os
from importlib import resources

BUNDLED_TEMPLATES = {
    "whole_heart_with_ao.vtp",
    "highres_template.vtp",
    "highres_template_distance.vtk",
}


def bundled_template(name: str) -> str:
    """Return absolute path to a template file shipped with the package."""
    return str(resources.files("linflonet.data.template").joinpath(name))


def resolve_template_path(path: str) -> str:
    """Use *path* if it exists, otherwise fall back to a bundled template by basename."""
    if os.path.isfile(path):
        return path

    basename = os.path.basename(path)
    if basename in BUNDLED_TEMPLATES:
        bundled = bundled_template(basename)
        if os.path.isfile(bundled):
            return bundled

    raise FileNotFoundError(
        f"Template file not found: {path!r}. "
        f"Provide an existing path or one of the bundled names: {sorted(BUNDLED_TEMPLATES)}"
    )
