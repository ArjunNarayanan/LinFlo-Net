"""Command-line interface for LinFlo-Net."""

from __future__ import annotations

import argparse
import os
import sys

from linflonet import __version__
from linflonet.paths import resolve_template_path
from linflonet.predict import (
    PredictionConfig,
    find_image_files,
    predict_folder,
    predict_images,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linflonet",
        description="Generate simulation-ready heart meshes from CT or MR images.",
    )
    parser.add_argument("--version", action="version", version=f"linflonet {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser(
        "predict",
        help="Predict heart meshes and segmentations from one or more images",
    )
    predict.add_argument(
        "--config",
        help="YAML config file (overrides individual model/template options)",
    )
    predict.add_argument(
        "-i",
        "--image",
        action="append",
        dest="images",
        help="Path to a single input image. Repeat for multiple files.",
    )
    predict.add_argument(
        "-f",
        "--folder",
        help="Folder containing images, or a folder with an image/ subdirectory",
    )
    predict.add_argument(
        "-o",
        "--output",
        required=False,
        help="Output directory for meshes/ and segmentation/ subfolders",
    )
    predict.add_argument(
        "--model",
        help="Path to trained model checkpoint (best_model.pth)",
    )
    predict.add_argument(
        "--modality",
        choices=["ct", "mr"],
        help="Image modality used for intensity normalization",
    )
    predict.add_argument(
        "--template",
        help="Path to template mesh (.vtp). Defaults to data/template/whole_heart_with_ao.vtp.",
    )
    predict.add_argument(
        "--distance-map",
        dest="distance_map",
        help="Path to template distance map (.vtk or .pth). Required for UDF models "
        "(e.g. UDFLinearTransformSegmentFlow).",
    )
    predict.add_argument(
        "-e",
        "--extension",
        default=None,
        help="Input image file extension (default: .nii.gz, or files.extension in config)",
    )
    predict.add_argument(
        "--output-extension",
        choices=[".nii.gz", ".vti"],
        default=".nii.gz",
        help="Segmentation output format (default: .nii.gz)",
    )
    predict.add_argument(
        "-n",
        type=int,
        default=-1,
        help="When using --folder, limit to the first N images (-1 for all)",
    )

    return parser


def _resolve_config(args: argparse.Namespace) -> PredictionConfig:
    if args.config:
        try:
            config = PredictionConfig.from_yaml(args.config)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.model:
            config.model = args.model
        if args.modality:
            config.modality = args.modality
        if args.template:
            config.template = resolve_template_path(args.template)
        if args.output_extension:
            config.output_extension = args.output_extension
        if args.distance_map:
            config.template_distance_map = args.distance_map
    else:
        missing = []
        if not args.model:
            missing.append("--model")
        if not args.modality:
            missing.append("--modality")
        if missing:
            raise SystemExit(
                "Either --config or all of --model and --modality are required.\n"
                f"Missing: {', '.join(missing)}"
            )

        config = PredictionConfig(
            model=args.model,
            template=resolve_template_path(
                args.template or "data/template/whole_heart_with_ao.vtp"
            ),
            modality=args.modality,
            output_extension=args.output_extension,
            template_distance_map=args.distance_map,
        )

    return config


def _resolve_extension(args: argparse.Namespace, config: PredictionConfig) -> str:
    if args.extension is not None:
        return args.extension
    return config.extension


def _resolve_output_dir(
    args: argparse.Namespace, config: PredictionConfig, folder: str | None
) -> str:
    if args.output:
        return args.output
    if config.output_dir:
        return config.output_dir
    if args.images and len(args.images) == 1 and not folder:
        return os.path.dirname(os.path.abspath(args.images[0]))
    if folder:
        return folder
    if config.root_dir:
        return config.root_dir
    return "."


def _run_predict(args: argparse.Namespace) -> None:
    config = _resolve_config(args)
    folder = args.folder or config.root_dir
    extension = _resolve_extension(args, config)

    if not args.images and not folder:
        raise SystemExit(
            "Provide --folder/--image or set files.root_dir in --config."
        )

    if args.images:
        for image_fn in args.images:
            if not os.path.isfile(image_fn):
                raise SystemExit(f"Did not find image file: {image_fn}")

    if folder and not os.path.isdir(folder):
        raise SystemExit(f"Did not find folder: {folder}")

    out_dir = _resolve_output_dir(args, config, folder)

    if args.images and folder:
        image_files = list(args.images)
        image_files.extend(find_image_files(folder, extension))
    elif args.images:
        image_files = args.images
    else:
        try:
            predict_folder(
                config,
                folder,
                out_dir,
                extension=extension,
                max_files=args.n,
            )
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        return

    if args.n >= 0:
        image_files = image_files[: args.n]

    predict_images(config, image_files, out_dir, extension=extension)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _run_predict(args)


if __name__ == "__main__":
    main(sys.argv[1:])
