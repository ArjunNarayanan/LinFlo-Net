"""Command-line interface for LinFlo-Net."""

from __future__ import annotations

import argparse
import os
import sys

from linflonet import __version__
from linflonet.paths import bundled_template, resolve_template_path
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
        help="Path to a single input image (.nii or .nii.gz). Repeat for multiple files.",
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
        help="Path to template mesh (.vtp). Defaults to bundled whole-heart template.",
    )
    predict.add_argument(
        "--template-distance-map",
        help="Path to template distance map (.vtk or .pth) for flow models",
    )
    predict.add_argument(
        "--linear-transform",
        action="store_true",
        help="Use a linear-transform-only model (skip template distance map)",
    )
    predict.add_argument(
        "-e",
        "--extension",
        default=".nii.gz",
        help="Input image file extension (default: .nii.gz)",
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


def _resolve_config(args: argparse.Namespace) -> tuple[PredictionConfig, str]:
    if args.config:
        config = PredictionConfig.from_yaml(args.config)
        if args.model:
            config.model = args.model
        if args.modality:
            config.modality = args.modality
        if args.template:
            config.template = resolve_template_path(args.template)
        if args.template_distance_map:
            config.template_distance_map = resolve_template_path(args.template_distance_map)
        if args.output_extension:
            config.output_extension = args.output_extension
        if args.linear_transform:
            if args.template_distance_map:
                raise SystemExit(
                    "--linear-transform cannot be combined with --template-distance-map"
                )
            config.template_distance_map = None
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

        template_distance_map = None
        if args.linear_transform:
            if args.template_distance_map:
                raise SystemExit(
                    "--linear-transform cannot be combined with --template-distance-map"
                )
        elif args.template_distance_map:
            template_distance_map = resolve_template_path(args.template_distance_map)
        else:
            template_distance_map = bundled_template("highres_template_distance.vtk")

        config = PredictionConfig(
            model=args.model,
            template=resolve_template_path(
                args.template or bundled_template("whole_heart_with_ao.vtp")
            ),
            modality=args.modality,
            template_distance_map=template_distance_map,
            output_extension=args.output_extension,
        )

    if args.output:
        out_dir = args.output
    elif args.images and len(args.images) == 1 and not args.folder:
        out_dir = os.path.dirname(os.path.abspath(args.images[0]))
    elif args.folder:
        out_dir = args.folder
    else:
        out_dir = "."

    return config, out_dir


def _run_predict(args: argparse.Namespace) -> None:
    if not args.images and not args.folder:
        raise SystemExit("Provide --image and/or --folder.")

    if args.images:
        for image_fn in args.images:
            if not os.path.isfile(image_fn):
                raise SystemExit(f"Did not find image file: {image_fn}")

    if args.folder and not os.path.isdir(args.folder):
        raise SystemExit(f"Did not find folder: {args.folder}")

    config, out_dir = _resolve_config(args)

    if args.images and args.folder:
        image_files = list(args.images)
        image_files.extend(find_image_files(args.folder, args.extension))
    elif args.images:
        image_files = args.images
    elif args.folder:
        predict_folder(
            config,
            args.folder,
            out_dir,
            extension=args.extension,
            max_files=args.n,
        )
        return
    else:
        raise SystemExit("Provide --image and/or --folder.")

    if args.n >= 0:
        image_files = image_files[: args.n]

    predict_images(config, image_files, out_dir, extension=args.extension)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        _run_predict(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
