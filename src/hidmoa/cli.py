"""Command line entry point for HiDMoA."""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HiDMoA class-incremental learning.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset preset name defined in DATASET_PRESETS.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base random seed.")
    parser.add_argument("--repeats", type=int, default=None, help="Number of repeated seeds.")
    parser.add_argument("--device", default=None, help="Override training device, e.g. cuda or cpu.")
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a short smoke-test training configuration: 1 repeat, 1 epoch per model/router stage, and no FLOPs profiling.",
    )
    parser.add_argument(
        "--run-mode",
        default="hidmoa",
        choices=["hidmoa", "incremental_2"],
        help="Public alias for the HiDMoA method.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["RUN_MODE"] = "hidmoa" if args.run_mode == "hidmoa" else "incremental_2"
    if args.dataset:
        os.environ["CIL_ACTIVE_DATASET"] = args.dataset
    if args.seed is not None:
        os.environ["CIL_SEED"] = str(args.seed)
    if args.repeats is not None:
        os.environ["CIL_REPEATS"] = str(args.repeats)
    if args.device:
        os.environ["HIDMOA_DEVICE"] = args.device
    if args.quick_test:
        os.environ.setdefault("CIL_REPEATS", "1")
        os.environ.setdefault("CIL_PROFILE_FLOPS", "0")
        os.environ.setdefault("HIDMOA_EPOCHS_PER_TASK", "1")
        os.environ.setdefault("HIDMOA_EARLY_STOP_PATIENCE", "1")
        os.environ.setdefault("HIDMOA_FVAE_EPOCHS", "1")
        os.environ.setdefault("HIDMOA_FVAE_PATIENCE", "1")
        os.environ.setdefault("HIDMOA_FVAE_GENERATED_PER_CLASS", "8")
        os.environ.setdefault("HIDMOA_QUICK_SAMPLES_PER_CLASS", "8")

    from .main import main as run_main

    run_main()


if __name__ == "__main__":
    main()
