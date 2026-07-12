"""Validate HiDMoA dataset paths and class-incremental splits."""

from __future__ import annotations

import argparse
import os
from collections import Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check HiDMoA dataset layout and task splits.")
    parser.add_argument("--dataset", default=None, help="Dataset preset name, e.g. neu, xsdd, neu_xsdd.")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size for transform construction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset:
        os.environ["CIL_ACTIVE_DATASET"] = args.dataset

    from .config import DATA
    from .data import prepare_datasets

    image_size = int(args.image_size or DATA["image_size"])
    datasets = prepare_datasets(
        DATA["data_root"],
        DATA["class_names"],
        image_size,
        DATA.get("dataset_class_names"),
    )

    print("\n[HiDMoA data check]")
    print(f"dataset roots: {DATA['data_root']}")
    print(f"num classes: {DATA['num_classes']}")
    print(f"task splits: {DATA['task_splits']}")
    for split_name, ds in datasets.items():
        counts = Counter(label for _path, label in ds.samples)
        print(f"{split_name}: {len(ds)} images")
        missing = [cid for cid in range(DATA["num_classes"]) if counts.get(cid, 0) == 0]
        if missing:
            print(f"  warning: no samples for class ids {missing}")
    print("data check finished")


if __name__ == "__main__":
    main()
