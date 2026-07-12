"""Convert class-folder image datasets into the HiDMoA directory layout."""

from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw class folders like raw_root/class_name/*.jpg into "
            "images/{train,val,test} and labels/{train,val,test}."
        )
    )
    parser.add_argument("--raw-root", required=True, help="Input root containing one subfolder per class.")
    parser.add_argument("--out-root", required=True, help="Output dataset root for HiDMoA.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    return parser.parse_args()


def _split_items(items: list[Path], train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    n = len(items)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1 if n else 0), n)
    n_val = min(max(n_val, 1 if n - n_train > 1 else 0), n - n_train)
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


def main() -> None:
    args = parse_args()
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")

    raw_root = Path(args.raw_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw root not found: {raw_root}")

    class_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"no class subdirectories found under {raw_root}")

    rng = random.Random(args.seed)
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_names = [p.name for p in class_dirs]
    (out_root / "classes.txt").write_text(
        "\n".join(f"{idx} {name}" for idx, name in enumerate(class_names)) + "\n",
        encoding="utf-8",
    )

    copy_fn = shutil.move if args.move else shutil.copy2
    total = 0
    for class_id, class_dir in enumerate(class_dirs):
        images = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        rng.shuffle(images)
        splits = _split_items(images, args.train_ratio, args.val_ratio)
        class_token = _slug(class_dir.name) or f"class{class_id}"
        for split, split_images in splits.items():
            for idx, src in enumerate(split_images):
                dst_name = f"{class_token}_{idx:06d}{src.suffix.lower()}"
                dst_img = out_root / "images" / split / dst_name
                dst_label = out_root / "labels" / split / f"{Path(dst_name).stem}.txt"
                copy_fn(str(src), str(dst_img))
                dst_label.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
                total += 1
        print(
            f"{class_dir.name}: train={len(splits['train'])} "
            f"val={len(splits['val'])} test={len(splits['test'])}"
        )

    print(f"wrote {total} images to {out_root}")


if __name__ == "__main__":
    main()
