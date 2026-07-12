"""
NEU Surface Defect Dataset — YOLO 格式数据加载

数据目录结构:
    data_root/
      images/
        train/   crazing_1.jpg, inclusion_5.jpg, pitted_surface_10.jpg, ...
        val/
        test/
      labels/     (优先读取标签；支持数字 id 或类别名)

类别由文件名前缀决定 (不区分大小写):
    crazing_2.jpg        → Crazing       (class 0)
    inclusion_5.jpg      → Inclusion     (class 1)
    patches_3.jpg        → Patches       (class 2)
    pitted_surface_10.jpg→ Pitted_Surface(class 3)
    rolled-in_scale_7.jpg→ Rolled-in_Scale(class 4)
    scratches_1.jpg      → Scratches     (class 5)
"""

import os
import re
from typing import Dict, List, Optional, Set, Union

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────
class NEUYoloDataset(Dataset):
    """从 YOLO 格式目录读取，优先使用 label 文件，其次回退到文件名前缀。"""

    EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif")
    # 匹配 {classname}_{number}.ext
    _PATTERN = re.compile(r"^(.+)_(\d+)$")

    def __init__(
        self,
        image_dirs: Union[str, List[str]],
        class_names: List[str],
        transform=None,
        dataset_class_names: Optional[List[List[str]]] = None,
    ):
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        self.image_dirs = image_dirs
        self.transform = transform
        for image_dir in image_dirs:
            if not os.path.isdir(image_dir):
                raise FileNotFoundError(f"image directory not found: {image_dir}")

        # 类名 → class_id (不区分大小写 + 规范化)
        name_to_id: Dict[str, int] = {}
        for cid, name in enumerate(class_names):
            name_to_id[name.lower()] = cid
            name_to_id[self._normalize_token(name)] = cid

        dataset_class_names_list = dataset_class_names
        if dataset_class_names_list is not None and len(dataset_class_names_list) != len(image_dirs):
            print(
                f"[NEUYoloDataset] dataset_class_names length mismatch; ignore mapping fallback "
                f"(len={len(dataset_class_names_list)})"
            )
            dataset_class_names_list = None

        class_offsets = None
        local_name_maps = None
        if dataset_class_names_list:
            class_offsets = []
            local_name_maps = []
            cursor = 0
            for names in dataset_class_names_list:
                count = len(names)
                class_offsets.append((cursor, count))
                local_map: Dict[str, int] = {}
                for local_id, name in enumerate(names):
                    gid = cursor + local_id
                    local_map[str(name).lower()] = gid
                    local_map[self._normalize_token(str(name))] = gid
                local_name_maps.append(local_map)
                cursor += count

        self.samples = []
        from_label = 0
        from_name = 0
        for idx, image_dir in enumerate(image_dirs):
            class_offset, local_count = (None, None)
            if class_offsets is not None:
                class_offset, local_count = class_offsets[idx]
            local_name_to_id = local_name_maps[idx] if local_name_maps is not None else None
            label_dir = self._resolve_label_dir(image_dir)
            for fname in sorted(os.listdir(image_dir)):
                if not fname.lower().endswith(self.EXTENSIONS):
                    continue
                path = os.path.join(image_dir, fname)
                label_from_file = self._parse_label_file(
                    label_dir,
                    fname,
                    len(class_names),
                    name_to_id,
                    class_offset=class_offset,
                    local_class_count=local_count,
                    local_name_to_id=local_name_to_id,
                )
                label_from_name = self._parse_label_from_name(fname, name_to_id, local_name_to_id)

                if label_from_file is not None:
                    cid = label_from_file
                    from_label += 1
                elif label_from_name is not None:
                    cid = label_from_name
                    from_name += 1
                else:
                    continue

                self.samples.append((path, cid))

        # 统计
        counts: Dict[int, int] = {}
        for _, cid in self.samples:
            counts[cid] = counts.get(cid, 0) + 1
        print(f"[NEUYoloDataset] roots={len(image_dirs)}")
        for image_dir in image_dirs:
            print(f"  - {image_dir}")
        print(f"  label来源: labels={from_label}  filename={from_name}")
        print(f"  总计 {len(self.samples)} 张  |  "
              + "  ".join(f"{class_names[c]}:{counts.get(c, 0)}" for c in range(len(class_names))))

    @staticmethod
    def _resolve_label_dir(image_dir: str) -> Optional[str]:
        split_name = os.path.basename(image_dir)
        image_root = os.path.dirname(image_dir)
        root = os.path.dirname(image_root)
        label_dir = os.path.join(root, "labels", split_name)
        return label_dir if os.path.isdir(label_dir) else None

    def _parse_label_from_name(
        self,
        fname: str,
        name_to_id: Dict[str, int],
        local_name_to_id: Optional[Dict[str, int]] = None,
    ) -> Optional[int]:
        stem = os.path.splitext(fname)[0]
        match = self._PATTERN.match(stem)
        if not match:
            return None
        cls_name = match.group(1).lower()
        if local_name_to_id is not None:
            cid = local_name_to_id.get(cls_name)
            if cid is not None:
                return cid
            cid = local_name_to_id.get(self._normalize_token(cls_name))
            if cid is not None:
                return cid
        cid = name_to_id.get(cls_name)
        if cid is not None:
            return cid
        return name_to_id.get(self._normalize_token(cls_name))

    @staticmethod
    def _normalize_token(text: str) -> str:
        return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text.strip().lower())).strip("_")

    @staticmethod
    def _parse_label_file(
        label_dir: Optional[str],
        fname: str,
        num_classes: int,
        name_to_id: Dict[str, int],
        class_offset: Optional[int] = None,
        local_class_count: Optional[int] = None,
        local_name_to_id: Optional[Dict[str, int]] = None,
    ) -> Optional[int]:
        if label_dir is None:
            return None
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(label_dir, f"{stem}.txt")
        if not os.path.isfile(label_path):
            return None
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    text = line.strip()
                    for token in (text, parts[0]):
                        token_l = token.lower()
                        if local_name_to_id is not None and token_l in local_name_to_id:
                            return local_name_to_id[token_l]
                        token_n = NEUYoloDataset._normalize_token(token)
                        if local_name_to_id is not None and token_n in local_name_to_id:
                            return local_name_to_id[token_n]
                        if token_l in name_to_id:
                            return name_to_id[token_l]
                        if token_n in name_to_id:
                            return name_to_id[token_n]
                    return None
                if class_offset is not None and local_class_count is not None:
                    if 0 <= cid < local_class_count:
                        mapped = cid + int(class_offset)
                        return mapped if 0 <= mapped < num_classes else None
                    if class_offset == 0 and 0 <= cid < num_classes:
                        return cid
                return cid if 0 <= cid < num_classes else None
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ──────────────────────────────────────────────
#  Transforms
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int, is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ──────────────────────────────────────────────
#  创建全部数据集 (只调用一次, 所有场景共用)
# ──────────────────────────────────────────────
def prepare_datasets(
    data_root: Union[str, List[str]],
    class_names: List[str],
    image_size: int,
    dataset_class_names: Optional[List[List[str]]] = None,
) -> Dict[str, NEUYoloDataset]:
    """
    返回 4 个数据集:
      - 'train'      : 带数据增强
      - 'train_eval' : 不带增强 (用于特征提取 / imprinting)
      - 'val'        : 不带增强
      - 'test'       : 不带增强
    """
    roots = [data_root] if isinstance(data_root, str) else list(data_root)
    if not roots:
        raise ValueError("data_root is empty")

    for root in roots:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"data_root not found: {root}")

    def _split_dirs(split: str) -> List[str]:
        dirs = [os.path.join(root, "images", split) for root in roots]
        missing = [d for d in dirs if not os.path.isdir(d)]
        if missing:
            raise FileNotFoundError(f"image split directory not found: {missing}")
        return dirs

    train_tf = get_transforms(image_size, is_train=True)
    eval_tf = get_transforms(image_size, is_train=False)

    return {
        "train":      NEUYoloDataset(_split_dirs("train"), class_names, train_tf, dataset_class_names=dataset_class_names),
        "train_eval": NEUYoloDataset(_split_dirs("train"), class_names, eval_tf, dataset_class_names=dataset_class_names),
        "val":        NEUYoloDataset(_split_dirs("val"), class_names, eval_tf, dataset_class_names=dataset_class_names),
        "test":       NEUYoloDataset(_split_dirs("test"), class_names, eval_tf, dataset_class_names=dataset_class_names),
    }


# ──────────────────────────────────────────────
#  按任务构建 DataLoader
# ──────────────────────────────────────────────
def build_task_loaders(
    datasets: Dict[str, NEUYoloDataset],
    task_classes: List[int],
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    """
    从预建好的 datasets 中按 task_classes 过滤, 返回 4 个 DataLoader。
    """
    task_set: Set[int] = set(task_classes)
    quick_per_class = int(os.getenv("HIDMOA_QUICK_SAMPLES_PER_CLASS", "0"))

    def _make(key: str, shuffle: bool) -> DataLoader:
        ds = datasets[key]
        indices = [i for i, (_, lbl) in enumerate(ds.samples) if lbl in task_set]
        if quick_per_class > 0:
            kept = []
            per_class_counts = {cid: 0 for cid in task_set}
            for idx in indices:
                lbl = int(ds.samples[idx][1])
                if per_class_counts.get(lbl, 0) >= quick_per_class:
                    continue
                kept.append(idx)
                per_class_counts[lbl] = per_class_counts.get(lbl, 0) + 1
            indices = kept
        return DataLoader(
            Subset(ds, indices),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        )

    loaders = {
        "train":      _make("train",      shuffle=True),
        "train_eval": _make("train_eval", shuffle=False),
        "val":        _make("val",        shuffle=False),
        "test":       _make("test",       shuffle=False),
    }
    sizes = {k: len(v.dataset) for k, v in loaders.items()}
    print(f"[task loaders] classes={task_classes}  sizes={sizes}")
    return loaders
