"""
NEU 钢铁缺陷 — 类增量学习 & 全量基线 配置

结构:
  DATA            : 统一数据配置 (四个场景共用)
  RUN_MODE        : 运行模式开关
  INCREMENTAL_1   : 增量学习1 (冻结 backbone + 逐任务 MoE + VAE + Task-ID)
  INCREMENTAL_2   : 增量学习2 (冻结 backbone + 逐任务式增量训练配置与 incre1 对齐)
  INCREMENTAL_3   : 增量学习3 (初始参数与 incre2 一致, 但独立显式配置)
  FULL_1          : 全量基线1 (6 专家 + 线性 gate + top-2 自由路由)
  FULL_2          : 全量基线2 (6 专家 + 按任务固定分配路由)
  INCRE_*         : 外部增量方法统一配置（MRFA/TagFex/BEEF/EWC/iCaRL/TPL/PEC/SEMA 等）
"""

import os
from copy import deepcopy
import torch


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _repo_data_path(*parts):
    return os.path.join(PACKAGE_ROOT, "data", *parts)


def _normalize_experts_per_task(experts_per_task, moe_layers):
    """Normalize per-task expert config to a per-layer list aligned with moe_layers."""
    if isinstance(experts_per_task, int):
        return [int(experts_per_task)] * len(moe_layers)
    if isinstance(experts_per_task, (list, tuple)):
        if len(experts_per_task) != len(moe_layers):
            raise ValueError(
                f"experts_per_task length {len(experts_per_task)} does not match moe_layers {len(moe_layers)}"
            )
        return [int(v) for v in experts_per_task]
    if isinstance(experts_per_task, dict):
        missing = [name for name in moe_layers if name not in experts_per_task]
        if missing:
            raise KeyError(f"experts_per_task is missing layer keys: {missing}")
        return [int(experts_per_task[name]) for name in moe_layers]
    raise TypeError(
        "experts_per_task must be an int, list/tuple aligned with moe_layers, or dict keyed by moe layer name"
    )


def _shared_experts_per_layer(backbone: str) -> int:
    """Full-MoE baselines require the same expert count on every configured MoE layer."""
    base = COMMON_BACKBONE_CONFIGS[backbone]
    counts = _normalize_experts_per_task(base["experts_per_task"], base["moe_layers"])
    if len(set(counts)) != 1:
        raise ValueError(
            f"{backbone} experts_per_task must be identical across moe_layers for full baselines; got {counts}"
        )
    return counts[0]


# ============================================================
# 通用 backbone 配置
# ============================================================
COMMON_BACKBONE_CONFIGS = {
    "resnet18": {
        "feat_dim": 512,
        "moe_layers": ["layer2", "layer3", "layer4"],
        "moe_channels": {"layer2": 128, "layer3": 256, "layer4": 512},
        "bottleneck_ratio": {"layer2": 2, "layer3": 2, "layer4": 2},
        "experts_per_task": [2, 2, 2],
    },
    "resnet50": {
        "feat_dim": 2048,
        "moe_layers": ["layer2", "layer3", "layer4"],
        "moe_channels": {"layer2": 512, "layer3": 1024, "layer4": 2048},
        "bottleneck_ratio": {"layer2": 8, "layer3": 8, "layer4": 8},
        "experts_per_task": [2, 2, 2],
    },
    "resnet101": {
        "feat_dim": 2048,
        "moe_layers": ["layer2", "layer3", "layer4"],
        "moe_channels": {"layer2": 512, "layer3": 1024, "layer4": 2048},
        "bottleneck_ratio": {"layer2": 8, "layer3": 8, "layer4": 8},
        "experts_per_task": [2, 2, 2],
    },
    "efficientnet_b3": {
        "feat_dim": 1536,
        "moe_layers": ["stage4", "stage5", "stage6"],
        "moe_channels": {"stage4": 96, "stage5": 136, "stage6": 232},
        "bottleneck_ratio": {"stage4": 1, "stage5": 1, "stage6": 1},
        "experts_per_task": [3, 3, 3],
    },
    "convnext_small": {
        "feat_dim": 768,
        "moe_layers": ["stage2", "stage3", "stage4"],
        "moe_channels": {"stage2": 192, "stage3": 384, "stage4": 768},
        "bottleneck_ratio": {"stage2": 2, "stage3": 2, "stage4": 2},
        "experts_per_task": [3, 3, 3],
    },
    "vgg19": {
        "feat_dim": 512,
        "moe_layers": ["stage3", "stage4", "stage5"],
        "moe_channels": {"stage3": 256, "stage4": 512, "stage5": 512},
        "bottleneck_ratio": {"stage3": 2, "stage4": 2, "stage5": 2},
        "experts_per_task": [2, 2, 2],
    },
    "deit_small_patch16_224_in661": {
        "feat_dim": 384,
        "moe_layers": ["stage2", "stage3", "stage4"],
        "moe_channels": {"stage2": 384, "stage3": 384, "stage4": 384},
        "bottleneck_ratio": {"stage2": 4, "stage3": 4, "stage4": 4},
        "experts_per_task": [1, 1, 1],
    },
}

# ============================================================
# PrototypeHead 通用配置
# ============================================================
COMMON_HEAD = {
    "scale": 20.0,
    "imprint_init": True,
    "head_type": "cosine_prototype",  # prototype | cosine_prototype | linear | cos
    "prototype_mu_mode": "learnable",
}

# ============================================================
# 数据路径与预设
# - 可多数据集串联增量：neu_xsdd_magnetic_cr7 / dagm_gc10_kolek_bsd / neu_xsdd_magnetic_cr7_dagm_gc10_kolek_bsd
# - subject_data 单标签数据：cifar10 / deepweeds / dtd / eurosat / fashion_mnist / mnist / office31 / oxford_pet / pacs / patternnet / food11 / gtsrb
# ============================================================
DATA_ROOTS = {
    "neu": os.getenv("CIL_NEU_ROOT", _repo_data_path("neudata_yolo_701515")),
    "xsdd": os.getenv("CIL_XSDD_ROOT", _repo_data_path("xsdd_yolo_cls_701515")),
    "magnetic": os.getenv("CIL_MAGNETIC_ROOT", _repo_data_path("magnetic_yolo_cls_701515")),
    "cr7": os.getenv("CIL_CR7_ROOT", _repo_data_path("CR7-DET_yolo_cls_701515")),
    "gc10": os.getenv("CIL_GC10_ROOT", _repo_data_path("GC10_2300_yolo_cls_701515")),
    "dagm": os.getenv("CIL_DAGM_ROOT", _repo_data_path("DAGM_KaggleUpload_yolo_cls_701515")),
    "kolek": os.getenv("CIL_KOLEK_ROOT", _repo_data_path("KolektorSDD2_yolo")),
    "bsd": os.getenv("CIL_BSD_ROOT", _repo_data_path("BSData-main_yolo")),
    "cifar10": os.getenv("CIL_SUBJECT_CIFAR10_ROOT", _repo_data_path("subject", "cifar10_yolo_cls")),
    "deepweeds": os.getenv("CIL_SUBJECT_DEEPWEEDS_ROOT", _repo_data_path("subject", "deepweeds_yolo_cls")),
    "dtd": os.getenv("CIL_SUBJECT_DTD_ROOT", _repo_data_path("subject", "dtd_yolo_cls")),
    "eurosat": os.getenv("CIL_SUBJECT_EUROSAT_ROOT", _repo_data_path("subject", "EuroSAT_RGB_yolo_cls")),
    "fashion_mnist": os.getenv("CIL_SUBJECT_FASHION_MNIST_ROOT", _repo_data_path("subject", "fashion-mnist-master_yolo_cls")),
    "mnist": os.getenv("CIL_SUBJECT_MNIST_ROOT", _repo_data_path("subject", "MNIST_yolo_cls")),
    "office31": os.getenv("CIL_SUBJECT_OFFICE31_ROOT", _repo_data_path("subject", "office31_yolo_cls")),
    "oxford_pet": os.getenv("CIL_SUBJECT_OXFORD_PET_ROOT", _repo_data_path("subject", "Oxford-IIIT-Pet_yolo_cls")),
    "pacs": os.getenv("CIL_SUBJECT_PACS_ROOT", _repo_data_path("subject", "pacs_yolo_cls")),
    "patternnet": os.getenv("CIL_SUBJECT_PATTERNNET_ROOT", _repo_data_path("subject", "PatternNet_yolo_cls")),
    "food11": os.getenv("CIL_SUBJECT_FOOD11_ROOT", _repo_data_path("subject", "Food-11_yolo_cls")),
    "gtsrb": os.getenv("CIL_SUBJECT_GTSRB_ROOT", _repo_data_path("subject", "GTSRB_yolo_cls")),
}


def _load_yolo_class_names(dataset_root, fallback=None):
    if not os.path.isdir(dataset_root):
        if fallback is not None:
            return list(fallback)
        return []

    classes_path = os.path.join(dataset_root, "classes.txt")
    if os.path.isfile(classes_path):
        names = []
        with open(classes_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                parts = text.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    names.append(parts[1].strip())
                else:
                    names.append(text)
        if names:
            return names

    train_dir = os.path.join(dataset_root, "images", "train")
    if os.path.isdir(train_dir):
        prefixes = set()
        for fname in os.listdir(train_dir):
            stem, _ = os.path.splitext(fname)
            if "__" in stem:
                prefixes.add(stem.split("__", 1)[0])
        if prefixes:
            return sorted(prefixes)

    if fallback is not None:
        return list(fallback)

    if fallback is not None:
        return list(fallback)
    return []


def _single_task_preset(class_names):
    return _preset(class_names, [range(len(class_names))])


def _filter_dataset_order(dataset_order, excluded_keys):
    excluded = {str(key) for key in excluded_keys}
    filtered = [str(key) for key in dataset_order if str(key) not in excluded]
    if not filtered:
        raise ValueError(
            f"All datasets were excluded from order {dataset_order}; "
            f"excluded={sorted(excluded)}"
        )
    return tuple(filtered)


NEU_CLASSES = ["Crazing", "Inclusion", "Patches", "Pitted_Surface", "Rolled-in_Scale", "Scratches"]
XSDD_CLASSES = [
    "finishing_roll_printing",
    "iron_sheet_ash",
    "oxide_scale_of_plate_system",
    "oxide_scale_of_temperature_system",
    "red_iron",
    "slag_inclusion",
    "surface_scratch",
]
MAGNETIC_CLASSES = ["mt_blowhole", "mt_break", "mt_crack", "mt_fray", "mt_free", "mt_uneven"]
CR7_CLASSES = ["dents", "inclusions", "linear", "macular", "oil_spots", "pits", "punching"]
GC10_CLASSES = [f"class{i}" for i in range(1, 11)]
KOLEK_CLASSES = ["normal", "pitting"]
BSD_CLASSES = ["normal", "pitting"]
CIFAR10_CLASSES = _load_yolo_class_names(
    DATA_ROOTS["cifar10"],
    fallback=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
)
DEEPWEEDS_CLASSES = _load_yolo_class_names(DATA_ROOTS["deepweeds"])
DTD_CLASSES = _load_yolo_class_names(DATA_ROOTS["dtd"])
EUROSAT_CLASSES = _load_yolo_class_names(DATA_ROOTS["eurosat"])
FASHION_MNIST_CLASSES = _load_yolo_class_names(DATA_ROOTS["fashion_mnist"])
MNIST_CLASSES = _load_yolo_class_names(DATA_ROOTS["mnist"], fallback=[str(i) for i in range(10)])
OFFICE31_CLASSES = _load_yolo_class_names(DATA_ROOTS["office31"])
OXFORD_PET_CLASSES = _load_yolo_class_names(DATA_ROOTS["oxford_pet"])
PACS_CLASSES = _load_yolo_class_names(DATA_ROOTS["pacs"])
PATTERNNET_CLASSES = _load_yolo_class_names(DATA_ROOTS["patternnet"])
FOOD11_CLASSES = _load_yolo_class_names(DATA_ROOTS["food11"])
GTSRB_CLASSES = _load_yolo_class_names(DATA_ROOTS["gtsrb"])

# subject_data 多数据集串联时可排除若干数据集；[] 表示所有 task 全跑
# 示例:
#   []                         -> 全部 subject dataset session
#   ["pacs", "dtd"]           -> 去掉这 2 个 task
#   ["cifar10", ..., "pacs"]  -> 只剩 patternnet，1 个 session
SUBJECT_MIX_DATASET_KEYS = (
    "cifar10",
    "deepweeds",
    "dtd",
    "eurosat",
    "fashion_mnist",
    "mnist",
    "office31",
    "oxford_pet",
    "pacs",
    "patternnet",
    "food11",
    "gtsrb",
)
SUBJECT_MIX_EXCLUDED_DATASETS = ["dtd", "office31", "pacs", "gtsrb"]  # 修改这里排除不想跑的 dataset
_invalid_subject_mix_excluded = sorted(
    set(str(key) for key in SUBJECT_MIX_EXCLUDED_DATASETS) - set(SUBJECT_MIX_DATASET_KEYS)
)
if _invalid_subject_mix_excluded:
    raise ValueError(
        f"Unknown SUBJECT_MIX_EXCLUDED_DATASETS: {_invalid_subject_mix_excluded}; "
        f"choose from {list(SUBJECT_MIX_DATASET_KEYS)}"
    )


def _preset(class_names, task_splits, *, datasets=None):
    preset = {
        "class_names": list(class_names),
        "task_splits": [list(task) for task in task_splits],
    }
    if datasets is not None:
        preset["datasets"] = [str(name) for name in datasets]
    return preset


def _offset_task_splits(task_splits, offset):
    return [[int(offset) + int(cid) for cid in task] for task in task_splits]


def _reorder_task_splits(task_splits, task_order):
    return [list(task_splits[idx]) for idx in task_order]


def _prefixed_classes(prefix, class_names):
    return [f"{prefix}_{name}" for name in class_names]


BASE_DATASET_PRESETS = {
    "neu": _preset(NEU_CLASSES, [[0, 3], [4, 5], [1, 2]]),
    "xsdd": _preset(XSDD_CLASSES, [[0, 3, 6], [1, 2], [4, 5]]),
    "magnetic": _preset(MAGNETIC_CLASSES, [range(len(MAGNETIC_CLASSES))]),
    "cr7": _preset(CR7_CLASSES, [range(len(CR7_CLASSES))]),
    "gc10": _preset(GC10_CLASSES, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
    "dagm": _preset(GC10_CLASSES, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
    "kolek": _preset(KOLEK_CLASSES, [range(len(KOLEK_CLASSES))]),
    "bsd": _preset(BSD_CLASSES, [range(len(BSD_CLASSES))]),
    "cifar10": _single_task_preset(CIFAR10_CLASSES),
    "deepweeds": _single_task_preset(DEEPWEEDS_CLASSES),
    "dtd": _single_task_preset(DTD_CLASSES),
    "eurosat": _single_task_preset(EUROSAT_CLASSES),
    "fashion_mnist": _single_task_preset(FASHION_MNIST_CLASSES),
    "mnist": _single_task_preset(MNIST_CLASSES),
    "office31": _single_task_preset(OFFICE31_CLASSES),
    "oxford_pet": _single_task_preset(OXFORD_PET_CLASSES),
    "pacs": _single_task_preset(PACS_CLASSES),
    "patternnet": _single_task_preset(PATTERNNET_CLASSES),
    "food11": _single_task_preset(FOOD11_CLASSES),
    "gtsrb": _single_task_preset(GTSRB_CLASSES),
}


def _compose_incremental_chain(*dataset_keys):
    class_names, task_splits, offset = [], [], 0
    for key in dataset_keys:
        preset = BASE_DATASET_PRESETS[str(key)]
        names = list(preset["class_names"])
        class_names.extend(names)
        task_splits.extend(_offset_task_splits(preset["task_splits"], offset))
        offset += len(names)
    return _preset(class_names, task_splits)


def _compose_dataset_chain(*dataset_keys):
    class_names, task_splits, offset = [], [], 0
    for key in dataset_keys:
        names = list(BASE_DATASET_PRESETS[str(key)]["class_names"])
        class_names.extend(names)
        task_splits.append(list(range(offset, offset + len(names))))
        offset += len(names)
    return _preset(class_names, task_splits, datasets=dataset_keys)


def _compose_prefixed_dataset_chain(*dataset_keys):
    class_names, task_splits, offset = [], [], 0
    for key in dataset_keys:
        names = _prefixed_classes(str(key), BASE_DATASET_PRESETS[str(key)]["class_names"])
        class_names.extend(names)
        task_splits.append(list(range(offset, offset + len(names))))
        offset += len(names)
    return _preset(class_names, task_splits, datasets=dataset_keys)


def _compose_dataset_chain_with_prefixes(dataset_keys, prefixed_keys):
    prefixed_keys = {str(key) for key in prefixed_keys}
    class_names, task_splits, offset = [], [], 0
    for key in dataset_keys:
        key = str(key)
        names = list(BASE_DATASET_PRESETS[key]["class_names"])
        if key in prefixed_keys:
            names = _prefixed_classes(key, names)
        class_names.extend(names)
        task_splits.append(list(range(offset, offset + len(names))))
        offset += len(names)
    return _preset(class_names, task_splits, datasets=dataset_keys)


def _register_dataset_order_presets(preset_prefix, dataset_orders, compose_fn):
    for idx, dataset_order in enumerate(dataset_orders, start=1):
        dataset_order = tuple(str(key) for key in dataset_order)
        order_name = f"{preset_prefix}_order{idx}"
        DATASET_PRESETS[order_name] = compose_fn(*dataset_order)
        DATASET_PRESETS["_".join(dataset_order)] = deepcopy(DATASET_PRESETS[order_name])


DATASET_PRESETS = deepcopy(BASE_DATASET_PRESETS)
DATASET_PRESETS["neu_xsdd"] = _compose_dataset_chain("neu", "xsdd")
DATASET_PRESETS["dagm_gc10"] = _preset(
    [*_prefixed_classes("dagm", GC10_CLASSES), *_prefixed_classes("gc10", GC10_CLASSES)],
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
    datasets=["dagm", "gc10"],
)


_register_dataset_order_presets(
    "neu_xsdd_magnetic_cr7",
    [
        ("neu", "xsdd", "magnetic", "cr7"),
        ("cr7", "xsdd", "neu", "magnetic"),
        ("magnetic", "neu", "cr7", "xsdd"),
        ("xsdd", "magnetic", "cr7", "neu"),
        ("cr7", "magnetic", "xsdd", "neu"),
    ],
    _compose_dataset_chain,
)

_register_dataset_order_presets(
    "dagm_gc10_kolek_bsd",
    [
        ("dagm", "gc10", "kolek", "bsd"),
        ("bsd", "gc10", "dagm", "kolek"),
        ("kolek", "dagm", "bsd", "gc10"),
        ("gc10", "kolek", "bsd", "dagm"),
        ("bsd", "kolek", "gc10", "dagm"),
    ],
    _compose_prefixed_dataset_chain,
)

_register_dataset_order_presets(
    "neu_xsdd_magnetic_cr7_dagm_gc10_kolek_bsd",
    [
        ("neu", "xsdd", "magnetic", "cr7", "dagm", "gc10", "kolek", "bsd"),
        ("dagm", "neu", "gc10", "xsdd", "kolek", "magnetic", "bsd", "cr7"),
        ("xsdd", "bsd", "magnetic", "dagm", "cr7", "kolek", "neu", "gc10"),
        ("gc10", "cr7", "neu", "kolek", "xsdd", "dagm", "magnetic", "bsd"),
        ("magnetic", "kolek", "dagm", "neu", "bsd", "xsdd", "gc10", "cr7"),
    ],
    lambda *dataset_order: _compose_dataset_chain_with_prefixes(
        dataset_order,
        prefixed_keys=("dagm", "gc10", "kolek", "bsd"),
    ),
)

for idx, dataset_order in enumerate(
    [
        SUBJECT_MIX_DATASET_KEYS,
        ("patternnet", "mnist", "office31", "pacs", "cifar10", "oxford_pet", "dtd", "deepweeds", "fashion_mnist", "eurosat", "food11", "gtsrb"),
        ("deepweeds", "oxford_pet", "fashion_mnist", "patternnet", "office31", "dtd", "pacs", "cifar10", "eurosat", "mnist", "gtsrb", "food11"),
        ("pacs", "eurosat", "mnist", "office31", "dtd", "patternnet", "deepweeds", "oxford_pet", "cifar10", "fashion_mnist", "food11", "gtsrb"),
        ("fashion_mnist", "patternnet", "deepweeds", "office31", "cifar10", "pacs", "mnist", "eurosat", "oxford_pet", "dtd", "gtsrb", "food11"),
    ],
    start=1,
):
    filtered_order = _filter_dataset_order(dataset_order, SUBJECT_MIX_EXCLUDED_DATASETS)
    DATASET_PRESETS[f"subject_mix_order{idx}"] = _compose_prefixed_dataset_chain(*filtered_order)

DATASET_PRESETS["subject_mix"] = deepcopy(DATASET_PRESETS["subject_mix_order1"])

# 修改这里即可切换:
#   "neu_xsdd_magnetic_cr7_order1..5"
#   "dagm_gc10_kolek_bsd_order1..5"
#   "neu_xsdd_magnetic_cr7_dagm_gc10_kolek_bsd_order1..5"
#   "subject_mix_order1..5"  # cifar10 / deepweeds / dtd / eurosat / fashion_mnist / mnist / office31 / oxford_pet / pacs / patternnet / food11 / gtsrb
ACTIVE_DATASET = os.getenv("CIL_ACTIVE_DATASET", "neu_xsdd")


def _build_data_config(active_dataset: str):
    if active_dataset not in DATASET_PRESETS:
        raise KeyError(f"Unknown ACTIVE_DATASET: {active_dataset}")

    preset = deepcopy(DATASET_PRESETS[active_dataset])
    if "datasets" in preset:
        dataset_keys = [str(k) for k in preset["datasets"]]
        data_root = [DATA_ROOTS[k] for k in dataset_keys]
        dataset_class_names = [
            DATASET_PRESETS[k]["class_names"] for k in dataset_keys
        ]
    elif active_dataset == "neu_xsdd":
        data_root = [DATA_ROOTS["neu"], DATA_ROOTS["xsdd"]]
        dataset_class_names = [
            DATASET_PRESETS["neu"]["class_names"],
            DATASET_PRESETS["xsdd"]["class_names"],
        ]
    else:
        data_root = DATA_ROOTS[active_dataset]
        dataset_class_names = [
            DATASET_PRESETS[active_dataset]["class_names"],
        ]

    return {
        "data_root": data_root,
        "dataset_class_names": dataset_class_names,
        "image_size": 224,
        "num_classes": len(preset["class_names"]),
        "class_names": preset["class_names"],
        "task_splits": preset["task_splits"],
        "num_workers": int(os.getenv("CIL_NUM_WORKERS", "0")),
        "seed": int(os.getenv("CIL_SEED", "42")),
        # repeats=1: 仅 base_seed；repeats=3: base_seed, base_seed+1, base_seed+2
        "repeats": int(os.getenv("CIL_REPEATS", "3")),
    }


DATA = _build_data_config(ACTIVE_DATASET)


def experiment_seeds(base_seed=None, repeats=None):
    """Return the list of random seeds for one experiment run."""
    base = int(base_seed if base_seed is not None else DATA["seed"])
    n = int(repeats if repeats is not None else DATA.get("repeats", 1))
    if n <= 1:
        return [base]
    return [base + i for i in range(n)]
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)  # .../Incre_neu
BASELINES_ROOT = os.path.join(PROJECT_ROOT, "baselines_0610")
BASELINES_ROOT = os.getenv("CIL_BASELINES_ROOT", BASELINES_ROOT)

# 统一外部仓库默认路径（可按本机实际位置修改）
PYCIL_REPO_DIR = os.path.join(BASELINES_ROOT, "PyCIL")
MRFA_REPO_DIR = os.path.join(BASELINES_ROOT, "MRFA_ICML2024")
TAGFEX_REPO_DIR = os.path.join(BASELINES_ROOT, "TagFex_CVPR2025")
DER_REPO_DIR = os.path.join(BASELINES_ROOT, "DER-ClassIL.pytorch")
BEEF_REPO_DIR = os.path.join(BASELINES_ROOT, "ICLR23-BEEF")
SEED_PAPER_REPO_DIR = os.path.join(BASELINES_ROOT, "SEED_official_github")
TPL_REPO_DIR = os.path.join(BASELINES_ROOT, "TPL")
GENCLASSIFIER_REPO_DIR = os.path.join(BASELINES_ROOT, "GenClassifier")
PEC_REPO_DIR = os.path.join(BASELINES_ROOT, "PEC")
MOVE_REPO_DIR = os.path.join(BASELINES_ROOT, "MoVE")
MORE_OFFICIAL_REPO_DIR = os.path.join(BASELINES_ROOT, "MORE_official_github")
MORE_PAPER_REPO_DIR = MORE_OFFICIAL_REPO_DIR
ITAML_PAPER_REPO_DIR = os.path.join(BASELINES_ROOT, "iTAML_official_github")
DIVA_REPO_DIR = os.path.join(BASELINES_ROOT, "DIVA")
SEMA_REPO_DIR = os.path.join(BASELINES_ROOT, "SEMA")
MOEADAPTERSPP_REPO_DIR = os.path.join(BASELINES_ROOT, "MOEAdaptersPP")
MFGR_PAPER_REPO_DIR = os.path.join(BASELINES_ROOT, "MFGR_official_github")
GFRIL_PAPER_REPO_DIR = os.path.join(BASELINES_ROOT, "GFR-IL_official_github")
BUILD_REPO_DIR = os.path.join(BASELINES_ROOT, "BUILD")


def _common_backbone_model(backbone: str, *, pretrained: bool = True, scale: float = COMMON_HEAD["scale"], **extra):
    base = deepcopy(COMMON_BACKBONE_CONFIGS[backbone])
    model = {
        "backbone": backbone,
        "pretrained": pretrained,
        "feat_dim": base["feat_dim"],
        "moe_layers": base["moe_layers"],
        "moe_channels": base["moe_channels"],
        "bottleneck_ratio": base["bottleneck_ratio"],
        "scale": scale,
    }
    model.update(extra)
    return model


# ============================================================
# 全局运行开关
#   "incremental_1"  — 只跑增量1
#   "incremental_2"  — 只跑增量2
#   "incremental_3"  — 只跑增量3
#   "full_1"         — 只跑全量基线1 (gate 自由路由)
#   "full_2"         — 只跑全量基线2 (专家固定分配)
#   "all"            — 跑当前工程内置四个方法（incremental_1/2 + full_1/2）

#   "incre_ewc"      — 外部方法: EWC #参数约束2017
#   "incre_icarl"    — 外部方法: iCaRL #知识蒸馏 + 真实样本回放2017
#   "incre_mrfa"     — 外部方法: MRFA #对icarl的改造，真实样本回放2024
#   "incre_beef"     — 外部方法: BEEF #动态结构+真实回放2023
#   "gfril_paper"    — 外部方法: GFR-IL 论文包接入版 #generative feature replay 2020，当前任务 val 早停
#   "incre_diva"     — 外部方法: DiVA #discriminative VAE generative replay 2020
#   "move_paper"    — 外部方法: MoVE/HVCL 论文对齐版 #raw-image sparse MoVE layers + HVCL-style regularization
#   "itaml_paper"    — 外部方法: iTAML 原 GitHub 包接入版 #总回放预算600，当前任务 val 早停

#   "incre_genclassifier" — 外部方法: Generative Classifier #每类 VAE + Bayes 推理 (strict CIL) 2021 CVPRW
#   "mfgr_paper"     — 外部方法: MFGR 论文包接入版 #memory-free generative replay 2021，当前任务 val 早停

#   "more_paper"  _resnet18   — 外部方法: MORE/MORA 论文接入版 #总回放预算600，当前任务 val 早停
#   "build_paper"   — 外部方法: BUILD 论文接入版 #buffer-free multi-head OOD task-id prediction 2025
#   "incre_tpl"     — 外部方法: TPL #Likelihood Ratio Task Prediction 2024+真实回放 ICLR。transfomer。

#   "der_paper"      — 外部方法: DER #动态结构+真实回放2021
#   "incre_tagfex"   — 外部方法: TagFex #动态结构+真实回放2025
#   "seed_paper"     — 外部方法: SEED 原 GitHub 包接入版 #无样本专家集成，不回放，当前任务 val 早停
#   "incre_sema"     — 外部方法: SEMA 本地复刻版 不回放
#   "incre_pec"     — 外部方法: PEC #Prediction Error-based Classification，不判断taskid+不回放 2024 ICLR
#   *"moeadapterspp_paper" — 外部方法: MoE-Adapters++ #动态 MoE adapters + LEAS/DEeC+不回放 2025 TPAMI。transfomer。
# ============================================================

RUN_MODE = os.getenv("RUN_MODE", "incremental_2")

# 复现性：main.py 默认开启（CIL_DETERMINISTIC=1），并为所有外部子进程注入环境变量。
# 关闭严格确定性（更快、数值可能漂移）： export CIL_DETERMINISTIC=0
# 可选：CUBLAS_WORKSPACE_CONFIG=:4096:8  CIL_OMP_NUM_THREADS=1  CIL_MKL_NUM_THREADS=1
CIL_DETERMINISTIC = os.getenv("CIL_DETERMINISTIC", "1").strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_choice(name: str, default: str, choices: tuple) -> str:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw not in choices:
        raise ValueError(f"Unsupported {name}={raw!r}; choose one of {choices}")
    return raw


STRICT_REPRODUCIBILITY_MODES = ("all", "internal_fast")

# 效率统计开关（写入 cost.txt / efficiency.json）
# - profile_flops=True：用 thop/公式估算 flops_*（较慢）
# - profile_flops=False：不统计算力，仅 params / gpu_peak / time
# 环境变量：export CIL_PROFILE_FLOPS=0
#
# strict_reproducibility — 严格可复现（bootstrap / deterministic algorithms / TF32 / OMP=1）
#   all            — 内外部均严格（默认，与 20260518 现行为一致）
# 环境变量：export CIL_STRICT_REPRO=internal_fast
# 全局关闭严格模式：export CIL_DETERMINISTIC=0
EFFICIENCY = {
    "profile_flops": _env_bool("CIL_PROFILE_FLOPS", True),
    "strict_reproducibility": _env_choice(
        "CIL_STRICT_REPRO", "internal_fast", STRICT_REPRODUCIBILITY_MODES
    ),
}


def cil_deterministic_enabled() -> bool:
    return _env_bool("CIL_DETERMINISTIC", True)


def internal_strict_reproducibility_enabled() -> bool:
    if not cil_deterministic_enabled():
        return False
    mode = str(EFFICIENCY.get("strict_reproducibility", "all")).strip().lower()
    return mode == "all"


def external_strict_reproducibility_enabled() -> bool:
    """Strict repro for external baseline subprocesses (DER, iCaRL, …)."""
    return cil_deterministic_enabled()


def apply_internal_run_seed(seed: int) -> None:
    import random

    import numpy as np

    seed_i = int(seed)
    if internal_strict_reproducibility_enabled():
        try:
            from deterministic import apply_torch_deterministic

            apply_torch_deterministic(seed_i)
        except ImportError:
            random.seed(seed_i)
            np.random.seed(seed_i)
            torch.manual_seed(seed_i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_i)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        return

    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)


def _default_increment_step(task_splits):
    if len(task_splits) <= 1:
        return len(task_splits[0]) if task_splits else 0
    later_sizes = [len(x) for x in task_splits[1:] if len(x) > 0]
    return min(later_sizes) if later_sizes else len(task_splits[0])


# ============================================================
# 各外部方法公平对比超参数（逐方法独立配置，写入 train["fair"]）
# ============================================================
# ResNet 骨干（PyCIL / MRFA / TagFex）
FAIR_BACKBONE_CHOICES = ("resnet18", "resnet50")
_FAIR_BACKBONE = os.getenv("CIL_FAIR_BACKBONE", "resnet18").strip().lower()
if _FAIR_BACKBONE not in FAIR_BACKBONE_CHOICES:
    raise ValueError(
        f"Unsupported CIL_FAIR_BACKBONE={_FAIR_BACKBONE!r}; choose one of {FAIR_BACKBONE_CHOICES}"
    )


_FAIR_DATA_AUG = {
    "train": ["random_resized_crop", "horizontal_flip", "color_jitter", "normalize"],
    "test": ["resize", "center_crop", "normalize"],
}


def _fair_launcher_common():
    return {
        "seed": DATA["seed"],
        "image_size": DATA["image_size"],
        "class_names": deepcopy(DATA["class_names"]),
        "task_splits": deepcopy(DATA["task_splits"]),
        "init_cls": len(DATA["task_splits"][0]) if DATA["task_splits"] else 0,
        "increment": _default_increment_step(DATA["task_splits"]),
        "batch_size": 32,
        "num_workers": DATA["num_workers"],
        "early_stop_patience": 30,
        "early_stop_min_delta": 1e-4,
    }


# ---------------------------------------------------------------------------
# 各方法完整超参（与 baselines/*/exps 或模型默认一致；main.py 补丁写入 baseline 配置）
# 标注「# hard coding」表示改此处不会影响实际训练，需改对应 baseline 源码。
# PyCIL 系 EWC/iCaRL/DER：lr/wd/milestones/方法损失系数多在 models/*.py 模块常量中。
# ---------------------------------------------------------------------------

# MRFA — baselines/MRFA_ICML2024/exps/icarl/mrfa/neu_xsdd.json
FAIR_MRFA = {
    **_fair_launcher_common(),
    "memory_budget": 600,
    "backbone": _FAIR_BACKBONE,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": 4,
    "optimizer": "sgd",  # hard coding: 仅文档；MRFA 从 args['lr'] 读入后仍用模块 milestones
    "lr": 0.1,  # 经 patched JSON→args 生效（icarl_mrfa.__init__ 会读 args['lr']）
    "weight_decay": 5e-4,  # 经 patched JSON→args 生效
    "gamma": 0.1,  # 经 patched JSON→args 生效（用作 lr_decay）
    "scheduler": "multistep",  # hard coding: 仅文档；milestones 仍为 icarl_mrfa.py 默认
    "init_epochs": 100,
    "epochs": 80,
    "early_stop_patience": 20,
    "early_stop_min_delta": 1e-4,
    "topk": 1,
    "replay_percent": 0.05,
    "pretrained": True,
    "fixed_memory": False,
    "memory_per_class": 0,
    "perturb_p": [1e-4, 1e-4, 1e-4, 1e-4],
    "num_augmem": 1,
    "disable_perturb": False,
    "auto_kd": False,
    "model_name": "icarl_mrfa",
}

# TagFex — baselines/TagFex_CVPR2025/configs/all_in_one/neu_xsdd_tagfex_resnet18.yaml
FAIR_TAGFEX = {
    **_fair_launcher_common(),
    "memory_budget": 600,
    "backbone": _FAIR_BACKBONE,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": 4,
    "replay_percent": 0.05,
    "pretrained": True,
    "yaml_overrides": {
        "init_epochs": 100,
        "inc_epochs": 80,
        "eval_interval": 5,
        "early_stop_patience": 30,
        "early_stop_min_delta": 1e-4,
        "contrast_factor": 1,
        "contrast_kd_factor": 2,
        "aux_factor": 2,
        "trans_cls_factor": 1,
        "transfer_factor": 1,
        "grad_clip_norm": 5.0,
        "infonce_temp": 0.2,
        "infonce_kd_temp": 0.2,
        "kd_temp": 2,
        "num_aug": 2,
        "amp": True,
        "memory_configs": {
            "fixed_size": False,
            "memory_size": 600,
            "replay_percent": 0.05,
        },
        "trainloader_params": {"batch_size": 32, "num_workers": 4, "drop_last": False},
        "testloader_params": {"batch_size": 64, "num_workers": 4, "drop_last": False},
        "init_optimizer_configs": {
            "name": "sgd",
            "params": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
        },
        "init_scheduler_configs": {
            "name": "multistep",
            "params": {"milestones": [15, 22, 27], "gamma": 0.1},
        },
        "inc_optimizer_configs": {
            "name": "sgd",
            "params": {"lr": 0.01, "momentum": 0.9, "weight_decay": 2e-4},
        },
        "inc_scheduler_configs": {
            "name": "multistep",
            "params": {"milestones": [15, 22, 27], "gamma": 0.1},
        },
        "network_configs": {
            "classifier_type": "unified",
            "proj_hidden_dim": 2048,
            "proj_output_dim": 1024,
            "init_from_last": False,
            "init_from_interpolation": True,
            "init_interpolation_factor": 0.95,
            "attn_num_heads": 8,
            "merge_attn": True,
        },
        "backbone_configs": {
            "params": {
                "pretrained": True,
            },
        },
    },
}

# DER — DER-ClassIL.pytorch 官方包接入版
FAIR_DER_PAPER = {
    **_fair_launcher_common(),
    "memory_budget": 600,
    "backbone": _FAIR_BACKBONE,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": 4,
    "paper_source_dir": "DER-ClassIL.pytorch",
    "pretrained": False,  # 官方 DER BasicNet 随机初始化；不下载外部预训练权重
    "optimizer": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "gamma": 0.1,  # hard coding: lrate_decay 模块常量
    "scheduler": "multistep",  # hard coding
    "init_epochs": 100,
    "epochs": 80,
    "early_stop_patience": 20,
    "early_stop_min_delta": 1e-4,
    "topk": 1,
    "T": 2,  # hard coding: der.py 未使用
    "replay_percent": 0.05,
    "fixed_memory": False,
    "val_ratio": 0.0,
    "aux_loss_weight": 1.0,
    "mask_reg_weight": 1e-4,
    "mask_threshold": 0.5,
    "mask_scale": 10.0,
    "finetune_epochs": 50,
    "finetune_lr": 0.05,
    "finetune_weight_decay": 5e-4,
    "finetune_patience": 15,
    "grad_clip_norm": 5.0,
    "nf": 64,
    "model_name": "der_paper",
    "official_components": {
        "dynamic_representation_expansion": True,
        "auxiliary_old_vs_new_classifier": True,
        "classifier_balanced_finetune": True,
        "masking_pruning": "feature-channel binary gate added in run_cil_paper.py because local official configs are wo_mask/wo_prune",
    },
}

# BEEF — PyCIL exps/beef.json
FAIR_BEEF = {
    **_fair_launcher_common(),
    "memory_budget": 600,
    "backbone": _FAIR_BACKBONE,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": 8,
    "optimizer": "sgd",  # hard coding: 仅文档；beef_iso 用 CosineAnnealingLR
    "init_lr": 0.1,
    "lr": 0.1,
    "init_weight_decay": 5e-4,
    "weight_decay": 5e-4,
    "gamma": 0.1,  # hard coding: BEEF 未使用
    "scheduler": "multistep",  # hard coding: 仅文档
    "init_epochs": 100,
    "expansion_epochs": 80,
    "fusion_epochs": 30,
    "early_stop_patience": 30,
    "early_stop_min_delta": 1e-4,
    "topk": 1,
    "T": 2,  # hard coding: beef_iso.py 未从 args 读取 T
    "replay_percent": 0.05,
    "pretrained": True,
    "fixed_memory": False,
    "memory_per_class": 0,
    "logits_alignment": 1.7,
    "energy_weight": 0.01,
    "is_compress": False,
    "reduce_batch_size": False,
    "model_name": "beefiso",
}

# EWC — PyCIL models/ewc.py
FAIR_EWC = {
    **_fair_launcher_common(),
    "memory_budget": 0,
    "backbone": _FAIR_BACKBONE,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": 4,
    "optimizer": "sgd",  # hard coding
    "lr": 0.1,  # hard coding: ewc.py 用 init_lr/lrate 模块常量
    "momentum": 0.9,  # hard coding
    "weight_decay": 5e-4,  # hard coding
    "gamma": 0.1,  # hard coding
    "scheduler": "multistep",  # hard coding
    "init_epochs": 200,
    "epochs": 150,
    "early_stop_patience": 20,
    "early_stop_min_delta": 1e-4,
    "topk": 1,
    "T": 2,  # hard coding: ewc.py 未使用
    "replay_percent": 0.0,
    "pretrained": True,
    "val_ratio": 0.0,
    "model_name": "ewc",
    "lambda_ewc": 1000,  # hard coding: ewc.py lamda 模块常量
    "fishermax": 1e-4,  # hard coding: ewc.py fishermax 模块常量
}

# iCaRL — PyCIL models/icarl.py
FAIR_ICARL = {
    **_fair_launcher_common(),
    "memory_budget": 600,
    "backbone": _FAIR_BACKBONE,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": 4,
    "optimizer": "sgd",  # hard coding
    "lr": 0.1,  # hard coding: icarl.py 用 init_lr/lrate 模块常量
    "momentum": 0.9,  # hard coding
    "weight_decay": 2e-4,  # hard coding
    "gamma": 0.1,  # hard coding
    "scheduler": "multistep",  # hard coding
    "init_epochs": 200,
    "epochs": 150,
    "early_stop_patience": 20,
    "early_stop_min_delta": 1e-4,
    "topk": 1,
    "T": 2,  # hard coding: icarl.py KD 温度模块常量
    "replay_percent": 0.05,
    "pretrained": True,
    "fixed_memory": False,
    "val_ratio": 0.0,
    "model_name": "icarl",
}

# seed_paper — downloaded SEED source + NEU/XDDD adapter; exemplar-free, so no replay buffer.
FAIR_SEED_PAPER = {
    **_fair_launcher_common(),
    "memory_budget": 0,
    # SEED 专家网络（resnet_linear_turbo）
    "backbone": "resnet18",
    "network": "resnet18",
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": DATA["num_workers"],
    "nepochs": 100,
    "ftepochs": 50,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "clipping": 1.0,
    "alpha": 0.99,
    "tau": 3.0,
    "max_experts": 10,
    "gmms": 1,
    "shared": 1,
    "use_multivariate": True,
    "extra_aug": "",#fetril | none
    "pretrained": True,
    "early_stop_patience": 20,
    "early_stop_min_delta": 1e-4,
    "paper_source_dir": "SEED_official_github",
}

# TPL — baselines/TPL（ICLR 2024; HAT + Likelihood-Ratio Task Prediction）
FAIR_TPL = {
    **_fair_launcher_common(),
    "memory_budget": 600,
    "backbone": "deit_small_patch16_224_in661",  # 论文/官方代码明确使用 DeiT-small in661
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": 4,  # hard coding: TPL main.py/eval.py DataLoader 里写死 8
    "visual_encoder": "deit_small_patch16_224_in661",
    "pretrained": True,
    "pretrained_dir": "",
    "optimizer": "sgd_hat",  # approaches/train.py 使用 SGD_hat + momentum/nesterov
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "momentum": 0.9,  # hard coding
    "nesterov": True,  # hard coding
    "epochs": 40,
    "early_stop_patience": 15,
    "early_stop_min_delta": 1e-4,
    # HAT / task-prediction related hyperparameters.
    # K controls the number of nearest replay/task features used by the TPL evaluator.
    # smax and thres_cosh control HAT mask sharpening / compensation.
    "K": 5,
    "smax": 400,
    "clipgrad": 1.0,
    "thres_cosh": 50,
    "alpha": 0.2,
    "latent": 64,
    # Replay memory. "percent" keeps replay_percent per old class, capped by replay_buffer_size.
    # "fixed_buffer" allocates replay_buffer_size across seen classes.
    "replay_mode": "percent",
    "replay_percent": 0.05,
    "replay_buffer_size": 600,
    "replay_batch_size": 32,
    "warmup_ratio": 0.0,
    "gradient_accumulation_steps": 1,
    "lr_scheduler_type": "cosine",
    "num_warmup_steps": 0,
    "sequence_file": "",
    "baseline_name": "tpl_deit_small_patch16_224_in661_neu_xddd_hat",
    "base_dir": "",
}

# PEC — baselines/PEC（ICLR 2024; per-class student-teacher prediction error classification）
FAIR_PEC = {
    **_fair_launcher_common(),
    "memory_budget": 0,
    "backbone": "pec_cnn",  # hard coding: PEC 官方方法使用专用轻量 CNN / MLP，不使用预训练 backbone
    "pretrained": True,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 8,
    "num_workers": DATA["num_workers"],
    "epochs": 40,  # 参照当前外部方法统一 session 训练轮数；官方在线脚本常用单遍 1 epoch
    "early_stop_patience": 30,  # hard coding: 官方 PEC 训练循环未实现 early stopping
    "early_stop_min_delta": 1e-4,  # hard coding: 当前仅作公平配置占位
    "lr": 1e-3,
    "optimizer": "adam",
    "lr_scheduler": "linear",
    "force_no_augmentations": True,
    "pec_architecture": "cnn",
    "pec_num_layers": 2,
    "pec_width": 40,
    "pec_teacher_width_multiplier": 4,
    "pec_output_dim": 172,
    "pec_activation": "relu",
    "pec_normalize_layers": True,
    "pec_conv_layers": "(40, 3, 1)",
    "pec_conv_reduce_spatial_to": 4,
    "pec_train_chunk_size": 4,
}

FAIR_MOVE_PAPER = {
    **_fair_launcher_common(),
    "memory_budget": 0,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": DATA["num_workers"],
    "optimizer": "adam",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "epochs": 40,
    "early_stop_patience": 20,
    "early_stop_min_delta": 1e-4,
    "num_experts": 2,
    "top_k": 1,
    "kernel_width": 10.0,
    "hidden_dim": 256,
    "dropout": 0.0,
    "prior_kl_weight": 0.75,
    "gate_prior_kl_weight": 0.002,
    "gate_entropy_weight": 0.01,
    "expert_diversity_weight": 0.01,
    "distill_weight": 0.0,
    "distill_temperature": 1.0,
    "grad_clip_norm": 5.0,
    "generator_hidden_dim": 256,
    "generator_z_dim": 64,
    "generator_epochs": 80,
    "generator_early_stop_patience": 15,
    "generator_early_stop_min_delta": 1e-4,
    "generator_beta_kl": 1.0,
    "generator_recon_weight": 1.0,
    "generator_replay_per_class": 600,
    "generator_replay_temperature": 1.0,
}

# more_paper — official-github-based MORE entry; fixed global memory budget 600 and 5% per-class sampling.
FAIR_MORE_PAPER_DEIT = {
    **_fair_launcher_common(),
    "memory_budget": 600,
    # Default paper-style backbone."deit_small_patch16_224_in661"/"resnet18"
    "backbone": "deit_small_patch16_224_in661",
    "pretrained": True,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": DATA["num_workers"],
    "optimizer": "sgd",
    "epochs": 40,
    "early_stop_patience": 15,
    "early_stop_min_delta": 1e-3,
    "lr": 1e-3,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "scheduler": "none",
    "hidden_dim": 256,
    "dropout": 0.1,
    # Fixed memory buffer; samples are used as OOD data, not ordinary replay.
    "replay_percent": 0.05,
    "back_update": True,
    "back_update_epochs": 10,
    "back_update_lr": 0.01,
    "back_update_batch_size": 16,
    "distance_scale": 20.0,
    "use_distance_coeff": True,
    "grad_clip_norm": 5.0,
    "adapter_dim": 64,
    "smax": 500,
    "reg_lambda": 0.75,
    "paper_source_dir": "MORE_official_github",
}

FAIR_MORE_PAPER_RESNET18 = {
    **deepcopy(FAIR_MORE_PAPER_DEIT),
    "backbone": "resnet18",
}

# Keep the existing name as the default launch profile so current MORE runs still use DeiT-S/16.
FAIR_MORE_PAPER = deepcopy(FAIR_MORE_PAPER_DEIT)

# BUILD — Buffer-free CIL with post-hoc OOD task-id prediction (no memory/replay).
FAIR_BUILD_PAPER = {
    **_fair_launcher_common(),
    "memory_budget": 0,
    "backbone": "deit_small_patch16_224_in661",
    "pretrained": True,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": DATA["num_workers"],
    "optimizer": "sgd",
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "scheduler": "none",
    "epochs": 40,
    "early_stop_patience": 15,
    "early_stop_min_delta": 1e-4,
    "adapter_dim": 64,
    "dropout": 0.1,
    "smax": 500,
    "mask_reg_weight": 1e-4,
    "detector": "base",
    "scorer": "smmd",
    "react_percentile": 90,
    "dice_percentile": 85,
    "scale_percentile": 85,
    "md_scale": 20.0,
    "md_ridge": 1e-3,
    "grad_clip_norm": 5.0,
}

# itaml_paper — downloaded iTAML source + NEU/XDDD adapter; fixed global memory budget 600 and 5% per-class sampling.
FAIR_ITAML_PAPER = {
    **_fair_launcher_common(),
    "memory_budget": 600,
    "backbone": _FAIR_BACKBONE,
    "pretrained": True,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": DATA["num_workers"],
    "optimizer": "radam",
    "epochs": 80,
    "early_stop_patience": 15,
    "early_stop_min_delta": 1e-4,
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "scheduler": "milestone",  # paper: lr schedule at 20/40/60; local adapter keeps optimizer API simple
    "hidden_dim": 512,
    "embed_dim": 256,
    "dropout": 0.1,
    "replay_percent": 0.05,
    "inner_steps": 1,
    "beta": 1.0,
    # Paper predicts task from a same-task continuum; use 10 to better match the original inference assumption.
    "continuum_size": 10,
    "grad_clip_norm": 5.0,
    "paper_source_dir": "iTAML_official_github",
}

# SEMA — Self-Expansion of Pre-trained Models with Mixture of Adapters
FAIR_SEMA = {
    **_fair_launcher_common(),
    "memory_budget": 0,
    # Paper fixes a frozen ImageNet-1K pretrained ViT-B/16 backbone.
    "backbone": "vit_b_16",
    "pretrained": True,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": DATA["num_workers"],
    "optimizer": "sgd",
    "adapter_lr": 0.005,
    "rd_lr": 0.01,
    "weight_decay": 0.0,
    "scheduler": "cosine",
    # Paper does not lock a per-task epoch count in the main text; keep the
    # local adapter on a mid/heavy external-method budget.
    "epochs": 100,
    "early_stop_patience": 30,
    "early_stop_min_delta": 1e-4,
    "adapter_hidden_dim": 16,
    "rd_hidden_dim": 128,
    "rd_bottleneck_dim": 64,
    "expansion_layers": [9, 10, 11],  # paper default: last three transformer layers
    "expansion_threshold": 1.0,
    "expansion_min_fraction": 0.05,
    "rd_loss_weight": 1.0,
    "classifier_scale": 20.0,
    "grad_clip_norm": 5.0,
}

# MoE-Adapters++ — Dynamic MoE adapters + LEAS/DEeC local paper-aligned adapter.
FAIR_MOEADAPTERSPP = {
    **_fair_launcher_common(),
    "memory_budget": 0,
    "backbone": "clip_vit_b_16",
    "pretrained": True,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": DATA["num_workers"],
    "optimizer": "adamw",
    "lr": 1e-3,
    "weight_decay": 0.0,
    "epochs": 25,
    "early_stop_patience": 10,
    "early_stop_min_delta": 1e-4,
    "router_hidden_dim": 256,
    "expert_hidden_dim": 16,
    "ae_hidden_dim": 256,
    "ae_bottleneck_dim": 64,
    "recognition_layer": 6,
    "subsequent_layers": [7, 8, 9, 10, 11],
    "initial_experts": 2,
    "top_k": 2,
    "expansion_threshold": 1.0,
    "expansion_min_fraction": 0.05,
    "leas_loss_weight": 1.0,
    "deec_loss_weight": 1.0,
    "classifier_scale": 20.0,
    "grad_clip_norm": 5.0,
    "label_smoothing": 0.0,
    "paper_mode": 1,
    "paper_warmup_epochs": 1,
    "paper_preference_window": 10,
}
# DiVA — Discriminative VAE for Continual Learning with Generative Replay
FAIR_DIVA = {
    **_fair_launcher_common(),
    "memory_budget": 0,
    "backbone": "wide_resnet_style",
    "pretrained": True,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": DATA["num_workers"],
    "optimizer": "adam",
    "lr": 1e-3,
    "weight_decay": 0.0,
    "epochs": 80,
    "early_stop_patience": 20,
    "early_stop_min_delta": 1e-4,
    "hidden_dim": 512,
    "z_dim": 128,
    "dropout": 0.1,
    "generated_replay_per_class": 600,
    "replay_temperature": 1.0,
    "lambda_cls": 10.0,
    "beta_kl": 1.0,
    "recon_weight": 1.0,
    "input_noise_std": 0.0,
    # Image-space DiVA: WideResNet-style encoder, image decoder, generated image replay.
    "wide_widen_factor": 4,
    # Paper uses CycleGAN-style domain translation for CIFAR; keep it enabled here.
    "domain_translation": True,
    "dt_epochs": 10,
    "dt_lr": 2e-4,
    "dt_channels": 32,
    "dt_cycle_weight": 10.0,
    "dt_identity_weight": 0.5,
    "grad_clip_norm": 5.0,
}

FAIR_MFGR_PAPER = {
    **_fair_launcher_common(),
    "memory_budget": 0,
    "generated_replay_budget": 600,
    "replay_percent": 0.05,
    "early_stop_val_scope": "current_task_only",
    "backbone": _FAIR_BACKBONE,
    "pretrained": True,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": DATA["num_workers"],
    "optimizer": "adam",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 80,
    "early_stop_patience": 15,
    "early_stop_min_delta": 1e-4,
    "hidden_dim": 512,
    "z_dim": 128,
    "dropout": 0.1,
    "replay_temperature": 1.0,
    "lambda_cls": 10.0,
    "beta_kl": 1.0,
    "recon_weight": 1.0,
    "input_noise_std": 0.0,
    "grad_clip_norm": 5.0,
    "mfgr_aligned": True,
    "mfgr_classifier_backbone": "resnet18",
    "mfgr_latent_dim": 1000,
    "mfgr_generator_base_channels": 32,
    "mfgr_generator_epochs": 80,
    "mfgr_generator_steps_per_epoch": 20,
    "mfgr_generator_batch_size": 64,
    "mfgr_generated_batch_size": 64,
    "mfgr_generator_lr": 0.01,
    "mfgr_temperature": 2.0,
    "mfgr_momentum": 0.9,
    "mfgr_goh_ratio": 1.0,
    "mfgr_gie_ratio": 5.0,
    "mfgr_ga_ratio": 0.1,
    "mfgr_gtv_ratio": 10.0,
    "mfgr_gbn_ratio": 20.0,
    "mfgr_gkl_ratio": 0.1,
    "mfgr_kl_img_sample_num": 200,
    "mfgr_o_ce": 1.0,
    "mfgr_n_ce": 1.0,
    "mfgr_o_kd": 1.0,
    "mfgr_n_kd": 1.0,
    "lr": 1e-3,
    "epochs": 80,
    "early_stop_patience": 15,
    "paper_source_dir": "MFGR_official_github",
}

FAIR_GFRIL_PAPER = {
    **_fair_launcher_common(),
    "memory_budget": 0,
    "generated_replay_budget": 600,
    "replay_percent": 0.05,
    "early_stop_val_scope": "current_task_only",
    "backbone": _FAIR_BACKBONE,
    "gfril_aligned": True,
    "gfril_classifier_backbone": "resnet18",
    "gfril_latent_dim": 200,
    "gfril_hidden_dim": 512,
    "gfril_generator_epochs": 501,
    "gfril_generator_steps_per_epoch": 0,
    "gfril_gan_lr": 1e-4,
    "gfril_lambda_gp": 10.0,
    "gfril_n_critic": 5,
    "gfril_feature_distill_weight": 1.0,
    "gfril_replay_cls_weight": 1.0,
    "gfril_alignment_weight": 1e-3,
    "gfril_replay_batch_size": 0,
    "pretrained": True,
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 32,
    "num_workers": DATA["num_workers"],
    "optimizer": "adam",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 201,
    "early_stop_patience": 15,
    "early_stop_min_delta": 1e-4,
    "hidden_dim": 512,
    "z_dim": 200,
    "dropout": 0.1,
    "replay_temperature": 1.0,
    "lambda_cls": 10.0,
    "beta_kl": 1.0,
    "recon_weight": 1.0,
    "input_noise_std": 0.0,
    "grad_clip_norm": 5.0,
    "paper_source_dir": "GFR-IL_official_github",
}

# GenClassifier — baselines/GenClassifier (CVPRW 2021; per-class VAE + importance-sampling Bayes)
FAIR_GENCLASSIFIER = {
    **_fair_launcher_common(),
    "memory_budget": 0,
    "backbone": "resnet18",
    # "feature": frozen CNN features -> FeatureVAE
    # "raw": direct image-space VAE (ablation)
    "vae_input_space": "feature",
    "image_size": DATA["image_size"],
    "flops_backward_factor": 3.0,
    "batch_size": 64,
    "num_workers": DATA["num_workers"],
    "vae_epochs": 100,
    "vae_lr": 1e-3,
    "z_dim": 128,
    "h_dim": 512,
    "eval_importance_samples": 600,
    "early_stop_patience": 10,
    "early_stop_min_delta": 1e-4,
}


def _external_method_profile(
    method_id: str,
    *,
    paper: str,
    year: int,
    category: str,
    repo_url: str,
    framework: str,
    method_key: str,
    exemplar_free: bool,
    generator_replay: bool,
    fair: dict,
    memory_budget: int,
):
    return {
        "name": method_id,
        "paper": paper,
        "year": year,
        "category": category,
        "repo_url": repo_url,
        "framework": framework,
        "method_key": method_key,
        "train": {
            "device": os.getenv("HIDMOA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
            "fair": deepcopy(fair),
            "exemplar_free": exemplar_free,
            "generator_replay": generator_replay,
            "memory_budget": memory_budget,
            "output_dir": f"runs/{method_id}",
            "external_launcher": {
                "enabled": True,
                # 建议把仓库放在 Incre_neu/baselines/<repo_name>
                "repo_dir": "",
                "workdir": "",
                "entrypoint": "",
                "args": [],
                "env": {},
                "notes": "请根据仓库 README 填写 repo_dir/entrypoint/args；main.py 会记录并可尝试执行。",
            },
        },
    }


# ============================================================
# 场景1：增量学习 — 冻结backbone逐任务MoE + VAE + Task-ID
# ============================================================
INCREMENTAL_1 = {
    "name": "incremental_moe",
    # 切换 backbone: "resnet18" | "resnet50" | "resnet101" | "deit_small_patch16_224_in661"（需与 COMMON_BACKBONE_CONFIGS 键名一致）
    "model": _common_backbone_model(
        "deit_small_patch16_224_in661",
        pretrained=True,
        experts_per_task=COMMON_BACKBONE_CONFIGS["deit_small_patch16_224_in661"]["experts_per_task"],
    ),
    "train": {
        "device": os.getenv("HIDMOA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs_per_task": 50,
        # 当 model.pretrained=False 时：session1 训练 backbone+新adapter；后续 session 冻结 backbone。
        "backbone_train_first_session_if_not_pretrained": True,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 1e-4,
        # learnable | post_train_imprint（含义见 COMMON_HEAD["prototype_mu_mode"] 注释）
        "prototype_mu_mode": COMMON_HEAD["prototype_mu_mode"],
        # Adapter/expert 训练阶段的 task 内部分类辅助损失。
        # lambda=0.0 时完全关闭，退化为原来的 CE 训练；建议首轮从 0.05 开始。
        "class_internal_loss": {
            "lambda": 0.0,
            "temperature": 0.2,
            "supcon_weight": 0.0,
            "focal_weight": 1.0,
            "focal_gamma": 2.0,
        },
        "oracle_taskid": False,
        "generator_aux_val_source": "dataset_val",  # "split_train" | "dataset_val"
        "generator_aux_val_ratio": 0.3,
        "generator_type": "vqvae",  # "cvae" | "vqvae"

        # Task-ID 分类器
        "taskid_aux_val_source": "dataset_val",  # "split_train" | "dataset_val"; 仅在 taskid 使用真实样本时生效
        "taskid_aux_val_ratio": 0.3,
        "taskid_replay_source": "generated",  # "real" | "generated"
        "taskid_replay_source_real_ratio": 0.05,
        "taskid_replay_source_real_use_current_task_ratio": True,  # replay_source=real时：False=当前task用全部真实样本；True=当前task也按replay_source_real_ratio保留
        "taskid_replay_source_generated_real_replay_ratio": 0.0,
        "taskid_use_generated_current_task": True,
        "taskid_continue_from_prev_session": False,  # False=每次重新随机初始化; True=若上一session已有task-id分类器则继续训练
        "taskid_epochs_initial": 15,
        "taskid_epochs_per_task_add": 0,
        "taskid_early_stopping_patience": None,  # None / "none" 关闭早停
        "taskid_early_stopping_min_delta": 1e-4,
        "taskid_lr": 1e-4,
        "taskid_weight_decay": 0.0,              # 设 0.0 关闭
        "taskid_use_cosine_scheduler": False,       # 设 False 关闭
        "taskid_backbone": "resnet18",
        "taskid_pretrained": True,
        "taskid_image_size": 64,
        "taskid_hidden_dim": 512,
        "taskid_hidden_layers": 2,
        "taskid_dropout": 0.1,
        "taskid_encoder_dim": 256,
        "taskid_batch_size": 64,
        "taskid_contrastive": {
            "enabled": False,
            "lambda": 1,
            "temperature": 0.5,
            "projection_dim": 128,
            "mode": "hierarchical",  # supervised / hierarchical / hierarchical_cosine
            "same_class_weight": 1.0,  # hierarchical模式下：同class正样本权重
            "same_task_diff_class_weight": 0.25,  # hierarchical模式下：同task不同class正样本权重
        },
        "taskid_margin_loss": {
            "enabled": False,
            "type": "arcface",  # arcface / cosface
            "scale": 30.0,
            "margin": 0.35,
        },
        # 推理阶段 task router（仅 evaluate_all，不影响训练）
        #   top1 — Task-ID argmax 硬路由（默认，与原先一致）
        #   top2 — margin p1-p2 < alpha/T 时对 top-2 task 各跑缺陷头，任务内 log-softmax 后联合 argmax
        "task_router_inference": "top1",
        "task_router_alpha": 0.4,
        # 条件图像 VAE
        "vae": {
            "image_size": 128,
            "latent_dim": 64,
            "base_channels": 32,
            "channel_multipliers": [1, 2, 4, 8],
            "epochs": 200,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "beta_kl": 0.01,
            "kl_warmup_epochs": 60,
            "recon_weight": 1.0,
            "l1_weight": 1.0,
            "perceptual_weight": 0.15,
            "perceptual_pretrained": True,
            "perceptual_layers": 3,
            "latent_pool_noise_std": 0.02,
            "generated_per_class": 300,
            "batch_size": 64,
        },

        # 条件图像 VQ-VAE（统一参数，原 vqvae_neu）
        "vqvae": {
            "image_size": 128,
            "embedding_dim": 64,
            "num_embeddings": 128,           # 256→128: 先用小 codebook 确保充分利用
            "base_channels": 32,
            "channel_multipliers": [1, 2, 4, 8],
            "commitment_cost": 1.0,          # 0.25→1.0: 增强 encoder→codebook 约束
            "codebook_weight": 0.25,         # 1.0→0.25: 防止 vq_loss 主导 early stopping
            "ema_decay": 0.99,               # 新增: EMA 更新 codebook，防止 collapse
            "epochs": 300,                   # 160→300: 修复后需更多 epoch 收敛
            "early_stopping_patience": 40,   # 20→30: 给 codebook 更多时间稳定
            "early_stopping_min_delta": 1e-4,
            "lr": 2e-4,                     # 5e-4→2e-4: 降低 lr 让 encoder 慢漂移
            "weight_decay": 0.0,
            "recon_weight": 1.0,
            "perceptual_weight": 0.3,
            "perceptual_pretrained": True,
            "perceptual_layers": 3,
            "generated_per_class": 600,
            "batch_size": 32,
            "use_pixelcnn_prior": False,
            "pixelcnn_hidden_channels": 128,
            "pixelcnn_num_layers": 6,
            "pixelcnn_kernel_size": 5,
            "pixelcnn_dropout": 0.15,        # 0.0→0.15: 防止 PixelCNN 过拟合
            "pixelcnn_epochs": 120,
            "pixelcnn_early_stopping_patience": 15,
            "pixelcnn_early_stopping_min_delta": 1e-4,
            "pixelcnn_lr": 3e-4,
            "pixelcnn_weight_decay": 1e-5,
            "pixelcnn_batch_size": 32,
            "pixelcnn_sampling_temperature": 1,   # 1.0→0.9: 轻微锐化采样
            "pixelcnn_sampling_top_k": 120,           # 0→80: 过滤低概率 code
        },

        "output_dir": "runs/incremental_1",
    },
}


# ============================================================
# 场景2：增量学习2（incre2）
# 与 incre1 同结构，但 task-id 由 class-level 生成式路由器替代。
# 这里全部显式展开，后续修改 incre2 不会影响 incre1。
# ============================================================
INCREMENTAL_2 = {
    "name": "HiDMoA",
    "model": _common_backbone_model(
        "resnet18",  # deit_small_patch16_224_in661 | resnet50 | resnet18
        pretrained=True,
        experts_per_task=COMMON_BACKBONE_CONFIGS["resnet18"]["experts_per_task"],
    ),
    "train": {
        "device": os.getenv("HIDMOA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size": int(os.getenv("HIDMOA_BATCH_SIZE", "32")),
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs_per_task": int(os.getenv("HIDMOA_EPOCHS_PER_TASK", "50")),
        "backbone_train_first_session_if_not_pretrained": True,
        "early_stopping_patience": int(os.getenv("HIDMOA_EARLY_STOP_PATIENCE", "5")),
        "early_stopping_min_delta": 1e-4,
        "head_type": "cosine_prototype", # COMMON_HEAD["head_type"], # "prototype" | "cosine_prototype" | "linear" | "cos"
        "prototype_mu_mode": COMMON_HEAD["prototype_mu_mode"],# "learnable" "post_train_imprint"
        "class_internal_loss": {
            "lambda": 0.0,
            "temperature": 0.2,
            "supcon_weight": 0.0,
            "focal_weight": 1.0,
            "focal_gamma": 2.0,
        },
        "oracle_taskid": False,
        "task_router_inference": "top1",
        "task_router_alpha": 0.4,
        "task_router_class_score_mode": "raw",
        "vae_router_type": "fvae",
        "vae_router_grouping": "task",
        "vae_router_score_mode": "recon",  # IS | recon | elbo_k
        "vae_router_backbone": "resnet18",
        "vae_router_backbone_pretrained": True,
        "vae_router_backbone_image_size": DATA.get("image_size", 224),
        "vae_router_use_feature_space": True,
        "fvae": {
            "h_dim": 512,
            "z_dim": 128,
            "input_dim": DATA["feat_dim"] if "feat_dim" in DATA else 512,
            "epochs": int(os.getenv("HIDMOA_FVAE_EPOCHS", "200")),
            "early_stopping_patience": int(os.getenv("HIDMOA_FVAE_PATIENCE", "20")),
            "early_stopping_min_delta": 1e-4,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "beta_kl": 0.01,
            "kl_warmup_epochs": 0,
            "recon_weight": 1.0,
            "batch_size": int(os.getenv("HIDMOA_FVAE_BATCH_SIZE", "64")),
            "generated_per_class": int(os.getenv("HIDMOA_FVAE_GENERATED_PER_CLASS", "600")),
            "fvae_feature_batch": int(os.getenv("HIDMOA_FVAE_FEATURE_BATCH", "64")),
        },
        "fcvae": {
            "h_dim": 512,
            "z_dim": 128,
            "epochs": int(os.getenv("HIDMOA_FVAE_EPOCHS", "200")),
            "early_stopping_patience": int(os.getenv("HIDMOA_FVAE_PATIENCE", "20")),
            "early_stopping_min_delta": 1e-4,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "beta_kl": 0.01,
            "kl_warmup_epochs": 0,
            "recon_weight": 1.0,
            "batch_size": int(os.getenv("HIDMOA_FVAE_BATCH_SIZE", "64")),
            "generated_per_class": int(os.getenv("HIDMOA_FVAE_GENERATED_PER_CLASS", "600")),
            "fvae_feature_batch": int(os.getenv("HIDMOA_FVAE_FEATURE_BATCH", "64")),
            "latent_pool_noise_std": 0.0,
        },
        "fvqvae": {
            "h_dim": 512,
            "embedding_dim": 128,
            "num_embeddings": 512,
            "commitment_cost": 0.25,
            "codebook_weight": 1.0,
            "ema_decay": 0.99,
            "epochs": int(os.getenv("HIDMOA_FVAE_EPOCHS", "200")),
            "early_stopping_patience": int(os.getenv("HIDMOA_FVAE_PATIENCE", "20")),
            "early_stopping_min_delta": 1e-4,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "recon_weight": 1.0,
            "batch_size": int(os.getenv("HIDMOA_FVAE_BATCH_SIZE", "64")),
            "fvae_feature_batch": int(os.getenv("HIDMOA_FVAE_FEATURE_BATCH", "64")),
            "generated_per_class": int(os.getenv("HIDMOA_FVAE_GENERATED_PER_CLASS", "600")),
        },
        "vae": {
            "image_size": 128,
            "latent_dim": 64,
            "base_channels": 32,
            "channel_multipliers": [1, 2, 4, 8],
            "epochs": 200,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "beta_kl": 0.01,
            "kl_warmup_epochs": 60,
            "recon_weight": 1.0,
            "l1_weight": 1.0,
            "perceptual_weight": 0.15,
            "perceptual_pretrained": True,
            "perceptual_layers": 3,
            "latent_pool_noise_std": 0.02,
            "generated_per_class": 300,
            "batch_size": 64,
        },
        "vae_router_use_class_prior": False,
        "vae_router_eval_importance_samples": int(os.getenv("HIDMOA_ROUTER_IMPORTANCE_SAMPLES", "600")),
        "vae_router_aggregation": "logsumexp",
        "taskid_image_size": DATA.get("image_size", 224),
        "vae_router_aux_val_source": "dataset_val",
        "vae_router_aux_val_ratio": 0.3,
        "task_router_eval_grid_enabled": False,
        "task_router_eval_grid_inference": "top2",
        "task_router_eval_alpha_values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "task_router_eval_class_score_modes": ["raw", "cardinality", "zscore"],
    },
}


# ============================================================
# 场景3：增量学习3（incre3）
# 初始参数与 incre2 保持一致，但这里全部显式展开，
# 后续修改 incre3 不会影响 incre1 / incre2。
# ============================================================
INCREMENTAL_3 = {
    "name": "incremental_moe_class_vae_router_3",
    "model": _common_backbone_model(
        "resnet18",
        pretrained=True,
        experts_per_task=COMMON_BACKBONE_CONFIGS["resnet18"]["experts_per_task"],
        allow_old_expert_reuse=True,
        old_expert_top_k=1,
    ),
    "train": {
        "device": os.getenv("HIDMOA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs_per_task": 50,
        "backbone_train_first_session_if_not_pretrained": True,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 1e-4,
        "head_type": COMMON_HEAD["head_type"],  # "prototype" | "cosine_prototype" | "linear" | "cos"
        "prototype_mu_mode": COMMON_HEAD["prototype_mu_mode"],
        "class_internal_loss": {
            "lambda": 0.0,
            "temperature": 0.2,
            "supcon_weight": 0.0,
            "focal_weight": 1.0,
            "focal_gamma": 2.0,
        },
        "oracle_taskid": False,
        "task_router_inference": "top1",
        "task_router_alpha": 0.4,
        "task_router_class_score_mode": "raw",
        "vae_router_type": "fvae",
        "vae_router_grouping": "task",
        "vae_router_score_mode": "elbo_k",  # IS | recon | elbo_k
        "vae_router_backbone": "resnet18",
        "vae_router_backbone_pretrained": True,
        "vae_router_backbone_image_size": DATA.get("image_size", 224),
        "vae_router_use_feature_space": True,
        "fvae": {
            "h_dim": 512,
            "z_dim": 128,
            "input_dim": DATA["feat_dim"] if "feat_dim" in DATA else 512,
            "epochs": 200,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "beta_kl": 0.01,
            "kl_warmup_epochs": 0,
            "recon_weight": 1.0,
            "batch_size": 64,
            "generated_per_class": 600,
            "fvae_feature_batch": 64,
        },
        "fcvae": {
            "h_dim": 512,
            "z_dim": 128,
            "epochs": 200,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "beta_kl": 0.01,
            "kl_warmup_epochs": 0,
            "recon_weight": 1.0,
            "batch_size": 64,
            "generated_per_class": 600,
            "fvae_feature_batch": 64,
            "latent_pool_noise_std": 0.0,
        },
        "fvqvae": {
            "h_dim": 512,
            "embedding_dim": 128,
            "num_embeddings": 512,
            "commitment_cost": 0.25,
            "codebook_weight": 1.0,
            "ema_decay": 0.99,
            "epochs": 200,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "recon_weight": 1.0,
            "batch_size": 64,
            "fvae_feature_batch": 64,
            "generated_per_class": 600,
        },
        "vae_router_use_class_prior": False,
        "vae_router_eval_importance_samples": 600,  # shared by IS / recon / elbo_k
        "vae_router_aggregation": "logsumexp",
        "taskid_image_size": DATA.get("image_size", 224),
        "vae_router_aux_val_source": "dataset_val",
        "vae_router_aux_val_ratio": 0.3,
        "task_router_eval_grid_enabled": False,
        "task_router_eval_grid_inference": "top2",
        "task_router_eval_alpha_values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "task_router_eval_class_score_modes": ["raw", "cardinality", "zscore"],
    },
}


# ============================================================
# 场景4：全量基线 — 冻结backbone + 6个专家一次性训练全部6类
#   专家数 = per-layer experts_per_task(2) × num_tasks(3) = 6，与增量最终一致
#   线性 gate + top-2 稀疏路由
# ============================================================
FULL_1 = {
    "name": "full_moe_baseline",
    "model": _common_backbone_model(
        "resnet18",
        num_experts=_shared_experts_per_layer("resnet18") * len(DATA["task_splits"]),
        top_k=2,
    ),
    "train": {
        "device": os.getenv("HIDMOA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 30,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 1e-4,
        "imprint_init": COMMON_HEAD["imprint_init"],

        "output_dir": "runs/full_1",
    },
}


# ============================================================
# 场景5：全量基线2 — 冻结backbone + 6个专家按任务固定分配
#   复用 IncrementalMoEResNet 结构，但所有专家同时训练、不冻结
#   专家 0-1 → task 0 类 (Crazing, Pitted_Surface)
#   专家 2-3 → task 1 类 (Inclusion, Patches)
#   专家 4-5 → task 2 类 (Rolled-in_Scale, Scratches)
#   推理时: 逐任务前向 → 取最高置信度的预测
# ============================================================
FULL_2 = {
    "name": "full_moe_fixed_route",
    "model": _common_backbone_model("resnet18", experts_per_task=COMMON_BACKBONE_CONFIGS["resnet18"]["experts_per_task"]),
    "train": {
        "device": os.getenv("HIDMOA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 30,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 1e-4,
        "imprint_init": COMMON_HEAD["imprint_init"],

        # 推理方式: "max_confidence" | "oracle_task"
        #   max_confidence — 逐任务前向，取最高 logit (默认，现实可用)
        #   oracle_task    — 用真实标签路由 (性能上界，仅做参考)
        "eval_mode": "max_confidence",

        "output_dir": "runs/full_2",
    },
}


# ============================================================
# 外部方法配置
# 说明:
# 1) 外部方法默认通过外部仓库运行，当前工程只负责统一配置与调度。
# 2) 每个方法使用各自的 FAIR_* 字典（train["fair"]），便于单独调参。
# ============================================================
INCRE_MRFA = _external_method_profile(
    "incre_mrfa",
    paper="Multi-layer Rehearsal Feature Augmentation for Class-Incremental Learning",
    year=2024,
    category="real_sample_replay",
    repo_url="https://github.com/bwnzheng/MRFA_ICML2024",
    framework="pycil_based",
    method_key="mrfa",
    exemplar_free=False,
    generator_replay=False,
    fair=FAIR_MRFA,
    memory_budget=FAIR_MRFA["memory_budget"],
)

INCRE_TAGFEX = _external_method_profile(
    "incre_tagfex",
    paper="Task-Agnostic Guided Feature Expansion for Class-Incremental Learning",
    year=2025,
    category="real_sample_replay",
    repo_url="https://github.com/bwnzheng/TagFex_CVPR2025",
    framework="pycil_based",
    method_key="tagfex",
    exemplar_free=False,
    generator_replay=False,
    fair=FAIR_TAGFEX,
    memory_budget=FAIR_TAGFEX["memory_budget"],
)

INCRE_DER_PAPER = _external_method_profile(
    "der_paper",
    paper="DER: Dynamically Expandable Representation for Class Incremental Learning",
    year=2021,
    category="dynamic_structure",
    repo_url="https://github.com/Rhyssiyan/DER-ClassIL.pytorch",
    framework="official_github_adapted",
    method_key="der_paper",
    exemplar_free=False,
    generator_replay=False,
    fair=FAIR_DER_PAPER,
    memory_budget=FAIR_DER_PAPER["memory_budget"],
)

INCRE_BEEF = _external_method_profile(
    "incre_beef",
    paper="BEEF: Bi-Compatible Class-Incremental Learning via Energy-Based Expansion and Fusion",
    year=2023,
    category="dynamic_structure",
    repo_url="https://github.com/G-U-N/ICLR23-BEEF",
    framework="official_or_pycil",
    method_key="beef",
    exemplar_free=False,
    generator_replay=False,
    fair=FAIR_BEEF,
    memory_budget=FAIR_BEEF["memory_budget"],
)

INCRE_EWC = _external_method_profile(
    "incre_ewc",
    paper="Overcoming Catastrophic Forgetting in Neural Networks",
    year=2017,
    category="parameter_regularization",
    repo_url="https://github.com/LAMDA-CL/PyCIL",
    framework="pycil",
    method_key="ewc",
    exemplar_free=True,
    generator_replay=False,
    fair=FAIR_EWC,
    memory_budget=FAIR_EWC["memory_budget"],
)

INCRE_ICARL = _external_method_profile(
    "incre_icarl",
    paper="iCaRL: Incremental Classifier and Representation Learning",
    year=2017,
    category="knowledge_distillation_with_exemplars",
    repo_url="https://github.com/LAMDA-CL/PyCIL",
    framework="pycil",
    method_key="icarl",
    exemplar_free=False,
    generator_replay=False,
    fair=FAIR_ICARL,
    memory_budget=FAIR_ICARL["memory_budget"],
)

INCRE_SEED_PAPER = _external_method_profile(
    "seed_paper",
    paper="Divide and not forget: Ensemble of selectively trained experts in Continual Learning (SEED)",
    year=2024,
    category="paper_source_exemplar_free",
    repo_url="https://github.com/grypesc/SEED",
    framework="downloaded_official_source_with_neu_adapter",
    method_key="seed",
    exemplar_free=True,
    generator_replay=False,
    fair=FAIR_SEED_PAPER,
    memory_budget=FAIR_SEED_PAPER["memory_budget"],
)

INCRE_TPL = _external_method_profile(
    "incre_tpl",
    paper="Class Incremental Learning via Likelihood Ratio Based Task Prediction",
    year=2024,
    category="task_id_prediction_with_replay",
    repo_url="https://github.com/linhaowei1/TPL",
    framework="official",
    method_key="tpl",
    exemplar_free=False,
    generator_replay=False,
    fair=FAIR_TPL,
    memory_budget=FAIR_TPL["memory_budget"],
)

INCRE_PEC = _external_method_profile(
    "incre_pec",
    paper="Prediction Error-based Classification for Class-Incremental Learning",
    year=2024,
    category="exemplar_free_prediction_error_classifier",
    repo_url="https://github.com/michalzajac-ml/pec",
    framework="official",
    method_key="pec",
    exemplar_free=True,
    generator_replay=False,
    fair=FAIR_PEC,
    memory_budget=FAIR_PEC["memory_budget"],
)

INCRE_MOVE_PAPER = _external_method_profile(
    "move_paper",
    paper="Hierarchically Structured Task-Agnostic Continual Learning",
    year=2022,
    category="task_agnostic_hvcl_sparse_moe_paper_aligned",
    repo_url="https://github.com/hhihn/HVCL/",
    framework="local_paper_aligned",
    method_key="move",
    exemplar_free=True,
    generator_replay=True,
    fair=FAIR_MOVE_PAPER,
    memory_budget=FAIR_MOVE_PAPER["memory_budget"],
)

INCRE_SEMA = _external_method_profile(
    "incre_sema",
    paper="Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning",
    year=2025,
    category="pretrained_vit_self_expansion_modular_adapters",
    repo_url="https://github.com/huiyiwang01/SEMA-CL",
    framework="local_adapter_from_paper",
    method_key="sema",
    exemplar_free=True,
    generator_replay=False,
    fair=FAIR_SEMA,
    memory_budget=FAIR_SEMA["memory_budget"],
)

INCRE_MOEADAPTERSPP_PAPER = _external_method_profile(
    "moeadapterspp_paper",
    paper="MoE-Adapters++: Fast and Effective Continual Learning for Vision-Language Models",
    year=2025,
    category="dynamic_moe_adapters_leas_deec_exemplar_free",
    repo_url="https://github.com/JiazuoYu/MoE-Adapters",
    framework="local_paper_aligned_adapter",
    method_key="moeadapterspp",
    exemplar_free=True,
    generator_replay=False,
    fair=FAIR_MOEADAPTERSPP,
    memory_budget=FAIR_MOEADAPTERSPP["memory_budget"],
)
INCRE_MORE_PAPER = _external_method_profile(
    "more_paper",
    paper="MORE: Class Incremental Learning with Task Specific OOD Detection",
    year=2022,
    category="paper_source_multi_head_ood_task_id_inference",
    repo_url="https://github.com/k-gyuhak/MORE",
    framework="local_adapter_for_paper_method",
    method_key="more",
    exemplar_free=False,
    generator_replay=False,
    fair=FAIR_MORE_PAPER,
    memory_budget=FAIR_MORE_PAPER["memory_budget"],
)
INCRE_MORE_PAPER_RESNET18 = _external_method_profile(
    "more_paper_resnet18",
    paper="MORE: Class Incremental Learning with Task Specific OOD Detection",
    year=2022,
    category="paper_source_multi_head_ood_task_id_inference",
    repo_url="https://github.com/k-gyuhak/MORE",
    framework="local_adapter_for_paper_method_resnet18",
    method_key="more",
    exemplar_free=False,
    generator_replay=False,
    fair=FAIR_MORE_PAPER_RESNET18,
    memory_budget=FAIR_MORE_PAPER_RESNET18["memory_budget"],
)
INCRE_BUILD_PAPER = _external_method_profile(
    "build_paper",
    paper="Buffer-free Class-Incremental Learning with Out-of-Distribution Detection",
    year=2025,
    category="buffer_free_multi_head_ood_task_id_prediction",
    repo_url="no official BUILD GitHub link in local PDF",
    framework="local_paper_aligned",
    method_key="build",
    exemplar_free=True,
    generator_replay=False,
    fair=FAIR_BUILD_PAPER,
    memory_budget=FAIR_BUILD_PAPER["memory_budget"],
)

INCRE_ITAML_PAPER = _external_method_profile(
    "itaml_paper",
    paper="iTAML: An Incremental Task-Agnostic Meta-learning Approach",
    year=2020,
    category="paper_source_task_agnostic_meta_learning_with_exemplars",
    repo_url="https://github.com/brjathu/iTAML",
    framework="downloaded_official_source_with_neu_adapter",
    method_key="itaml",
    exemplar_free=False,
    generator_replay=False,
    fair=FAIR_ITAML_PAPER,
    memory_budget=FAIR_ITAML_PAPER["memory_budget"],
)

INCRE_DIVA = _external_method_profile(
    "incre_diva",
    paper="DiVA: Discriminative Variational Autoencoder for Continual Learning with Generative Replay",
    year=2020,
    category="generative_replay_discriminative_vae",
    repo_url="",
    framework="local_adapter_from_paper",
    method_key="diva",
    exemplar_free=True,
    generator_replay=True,
    fair=FAIR_DIVA,
    memory_budget=FAIR_DIVA["memory_budget"],
)

INCRE_MFGR_PAPER = _external_method_profile(
    "mfgr_paper",
    paper="Memory-Free Generative Replay For Class-Incremental Learning",
    year=2021,
    category="downloaded_paper_memory_free_generative_replay",
    repo_url="https://github.com/xmengxin/MFGR",
    framework="downloaded_official_source_with_neu_adapter",
    method_key="mfgr_paper",
    exemplar_free=True,
    generator_replay=True,
    fair=FAIR_MFGR_PAPER,
    memory_budget=FAIR_MFGR_PAPER["memory_budget"],
)

INCRE_GFRIL_PAPER = _external_method_profile(
    "gfril_paper",
    paper="Generative Feature Replay For Class-Incremental Learning",
    year=2020,
    category="downloaded_paper_generative_feature_replay",
    repo_url="https://github.com/xialeiliu/GFR-IL",
    framework="downloaded_official_source_with_neu_adapter",
    method_key="gfril_paper",
    exemplar_free=True,
    generator_replay=True,
    fair=FAIR_GFRIL_PAPER,
    memory_budget=FAIR_GFRIL_PAPER["memory_budget"],
)

INCRE_GENCLASSIFIER = _external_method_profile(
    "incre_genclassifier",
    paper="Class-Incremental Learning With Generative Classifiers",
    year=2021,
    category="generative_classifier",
    repo_url="https://github.com/GMvandeVen/class-incremental-learning",
    framework="official",
    method_key="genclassifier",
    exemplar_free=True,
    generator_replay=False,
    fair=FAIR_GENCLASSIFIER,
    memory_budget=FAIR_GENCLASSIFIER["memory_budget"],
)

# ============================================================
EXTERNAL_INCREMENTAL_METHODS = {
    "incre_mrfa": INCRE_MRFA,
    "incre_tagfex": INCRE_TAGFEX,
    "der_paper": INCRE_DER_PAPER,
    "incre_beef": INCRE_BEEF,
    "incre_ewc": INCRE_EWC,
    "incre_icarl": INCRE_ICARL,
    "seed_paper": INCRE_SEED_PAPER,
    "incre_tpl": INCRE_TPL,
    "incre_pec": INCRE_PEC,
    "move_paper": INCRE_MOVE_PAPER,
    "incre_sema": INCRE_SEMA,
    "moeadapterspp_paper": INCRE_MOEADAPTERSPP_PAPER,
    "more_paper": INCRE_MORE_PAPER,
    "more_paper_resnet18": INCRE_MORE_PAPER_RESNET18,
    "build_paper": INCRE_BUILD_PAPER,
    "itaml_paper": INCRE_ITAML_PAPER,
    "incre_diva": INCRE_DIVA,
    "mfgr_paper": INCRE_MFGR_PAPER,
    "gfril_paper": INCRE_GFRIL_PAPER,
    "incre_genclassifier": INCRE_GENCLASSIFIER,
}


def _set_launcher_candidates(profile: dict, launchers: list):
    profile["train"]["launcher_candidates"] = deepcopy(launchers)
    if launchers:
        profile["train"]["external_launcher"] = deepcopy(launchers[0])


# PyCIL 优先；若该方法不在 PyCIL 或命令失败，可切官方仓库
_set_launcher_candidates(INCRE_EWC, [
    {
        "enabled": True,
        "repo_dir": PYCIL_REPO_DIR,
        "workdir": PYCIL_REPO_DIR,
        "entrypoint": "python main.py",
        "args": ["--config", "./exps/ewc.json"],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    },
])
_set_launcher_candidates(INCRE_ICARL, [
    {
        "enabled": True,
        "repo_dir": PYCIL_REPO_DIR,
        "workdir": PYCIL_REPO_DIR,
        "entrypoint": "python main.py",
        "args": ["--config", "./exps/icarl.json"],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    },
])
_set_launcher_candidates(INCRE_DER_PAPER, [
    {
        "enabled": True,
        "repo_dir": DER_REPO_DIR,
        "workdir": DER_REPO_DIR,
        "entrypoint": "python run_cil_paper.py",
        "args": [
            "--seed", "{seed}",
            "--batch-size", "{batch_size}",
            "--num-workers", "{der_paper_num_workers}",
            "--image-size", "{der_paper_image_size}",
            "--backbone", "{der_paper_backbone}",
            "--pretrained", "{der_paper_pretrained}",
            "--epochs", "{der_paper_epochs}",
            "--learning-rate", "{der_paper_learning_rate}",
            "--momentum", "{der_paper_momentum}",
            "--weight-decay", "{der_paper_weight_decay}",
            "--early-stop-patience", "{der_paper_early_stop_patience}",
            "--early-stop-min-delta", "{der_paper_early_stop_min_delta}",
            "--memory-budget", "{der_paper_memory_budget}",
            "--replay-percent", "{der_paper_replay_percent}",
            "--aux-loss-weight", "{der_paper_aux_loss_weight}",
            "--mask-reg-weight", "{der_paper_mask_reg_weight}",
            "--mask-threshold", "{der_paper_mask_threshold}",
            "--mask-scale", "{der_paper_mask_scale}",
            "--finetune-epochs", "{der_paper_finetune_epochs}",
            "--finetune-lr", "{der_paper_finetune_lr}",
            "--finetune-weight-decay", "{der_paper_finetune_weight_decay}",
            "--finetune-patience", "{der_paper_finetune_patience}",
            "--grad-clip-norm", "{der_paper_grad_clip_norm}",
            "--nf", "{der_paper_nf}",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "CIL_TASK_SPLITS": "{task_splits_json}",
            "CIL_CLASS_NAMES_JSON": "{class_names_json}",
            "PYTHONUNBUFFERED": "1",
        },
    },
])
_set_launcher_candidates(INCRE_BEEF, [
    {
        "enabled": True,
        "repo_dir": PYCIL_REPO_DIR,
        "workdir": PYCIL_REPO_DIR,
        "entrypoint": "python main.py",
        "args": ["--config", "./exps/beef.json"],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    },
    {"enabled": True, "repo_dir": BEEF_REPO_DIR, "workdir": BEEF_REPO_DIR, "entrypoint": "python main.py", "args": [], "env": {}},
])
_set_launcher_candidates(INCRE_TAGFEX, [
    {
        "enabled": True,
        "repo_dir": TAGFEX_REPO_DIR,
        "workdir": TAGFEX_REPO_DIR,
        "entrypoint": "python main.py",
        "args": [
            "train",
            "--exp-configs",
            "./configs/all_in_one/neu_xsdd_tagfex_resnet18.yaml",
            "--dataset-root",
            "{data_root}",
            "--seed",
            "{seed}",
            "--device",
            "cuda",
            "--disable-save-ckpt",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    },
])

# 以下方法通常不在官方 PyCIL 主干中，直接走各自仓库
_set_launcher_candidates(INCRE_MRFA, [
    {
        "enabled": True,
        "repo_dir": MRFA_REPO_DIR,
        "workdir": MRFA_REPO_DIR,
        "entrypoint": "python main.py",
        "args": ["--config", "./exps/icarl/mrfa/neu_xsdd.json"],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    },
])
_set_launcher_candidates(INCRE_TPL, [
    {
        "enabled": True,
        "repo_dir": TPL_REPO_DIR,
        "workdir": TPL_REPO_DIR,
        "entrypoint": "python run_cil_neu_xddd.py",
        "args": [
            "--seed", "{seed}",
            "--batch-size", "{batch_size}",
            "--replay-batch-size", "{tpl_replay_batch_size}",
            "--replay-mode", "{tpl_replay_mode}",
            "--replay-percent", "{tpl_replay_percent}",
            "--replay-buffer-size", "{tpl_replay_buffer_size}",
            "--learning-rate", "{tpl_learning_rate}",
            "--num-train-epochs", "{tpl_num_train_epochs}",
            "--early-stop-patience", "{tpl_early_stop_patience}",
            "--early-stop-min-delta", "{tpl_early_stop_min_delta}",
            "--visual-encoder", "{tpl_visual_encoder}",
            "--latent", "{tpl_latent}",
            "--K", "{tpl_k}",
            "--alpha", "{tpl_alpha}",
            "--smax", "{tpl_smax}",
            "--clipgrad", "{tpl_clipgrad}",
            "--thres-cosh", "{tpl_thres_cosh}",
            "--weight-decay", "{tpl_weight_decay}",
            "--warmup-ratio", "{tpl_warmup_ratio}",
            "--lr-scheduler-type", "{tpl_lr_scheduler_type}",
            "--num-warmup-steps", "{tpl_num_warmup_steps}",
            "--gradient-accumulation-steps", "{tpl_gradient_accumulation_steps}",
            "--baseline-name", "{tpl_baseline_name}",
            "--base-dir", "{tpl_base_dir}",
            "--sequence-file", "{tpl_sequence_file}",
            "--pretrained-dir", "{tpl_pretrained_dir}",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "CIL_TASK_SPLITS": "{task_splits_json}",
            "CIL_CLASS_NAMES_JSON": "{class_names_json}",
            "CIL_PROJECT_DIR": THIS_DIR,
            "PYTHONUNBUFFERED": "1",
        },
    },
])
_set_launcher_candidates(INCRE_PEC, [
    {
        "enabled": True,
        "repo_dir": PEC_REPO_DIR,
        "workdir": PEC_REPO_DIR,
        "entrypoint": "python run_cil_neu_xddd.py",
        "args": [
            "--seed", "{seed}",
            "--batch-size", "{batch_size}",
            "--num-workers", "{pec_num_workers}",
            "--image-size", "{pec_image_size}",
            "--n-epochs", "{pec_num_train_epochs}",
            "--learning-rate", "{pec_learning_rate}",
            "--optim-kind", "{pec_optim_kind}",
            "--optim-scheduler", "{pec_lr_scheduler}",
            "--early-stop-patience", "{pec_early_stop_patience}",
            "--early-stop-min-delta", "{pec_early_stop_min_delta}",
            "--pec-architecture", "{pec_architecture}",
            "--pec-num-layers", "{pec_num_layers}",
            "--pec-width", "{pec_width}",
            "--pec-teacher-width-multiplier", "{pec_teacher_width_multiplier}",
            "--pec-output-dim", "{pec_output_dim}",
            "--pec-activation", "{pec_activation}",
            "--pec-normalize-layers", "{pec_normalize_layers}",
            "--pec-conv-layers", "{pec_conv_layers}",
            "--pec-conv-reduce-spatial-to", "{pec_conv_reduce_spatial_to}",
            "--pec-train-chunk-size", "{pec_train_chunk_size}",
            "--force-no-augmentations", "{pec_force_no_augmentations}",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "CIL_TASK_SPLITS": "{task_splits_json}",
            "CIL_CLASS_NAMES_JSON": "{class_names_json}",
            "PYTHONUNBUFFERED": "1",
        },
    },
])
_set_launcher_candidates(INCRE_MOVE_PAPER, [
    {
        "enabled": True,
        "repo_dir": MOVE_REPO_DIR,
        "workdir": MOVE_REPO_DIR,
        "entrypoint": "python run_cil_paper.py",
        "args": [
            "--seed", "{seed}",
            "--batch-size", "{batch_size}",
            "--num-workers", "{move_num_workers}",
            "--image-size", "{move_image_size}",
            "--epochs", "{move_epochs}",
            "--learning-rate", "{move_learning_rate}",
            "--weight-decay", "{move_weight_decay}",
            "--early-stop-patience", "{move_early_stop_patience}",
            "--early-stop-min-delta", "{move_early_stop_min_delta}",
            "--num-experts", "{move_num_experts}",
            "--top-k", "{move_top_k}",
            "--kernel-width", "{move_kernel_width}",
            "--hidden-dim", "{move_hidden_dim}",
            "--dropout", "{move_dropout}",
            "--prior-kl-weight", "{move_prior_kl_weight}",
            "--gate-prior-kl-weight", "{move_gate_prior_kl_weight}",
            "--gate-entropy-weight", "{move_gate_entropy_weight}",
            "--expert-diversity-weight", "{move_expert_diversity_weight}",
            "--generator-hidden-dim", "{move_generator_hidden_dim}",
            "--generator-z-dim", "{move_generator_z_dim}",
            "--generator-epochs", "{move_generator_epochs}",
            "--generator-early-stop-patience", "{move_generator_early_stop_patience}",
            "--generator-early-stop-min-delta", "{move_generator_early_stop_min_delta}",
            "--generator-beta-kl", "{move_generator_beta_kl}",
            "--generator-recon-weight", "{move_generator_recon_weight}",
            "--generator-replay-per-class", "{move_generator_replay_per_class}",
            "--generator-replay-temperature", "{move_generator_replay_temperature}",
            "--grad-clip-norm", "{move_grad_clip_norm}",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "CIL_TASK_SPLITS": "{task_splits_json}",
            "CIL_CLASS_NAMES_JSON": "{class_names_json}",
            "PYTHONUNBUFFERED": "1",
        },
    },
])
_set_launcher_candidates(INCRE_SEMA, [
    {
        "enabled": True,
        "repo_dir": SEMA_REPO_DIR,
        "workdir": SEMA_REPO_DIR,
        "entrypoint": "python run_cil_neu_xddd.py",
        "args": [
            "--seed", "{seed}",
            "--batch-size", "{batch_size}",
            "--num-workers", "{sema_num_workers}",
            "--image-size", "{sema_image_size}",
            "--backbone", "{sema_backbone}",
            "--pretrained", "{sema_pretrained}",
            "--epochs", "{sema_epochs}",
            "--adapter-lr", "{sema_adapter_lr}",
            "--rd-lr", "{sema_rd_lr}",
            "--weight-decay", "{sema_weight_decay}",
            "--early-stop-patience", "{sema_early_stop_patience}",
            "--early-stop-min-delta", "{sema_early_stop_min_delta}",
            "--adapter-hidden-dim", "{sema_adapter_hidden_dim}",
            "--rd-hidden-dim", "{sema_rd_hidden_dim}",
            "--rd-bottleneck-dim", "{sema_rd_bottleneck_dim}",
            "--expansion-layers", "{sema_expansion_layers}",
            "--expansion-threshold", "{sema_expansion_threshold}",
            "--expansion-min-fraction", "{sema_expansion_min_fraction}",
            "--rd-loss-weight", "{sema_rd_loss_weight}",
            "--classifier-scale", "{sema_classifier_scale}",
            "--grad-clip-norm", "{sema_grad_clip_norm}",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "CIL_TASK_SPLITS": "{task_splits_json}",
            "CIL_CLASS_NAMES_JSON": "{class_names_json}",
            "PYTHONUNBUFFERED": "1",
        },
    },
])
_set_launcher_candidates(INCRE_MOEADAPTERSPP_PAPER, [
    {
        "enabled": True,
        "repo_dir": MOEADAPTERSPP_REPO_DIR,
        "workdir": MOEADAPTERSPP_REPO_DIR,
        "entrypoint": "python run_cil_neu_xddd.py",
        "args": [
            "--seed", "{seed}",
            "--batch-size", "{batch_size}",
            "--num-workers", "{moeadapterspp_num_workers}",
            "--image-size", "{moeadapterspp_image_size}",
            "--backbone", "{moeadapterspp_backbone}",
            "--pretrained", "{moeadapterspp_pretrained}",
            "--epochs", "{moeadapterspp_epochs}",
            "--learning-rate", "{moeadapterspp_learning_rate}",
            "--weight-decay", "{moeadapterspp_weight_decay}",
            "--early-stop-patience", "{moeadapterspp_early_stop_patience}",
            "--early-stop-min-delta", "{moeadapterspp_early_stop_min_delta}",
            "--router-hidden-dim", "{moeadapterspp_router_hidden_dim}",
            "--expert-hidden-dim", "{moeadapterspp_expert_hidden_dim}",
            "--ae-hidden-dim", "{moeadapterspp_ae_hidden_dim}",
            "--ae-bottleneck-dim", "{moeadapterspp_ae_bottleneck_dim}",
            "--recognition-layer", "{moeadapterspp_recognition_layer}",
            "--subsequent-layers", "{moeadapterspp_subsequent_layers}",
            "--initial-experts", "{moeadapterspp_initial_experts}",
            "--top-k", "{moeadapterspp_top_k}",
            "--expansion-threshold", "{moeadapterspp_expansion_threshold}",
            "--expansion-min-fraction", "{moeadapterspp_expansion_min_fraction}",
            "--leas-loss-weight", "{moeadapterspp_leas_loss_weight}",
            "--deec-loss-weight", "{moeadapterspp_deec_loss_weight}",
            "--classifier-scale", "{moeadapterspp_classifier_scale}",
            "--grad-clip-norm", "{moeadapterspp_grad_clip_norm}",
            "--label-smoothing", "{moeadapterspp_label_smoothing}",
            "--paper-mode", "{moeadapterspp_paper_mode}",
            "--paper-warmup-epochs", "{moeadapterspp_paper_warmup_epochs}",
            "--paper-preference-window", "{moeadapterspp_paper_preference_window}",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "CIL_TASK_SPLITS": "{task_splits_json}",
            "CIL_CLASS_NAMES_JSON": "{class_names_json}",
            "CIL_METHOD_NAME": "moeadapterspp_paper",
            "CIL_PROJECT_DIR": THIS_DIR,
            "PYTHONUNBUFFERED": "1",
        },
    },
])
_set_launcher_candidates(INCRE_SEED_PAPER, [
    {
        "enabled": True,
        "repo_dir": SEED_PAPER_REPO_DIR,
        "workdir": SEED_PAPER_REPO_DIR,
        "entrypoint": "python run_cil_paper.py",
        "args": [
            "--gpu", "0",
            "--seed", "{seed}",
            "--batch-size", "{batch_size}",
            "--num-workers", "{seed_num_workers}",
            "--num-tasks", "{num_tasks}",
            "--nc-first-task", "{init_cls}",
            "--network", "{seed_network}",
            "--pretrained", "{seed_pretrained}",
            "--nepochs", "{seed_nepochs}",
            "--ftepochs", "{seed_ftepochs}",
            "--lr", "{seed_lr}",
            "--weight-decay", "{seed_weight_decay}",
            "--momentum", "{seed_momentum}",
            "--clipping", "{seed_clipping}",
            "--max-experts", "{seed_max_experts}",
            "--gmms", "{seed_gmms}",
            "--alpha", "{seed_alpha}",
            "--tau", "{seed_tau}",
            "--shared", "{seed_shared}",
            "--extra-aug", "{seed_extra_aug}",
            "--early-stop-patience", "{seed_early_stop_patience}",
            "--early-stop-min-delta", "{seed_early_stop_min_delta}",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "CIL_TASK_SPLITS": "{task_splits_json}",
            "CIL_CLASS_NAMES_JSON": "{class_names_json}",
            "CIL_METHOD_NAME": "seed_paper",
            "CIL_PROJECT_DIR": THIS_DIR,
            "PYTHONUNBUFFERED": "1",
        },
    },
])
_set_launcher_candidates(INCRE_MORE_PAPER, [
    {
        "enabled": True,
        "repo_dir": MORE_PAPER_REPO_DIR,
        "workdir": MORE_PAPER_REPO_DIR,
        "entrypoint": "python run_cil_paper.py",
        "args": [
            "--seed", "{seed}",
            "--batch-size", "{batch_size}",
            "--num-workers", "{more_num_workers}",
            "--image-size", "{more_image_size}",
            "--backbone", "{more_backbone}",
            "--pretrained", "{more_pretrained}",
            "--epochs", "{more_epochs}",
            "--learning-rate", "{more_learning_rate}",
            "--weight-decay", "{more_weight_decay}",
            "--momentum", "{more_momentum}",
            "--early-stop-patience", "{more_early_stop_patience}",
            "--early-stop-min-delta", "{more_early_stop_min_delta}",
            "--hidden-dim", "{more_hidden_dim}",
            "--dropout", "{more_dropout}",
            "--replay-percent", "{more_replay_percent}",
            "--memory-budget", "{memory_budget}",
            "--back-update", "{more_back_update}",
            "--back-update-epochs", "{more_back_update_epochs}",
            "--back-update-lr", "{more_back_update_lr}",
            "--back-update-batch-size", "{more_back_update_batch_size}",
            "--distance-scale", "{more_distance_scale}",
            "--use-distance-coeff", "{more_use_distance_coeff}",
            "--grad-clip-norm", "{more_grad_clip_norm}",
            "--adapter-latent", "{more_adapter_dim}",
            "--smax", "{more_smax}",
            "--lamb0", "{more_reg_lambda}",
            "--lamb1", "{more_reg_lambda}",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "CIL_TASK_SPLITS": "{task_splits_json}",
            "CIL_CLASS_NAMES_JSON": "{class_names_json}",
            "CIL_METHOD_NAME": "more_paper",
            "CIL_PROJECT_DIR": THIS_DIR,
            "PYTHONUNBUFFERED": "1",
        },
    },
])
_set_launcher_candidates(INCRE_MORE_PAPER_RESNET18, [
    {
        "enabled": True,
        "repo_dir": MORE_PAPER_REPO_DIR,
        "workdir": MORE_PAPER_REPO_DIR,
        "entrypoint": "python run_cil_paper.py",
        "args": [
            "--seed", "{seed}",
            "--batch-size", "{batch_size}",
            "--num-workers", "{more_num_workers}",
            "--image-size", "{more_image_size}",
            "--backbone", "{more_backbone}",
            "--pretrained", "{more_pretrained}",
            "--epochs", "{more_epochs}",
            "--learning-rate", "{more_learning_rate}",
            "--weight-decay", "{more_weight_decay}",
            "--momentum", "{more_momentum}",
            "--early-stop-patience", "{more_early_stop_patience}",
            "--early-stop-min-delta", "{more_early_stop_min_delta}",
            "--hidden-dim", "{more_hidden_dim}",
            "--dropout", "{more_dropout}",
            "--replay-percent", "{more_replay_percent}",
            "--memory-budget", "{memory_budget}",
            "--back-update", "{more_back_update}",
            "--back-update-epochs", "{more_back_update_epochs}",
            "--back-update-lr", "{more_back_update_lr}",
            "--back-update-batch-size", "{more_back_update_batch_size}",
            "--distance-scale", "{more_distance_scale}",
            "--use-distance-coeff", "{more_use_distance_coeff}",
            "--grad-clip-norm", "{more_grad_clip_norm}",
            "--adapter-latent", "{more_adapter_dim}",
            "--smax", "{more_smax}",
            "--lamb0", "{more_reg_lambda}",
            "--lamb1", "{more_reg_lambda}",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "CIL_TASK_SPLITS": "{task_splits_json}",
            "CIL_CLASS_NAMES_JSON": "{class_names_json}",
            "CIL_METHOD_NAME": "more_paper_resnet18",
            "CIL_PROJECT_DIR": THIS_DIR,
            "PYTHONUNBUFFERED": "1",
        },
    },
])
_set_launcher_candidates(INCRE_BUILD_PAPER, [
    {
        "enabled": True,
        "repo_dir": BUILD_REPO_DIR,
        "workdir": BUILD_REPO_DIR,
        "entrypoint": "python run_cil_paper.py",
        "args": [
            "--seed", "{seed}",
            "--batch-size", "{batch_size}",
            "--num-workers", "{build_num_workers}",
            "--image-size", "{build_image_size}",
            "--backbone", "{build_backbone}",
            "--pretrained", "{build_pretrained}",
            "--epochs", "{build_epochs}",
            "--learning-rate", "{build_learning_rate}",
            "--weight-decay", "{build_weight_decay}",
            "--momentum", "{build_momentum}",
            "--early-stop-patience", "{build_early_stop_patience}",
            "--early-stop-min-delta", "{build_early_stop_min_delta}",
            "--adapter-dim", "{build_adapter_dim}",
            "--dropout", "{build_dropout}",
            "--smax", "{build_smax}",
            "--mask-reg-weight", "{build_mask_reg_weight}",
            "--detector", "{build_detector}",
            "--scorer", "{build_scorer}",
            "--react-percentile", "{build_react_percentile}",
            "--dice-percentile", "{build_dice_percentile}",
            "--scale-percentile", "{build_scale_percentile}",
            "--md-scale", "{build_md_scale}",
            "--md-ridge", "{build_md_ridge}",
            "--grad-clip-norm", "{build_grad_clip_norm}",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "CIL_TASK_SPLITS": "{task_splits_json}",
            "CIL_CLASS_NAMES_JSON": "{class_names_json}",
            "CIL_METHOD_NAME": "build_paper",
            "CIL_PROJECT_DIR": THIS_DIR,
            "PYTHONUNBUFFERED": "1",
        },
    },
])
_set_launcher_candidates(INCRE_ITAML_PAPER, [
    {
        "enabled": True,
        "repo_dir": ITAML_PAPER_REPO_DIR,
        "workdir": ITAML_PAPER_REPO_DIR,
        "entrypoint": "python run_cil_paper.py",
        "args": [
            "--seed", "{seed}",
            "--batch-size", "{batch_size}",
            "--num-workers", "{itaml_num_workers}",
            "--image-size", "{itaml_image_size}",
            "--backbone", "{itaml_backbone}",
            "--pretrained", "{itaml_pretrained}",
            "--epochs", "{itaml_epochs}",
            "--learning-rate", "{itaml_learning_rate}",
            "--optimizer", "{itaml_optimizer}",
            "--weight-decay", "{itaml_weight_decay}",
            "--momentum", "{itaml_momentum}",
            "--early-stop-patience", "{itaml_early_stop_patience}",
            "--early-stop-min-delta", "{itaml_early_stop_min_delta}",
            "--hidden-dim", "{itaml_hidden_dim}",
            "--embed-dim", "{itaml_embed_dim}",
            "--dropout", "{itaml_dropout}",
            "--replay-percent", "{itaml_replay_percent}",
            "--memory-budget", "{memory_budget}",
            "--inner-steps", "{itaml_inner_steps}",
            "--beta", "{itaml_beta}",
            "--continuum-size", "{itaml_continuum_size}",
            "--grad-clip-norm", "{itaml_grad_clip_norm}",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "CIL_TASK_SPLITS": "{task_splits_json}",
            "CIL_CLASS_NAMES_JSON": "{class_names_json}",
            "CIL_METHOD_NAME": "itaml_paper",
            "CIL_PROJECT_DIR": THIS_DIR,
            "PYTHONUNBUFFERED": "1",
        },
    },
])
_set_launcher_candidates(INCRE_DIVA, [
    {
        "enabled": True,
        "repo_dir": DIVA_REPO_DIR,
        "workdir": DIVA_REPO_DIR,
        "entrypoint": "python run_cil_neu_xddd.py",
        "args": [
            "--seed", "{seed}",
            "--batch-size", "{batch_size}",
            "--num-workers", "{diva_num_workers}",
            "--image-size", "{diva_image_size}",
            "--backbone", "{diva_backbone}",
            "--pretrained", "{diva_pretrained}",
            "--epochs", "{diva_epochs}",
            "--learning-rate", "{diva_learning_rate}",
            "--weight-decay", "{diva_weight_decay}",
            "--early-stop-patience", "{diva_early_stop_patience}",
            "--early-stop-min-delta", "{diva_early_stop_min_delta}",
            "--hidden-dim", "{diva_hidden_dim}",
            "--z-dim", "{diva_z_dim}",
            "--dropout", "{diva_dropout}",
            "--generated-replay-per-class", "{diva_generated_replay_per_class}",
            "--replay-temperature", "{diva_replay_temperature}",
            "--lambda-cls", "{diva_lambda_cls}",
            "--beta-kl", "{diva_beta_kl}",
            "--recon-weight", "{diva_recon_weight}",
            "--input-noise-std", "{diva_input_noise_std}",
            "--wide-widen-factor", "{diva_wide_widen_factor}",
            "--domain-translation", "{diva_domain_translation}",
            "--dt-epochs", "{diva_dt_epochs}",
            "--dt-lr", "{diva_dt_lr}",
            "--dt-channels", "{diva_dt_channels}",
            "--dt-cycle-weight", "{diva_dt_cycle_weight}",
            "--dt-identity-weight", "{diva_dt_identity_weight}",
            "--grad-clip-norm", "{diva_grad_clip_norm}",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "CIL_TASK_SPLITS": "{task_splits_json}",
            "CIL_CLASS_NAMES_JSON": "{class_names_json}",
            "PYTHONUNBUFFERED": "1",
        },
    },
])
_PAPER_GENERATIVE_REPLAY_ARGS = [
    "--method-name", "{method_key}",
    "--paper-source", "{paper_source_dir}",
    "--seed", "{seed}",
    "--batch-size", "{batch_size}",
    "--num-workers", "{paper_num_workers}",
    "--image-size", "{paper_image_size}",
    "--backbone", "{paper_backbone}",
    "--pretrained", "{paper_pretrained}",
    "--epochs", "{paper_epochs}",
    "--learning-rate", "{paper_learning_rate}",
    "--weight-decay", "{paper_weight_decay}",
    "--early-stop-patience", "{paper_early_stop_patience}",
    "--early-stop-min-delta", "{paper_early_stop_min_delta}",
    "--hidden-dim", "{paper_hidden_dim}",
    "--z-dim", "{paper_z_dim}",
    "--dropout", "{paper_dropout}",
    "--generated-replay-budget", "{paper_generated_replay_budget}",
    "--replay-percent", "{paper_replay_percent}",
    "--replay-temperature", "{paper_replay_temperature}",
    "--lambda-cls", "{paper_lambda_cls}",
    "--beta-kl", "{paper_beta_kl}",
    "--recon-weight", "{paper_recon_weight}",
    "--mfgr-aligned", "{paper_mfgr_aligned}",
    "--mfgr-classifier-backbone", "{paper_mfgr_classifier_backbone}",
    "--mfgr-latent-dim", "{paper_mfgr_latent_dim}",
    "--mfgr-generator-base-channels", "{paper_mfgr_generator_base_channels}",
    "--mfgr-generator-epochs", "{paper_mfgr_generator_epochs}",
    "--mfgr-generator-steps-per-epoch", "{paper_mfgr_generator_steps_per_epoch}",
    "--mfgr-generator-batch-size", "{paper_mfgr_generator_batch_size}",
    "--mfgr-generated-batch-size", "{paper_mfgr_generated_batch_size}",
    "--mfgr-generator-lr", "{paper_mfgr_generator_lr}",
    "--mfgr-temperature", "{paper_mfgr_temperature}",
    "--mfgr-momentum", "{paper_mfgr_momentum}",
    "--mfgr-goh-ratio", "{paper_mfgr_goh_ratio}",
    "--mfgr-gie-ratio", "{paper_mfgr_gie_ratio}",
    "--mfgr-ga-ratio", "{paper_mfgr_ga_ratio}",
    "--mfgr-gtv-ratio", "{paper_mfgr_gtv_ratio}",
    "--mfgr-gbn-ratio", "{paper_mfgr_gbn_ratio}",
    "--mfgr-gkl-ratio", "{paper_mfgr_gkl_ratio}",
    "--mfgr-kl-img-sample-num", "{paper_mfgr_kl_img_sample_num}",
    "--mfgr-o-ce", "{paper_mfgr_o_ce}",
    "--mfgr-n-ce", "{paper_mfgr_n_ce}",
    "--mfgr-o-kd", "{paper_mfgr_o_kd}",
    "--mfgr-n-kd", "{paper_mfgr_n_kd}",
    "--gfril-aligned", "{paper_gfril_aligned}",
    "--gfril-classifier-backbone", "{paper_gfril_classifier_backbone}",
    "--gfril-latent-dim", "{paper_gfril_latent_dim}",
    "--gfril-hidden-dim", "{paper_gfril_hidden_dim}",
    "--gfril-generator-epochs", "{paper_gfril_generator_epochs}",
    "--gfril-generator-steps-per-epoch", "{paper_gfril_generator_steps_per_epoch}",
    "--gfril-gan-lr", "{paper_gfril_gan_lr}",
    "--gfril-lambda-gp", "{paper_gfril_lambda_gp}",
    "--gfril-n-critic", "{paper_gfril_n_critic}",
    "--gfril-feature-distill-weight", "{paper_gfril_feature_distill_weight}",
    "--gfril-replay-cls-weight", "{paper_gfril_replay_cls_weight}",
    "--gfril-alignment-weight", "{paper_gfril_alignment_weight}",
    "--gfril-replay-batch-size", "{paper_gfril_replay_batch_size}",
    "--input-noise-std", "{paper_input_noise_std}",
    "--grad-clip-norm", "{paper_grad_clip_norm}",
]
_PAPER_GENERATIVE_REPLAY_ENV = {
    "CIL_DATA_ROOTS": "{data_root}",
    "CIL_TASK_SPLITS": "{task_splits_json}",
    "CIL_CLASS_NAMES_JSON": "{class_names_json}",
    "CIL_METHOD_NAME": "{method_key}",
    "CIL_PROJECT_DIR": THIS_DIR,
    "PYTHONUNBUFFERED": "1",
}
_set_launcher_candidates(INCRE_MFGR_PAPER, [
    {
        "enabled": True,
        "repo_dir": MFGR_PAPER_REPO_DIR,
        "workdir": MFGR_PAPER_REPO_DIR,
        "entrypoint": "python run_cil_paper.py",
        "args": deepcopy(_PAPER_GENERATIVE_REPLAY_ARGS),
        "env": deepcopy(_PAPER_GENERATIVE_REPLAY_ENV),
    },
])
_set_launcher_candidates(INCRE_GFRIL_PAPER, [
    {
        "enabled": True,
        "repo_dir": GFRIL_PAPER_REPO_DIR,
        "workdir": GFRIL_PAPER_REPO_DIR,
        "entrypoint": "python run_cil_paper.py",
        "args": deepcopy(_PAPER_GENERATIVE_REPLAY_ARGS),
        "env": deepcopy(_PAPER_GENERATIVE_REPLAY_ENV),
    },
])
_set_launcher_candidates(INCRE_GENCLASSIFIER, [
    {
        "enabled": True,
        "repo_dir": GENCLASSIFIER_REPO_DIR,
        "workdir": GENCLASSIFIER_REPO_DIR,
        "entrypoint": "python {gc_entrypoint}",
        "args": [
            "--gpu", "0",
            "--seed", "{seed}",
            "--batch-size", "{batch_size}",
            "--num-workers", "{gc_num_workers}",
            "--image-size", "{gc_image_size}",
            "--vae-epochs", "{gc_vae_epochs}",
            "--vae-lr", "{gc_vae_lr}",
            "--z-dim", "{gc_z_dim}",
            "--h-dim", "{gc_h_dim}",
            "--eval-s", "{gc_eval_s}",
            "--backbone", "{gc_backbone}",
            "--early-stop-patience", "{gc_early_stop_patience}",
            "--early-stop-min-delta", "{gc_early_stop_min_delta}",
        ],
        "env": {
            "CIL_DATA_ROOTS": "{data_root}",
            "CIL_TASK_SPLITS": "{task_splits_json}",
            "CIL_CLASS_NAMES_JSON": "{class_names_json}",
            "PYTHONUNBUFFERED": "1",
        },
    },
])


# ============================================================
# 汇总
# ============================================================
PROFILES = [
    INCREMENTAL_1,
    INCREMENTAL_2,
    INCREMENTAL_3,
    FULL_1,
    FULL_2,
    INCRE_MRFA,
    INCRE_TAGFEX,
    INCRE_DER_PAPER,
    INCRE_BEEF,
    INCRE_EWC,
    INCRE_ICARL,
    INCRE_SEED_PAPER,
    INCRE_TPL,
    INCRE_PEC,
    INCRE_MOVE_PAPER,
    INCRE_SEMA,
    INCRE_MOEADAPTERSPP_PAPER,
    INCRE_MORE_PAPER,
    INCRE_MORE_PAPER_RESNET18,
    INCRE_BUILD_PAPER,
    INCRE_ITAML_PAPER,
    INCRE_DIVA,
    INCRE_MFGR_PAPER,
    INCRE_GFRIL_PAPER,
    INCRE_GENCLASSIFIER,
]
ACTIVE_PROFILES = [p["name"] for p in PROFILES]

CONFIG = {
    "run_mode": RUN_MODE,
    "data": deepcopy(DATA),
    "efficiency": deepcopy(EFFICIENCY),
    "fair_profiles": {
        "mrfa": deepcopy(FAIR_MRFA),
        "tagfex": deepcopy(FAIR_TAGFEX),
        "der_paper": deepcopy(FAIR_DER_PAPER),
        "beef": deepcopy(FAIR_BEEF),
        "ewc": deepcopy(FAIR_EWC),
        "icarl": deepcopy(FAIR_ICARL),
        "seed_paper": deepcopy(FAIR_SEED_PAPER),
        "tpl": deepcopy(FAIR_TPL),
        "pec": deepcopy(FAIR_PEC),
        "move_paper": deepcopy(FAIR_MOVE_PAPER),
        "sema": deepcopy(FAIR_SEMA),
        "moeadapterspp_paper": deepcopy(FAIR_MOEADAPTERSPP),
        "more_paper": deepcopy(FAIR_MORE_PAPER),
        "more_paper_deit": deepcopy(FAIR_MORE_PAPER_DEIT),
        "more_paper_resnet18": deepcopy(FAIR_MORE_PAPER_RESNET18),
        "build_paper": deepcopy(FAIR_BUILD_PAPER),
        "itaml_paper": deepcopy(FAIR_ITAML_PAPER),
        "diva": deepcopy(FAIR_DIVA),
        "mfgr_paper": deepcopy(FAIR_MFGR_PAPER),
        "gfril_paper": deepcopy(FAIR_GFRIL_PAPER),
        "genclassifier": deepcopy(FAIR_GENCLASSIFIER),
    },
    "profiles": deepcopy(PROFILES),
    "external_incremental_methods": deepcopy(EXTERNAL_INCREMENTAL_METHODS),
    "active_profiles": deepcopy(ACTIVE_PROFILES),
}
