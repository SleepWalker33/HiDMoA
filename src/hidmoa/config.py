"""HiDMoA 配置（当前仅保留 incremental_2，默认 hidmoa 跑法）。"""

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
    """Shared-MoE configurations require the same expert count on every configured MoE layer."""
    base = COMMON_BACKBONE_CONFIGS[backbone]
    counts = _normalize_experts_per_task(base["experts_per_task"], base["moe_layers"])
    if len(set(counts)) != 1:
        raise ValueError(
            f"{backbone} experts_per_task must be identical across moe_layers; got {counts}"
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
# - 默认内置 neu_xsdd；新增预设可按需在 DATASET_PRESETS 中追加，需同步 DATA_ROOTS/BASE_DATASET_PRESETS。
# - 运行时会校验 ACTIVE_DATASET 必须存在于 DATASET_PRESETS 中。
# ============================================================
DATA_ROOTS = {
    "neu": os.getenv("CIL_NEU_ROOT", _repo_data_path("neudata_yolo_701515")),
    "xsdd": os.getenv("CIL_XSDD_ROOT", _repo_data_path("xsdd_yolo_cls_701515")),
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
    "neu": _single_task_preset(NEU_CLASSES),
    "xsdd": _single_task_preset(XSDD_CLASSES),
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


DATASET_PRESETS = deepcopy(BASE_DATASET_PRESETS)
DATASET_PRESETS["neu_xsdd"] = _compose_dataset_chain("neu", "xsdd")
ACTIVE_DATASET = os.getenv("CIL_ACTIVE_DATASET", "neu_xsdd")


def _build_data_config(active_dataset: str):
    if active_dataset not in DATASET_PRESETS:
        raise KeyError(f"Unknown ACTIVE_DATASET: {active_dataset}")

    preset = deepcopy(DATASET_PRESETS[active_dataset])
    dataset_keys = [str(k) for k in preset.get("datasets", ())]
    if not dataset_keys:
        # 兼容旧配置；当前仓库仅使用 neu_xsdd，因此通常不会走这里
        return {
            "data_root": DATA_ROOTS[active_dataset],
            "dataset_class_names": [preset["class_names"]],
            "image_size": 224,
            "num_classes": len(preset["class_names"]),
            "class_names": preset["class_names"],
            "task_splits": preset["task_splits"],
            "num_workers": int(os.getenv("CIL_NUM_WORKERS", "0")),
            "seed": int(os.getenv("CIL_SEED", "42")),
            "repeats": int(os.getenv("CIL_REPEATS", "3")),
        }

    data_root = [DATA_ROOTS[k] for k in dataset_keys]
    dataset_class_names = [DATASET_PRESETS[k]["class_names"] for k in dataset_keys]

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
# 固定走 incremental_2（hidmoa）
# ============================================================

RUN_MODE = os.getenv("RUN_MODE", "incremental_2")

# 复现性：main.py 默认开启（CIL_DETERMINISTIC=1）。
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
#   all            — HiDMoA 当前方法严格复现
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
# 场景2：增量学习2（incremental_2，HiDMoA）
# 仅保留当前方法与参数；其他场景已移除。
# ============================================================
INCREMENTAL_2 = {
    "name": "HiDMoA",
    "model": _common_backbone_model(
        "resnet18",  # default backbone
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


PROFILES = [
    INCREMENTAL_2,
]
ACTIVE_PROFILES = [p["name"] for p in PROFILES]

CONFIG = {
    "run_mode": RUN_MODE,
    "data": deepcopy(DATA),
    "efficiency": deepcopy(EFFICIENCY),
    "fair_profiles": {},
    "profiles": deepcopy(PROFILES),
    "active_profiles": deepcopy(ACTIVE_PROFILES),
}
