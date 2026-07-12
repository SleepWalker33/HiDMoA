"""
NEU 钢铁缺陷 — 类增量学习 & 全量基线 主入口
"""

import os
import sys
import json
import shlex
import subprocess
import time
import re
import selectors
import math
from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

_BASELINES_COMMON = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines_0610", "common")
)
if os.path.isdir(_BASELINES_COMMON) and _BASELINES_COMMON not in sys.path:
    sys.path.insert(0, _BASELINES_COMMON)
try:
    from deterministic import (
        apply_torch_deterministic,
        bootstrap_deterministic_env,
        deterministic_enabled,
        merge_deterministic_env,
    )
except ImportError:
    def apply_torch_deterministic(seed: int) -> int:
        import random as _random

        import numpy as _np
        seed_i = int(seed)
        _random.seed(seed_i)
        _np.random.seed(seed_i)
        torch.manual_seed(seed_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_i)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return seed_i

    def bootstrap_deterministic_env(seed=None) -> None:
        seed_i = int(seed if seed is not None else os.environ.get("CIL_SEED", "0"))
        os.environ["PYTHONHASHSEED"] = str(seed_i)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
        )

    def deterministic_enabled() -> bool:
        raw = os.environ.get("CIL_DETERMINISTIC", "1").strip().lower()
        return raw not in ("0", "false", "no", "off", "disable", "disabled")

    def merge_deterministic_env(env=None, seed=None) -> Dict[str, str]:
        merged = dict(env or os.environ)
        if deterministic_enabled():
            bootstrap_deterministic_env(seed)
        merged.update(
            {
                "CIL_DETERMINISTIC": os.environ.get("CIL_DETERMINISTIC", "1"),
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "0"),
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            }
        )
        if seed is not None:
            merged["CIL_SEED"] = str(int(seed))
        return merged

import numpy as np
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
import torch.nn.functional as F

from .config import (
    ACTIVE_DATASET,
    COMMON_HEAD,
    DATA,
    EFFICIENCY,
    EXTERNAL_INCREMENTAL_METHODS,
    FULL_1,
    FULL_2,
    INCREMENTAL_1,
    INCREMENTAL_2,
    INCREMENTAL_3,
    RUN_MODE,
    apply_internal_run_seed,
    cil_deterministic_enabled,
    experiment_seeds,
    internal_strict_reproducibility_enabled,
)
from .data import IMAGENET_MEAN, IMAGENET_STD, build_task_loaders, prepare_datasets
from .generators import sample_generator_images, train_generator
from .profiling import (
    EfficiencyTracker,
    accumulate_generator_flops_split,
    count_loader_samples,
    estimate_eval_flops_from_steps,
    estimate_train_flops_from_steps,
    is_profile_flops_enabled,
    measure_full_forward_per_image,
    measure_incremental1_forward_per_image,
    measure_incremental_top1_forward_per_image,
    measure_incremental1_train_step_flops,
    measure_incremental2_forward_per_image,
    profile_module_forward_flops,
    save_efficiency,
    set_profile_flops_enabled,
)
from .logger import (
    TrainLogger,
    aggregate_report_section,
    create_profile_run_dir,
    create_root_run_dir,
    plot_loss_curves,
    profile_subdir,
    save_config,
    save_router_stats,
    save_summary,
    write_cost_report,
    write_seed_test_report,
    write_test_report,
)
from .models import (
    CosineHead,
    CosinePrototypeHead,
    FullMoEResNet,
    IncrementalMoEResNet,
    LinearHead,
    PrototypeHead,
    StandardPrototypeHead,
    FrozenFeatureExtractor,
)
from .train import (
    collect_incremental_routing_stats,
    collect_task_images,
    evaluate_all,
    get_incremental_2_alpha_cache,
    _search_incremental_2_task_router_alpha,
    train_feature_cvae,
    train_feature_task_id_classifier,
    train_feature_vae,
    train_feature_vqvae,
    train_task_experts,
    train_task_id_classifier,
    reset_incremental_2_alpha_cache,
    _set_incremental_2_alpha_cache,
)
from .train_full import evaluate_full, evaluate_full_fixed, train_full, train_full_fixed


def _normalize_run_mode(mode: str) -> str:
    mode = str(mode).strip()
    legacy_internal_aliases = {
        "incre_incre1": "incremental_1",
        "incre_incre2": "incremental_2",
        "incre_incre3": "incremental_3",
        "hidmoa": "incremental_2",
        "HiDMoA": "incremental_2",
    }
    if mode in legacy_internal_aliases:
        alias = legacy_internal_aliases[mode]
        print(f"[warn] RUN_MODE '{mode}' remapped to '{alias}'")
        return alias
    if mode in EXTERNAL_INCREMENTAL_METHODS:
        return mode
    if mode.startswith("incremental_"):
        alias = "incre_" + mode[len("incremental_") :]
        if alias in EXTERNAL_INCREMENTAL_METHODS:
            print(f"[warn] RUN_MODE '{mode}' remapped to '{alias}'")
            return alias
    return mode


def set_seed(seed: int):
    apply_internal_run_seed(int(seed))


def count_params(model, trainable_only=True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _imprint_or_fail(head, feat_parts, label_parts, class_ids, context: str):
    if not feat_parts:
        raise ValueError(f"{context}: no samples found for classes {class_ids}")
    if not hasattr(head, "imprint"):
        return
    head.imprint(torch.cat(feat_parts), torch.cat(label_parts), class_ids)


_HEAD_TYPES = ("prototype", "cosine_prototype", "linear", "cos")
_PROTOTYPE_MU_MODES = ("learnable", "post_train_imprint")


def _resolve_head_type(cfg_t: dict) -> str:
    head_type = str(cfg_t.get("head_type", COMMON_HEAD.get("head_type", "prototype"))).strip().lower()
    if head_type not in _HEAD_TYPES:
        raise ValueError(f"unsupported head_type: {head_type!r}; choose one of {_HEAD_TYPES}")
    return head_type


def _resolve_prototype_mu_mode(cfg_t: dict) -> str:
    mode = str(cfg_t.get("prototype_mu_mode", COMMON_HEAD.get("prototype_mu_mode", "learnable"))).strip().lower()
    if mode not in _PROTOTYPE_MU_MODES:
        raise ValueError(f"unsupported prototype_mu_mode: {mode!r}; choose one of {_PROTOTYPE_MU_MODES}")
    return mode


def _build_defect_head(cfg_m: dict, cfg_t: dict, num_classes: int, device: torch.device) -> torch.nn.Module:
    head_type = _resolve_head_type(cfg_t)
    if head_type == "prototype":
        return StandardPrototypeHead(cfg_m["feat_dim"], num_classes, cfg_m["scale"]).to(device)
    if head_type == "cosine_prototype":
        return CosinePrototypeHead(cfg_m["feat_dim"], num_classes, cfg_m["scale"]).to(device)
    if head_type == "cos":
        return CosineHead(cfg_m["feat_dim"], num_classes, cfg_m["scale"]).to(device)
    return LinearHead(cfg_m["feat_dim"], num_classes).to(device)


@torch.no_grad()
def _refresh_head_prototypes_from_loader(
    model: IncrementalMoEResNet,
    head: torch.nn.Module,
    loader,
    task_id: int,
    task_classes: list,
    device: torch.device,
    context: str,
) -> None:
    """训后在无增强 train_eval 上按类特征均值重算 μ（覆盖 checkpoint 中的 prototypes）。"""
    model.eval()
    head.eval()
    feat_parts, label_parts = [], []
    for imgs, lbls in loader:
        feat_parts.append(model(imgs.to(device), task_id))
        label_parts.append(lbls)
    _imprint_or_fail(head, feat_parts, label_parts, task_classes, context)


def prepare_data():
    d = DATA
    return prepare_datasets(
        d["data_root"],
        d["class_names"],
        d["image_size"],
        dataset_class_names=d.get("dataset_class_names", None),
    )


def _models_dir(root_run_dir: str) -> str:
    path = os.path.join(root_run_dir, "models")
    os.makedirs(path, exist_ok=True)
    return path


def _profile_models_dir(run_dir: str) -> str:
    path = os.path.join(run_dir, "models")
    os.makedirs(path, exist_ok=True)
    return path


def _generator_image_size(cfg_t: Dict, generator_type: str) -> int:
    if generator_type in {"cvae", "vae"} and "vae" in cfg_t:
        return int(cfg_t["vae"]["image_size"])
    if generator_type == "vqvae":
        if "vqvae" in cfg_t:
            return int(cfg_t["vqvae"]["image_size"])
        return int(cfg_t["vqvae_image_size"])
    if generator_type in {"fvae", "fcvae", "fvqvae"}:
        for section in (generator_type, "fvae", "fcvae", "fvqvae"):
            if section in cfg_t and "image_size" in cfg_t[section]:
                return int(cfg_t[section]["image_size"])
        return int(cfg_t.get("taskid_image_size", int(cfg_t.get("cvae_image_size", 224))))
    if generator_type == "diffusion_lora":
        if "diffusion_lora" in cfg_t:
            return int(cfg_t["diffusion_lora"]["image_size"])
        return int(cfg_t["diffusion_lora_image_size"])
    return int(cfg_t["cvae_image_size"])


def _generator_num_per_class(cfg_t: Dict, generator_type: str) -> int:
    if generator_type in {"cvae", "vae"} and "vae" in cfg_t:
        return int(cfg_t["vae"]["generated_per_class"])
    if generator_type == "vqvae":
        if "vqvae" in cfg_t:
            return int(cfg_t["vqvae"]["generated_per_class"])
        return int(cfg_t["vqvae_generated_per_class"])
    if generator_type == "diffusion_lora":
        if "diffusion_lora" in cfg_t:
            return int(cfg_t["diffusion_lora"]["generated_per_class"])
        return int(cfg_t["diffusion_lora_generated_per_class"])
    if generator_type in {"fvae", "fcvae", "fvqvae"}:
        for section in (generator_type, "fvae", "fcvae", "fvqvae"):
            if section in cfg_t and "generated_per_class" in cfg_t[section]:
                return int(cfg_t[section]["generated_per_class"])
        return int(cfg_t.get("vae_generated_per_class", 200))
    return int(cfg_t["vae_generated_per_class"])


def _fvae_cfg(cfg_t: Dict) -> Dict:
    cfg = dict(cfg_t.get("fvae", {}))
    cfg["h_dim"] = int(cfg.get("h_dim", 512))
    cfg["z_dim"] = int(cfg.get("z_dim", 128))
    cfg["epochs"] = int(cfg.get("epochs", cfg_t.get("vae", {}).get("epochs", 200)))
    cfg["early_stopping_patience"] = int(cfg.get("early_stopping_patience", cfg_t.get("vae", {}).get("early_stopping_patience", 20)))
    cfg["early_stopping_min_delta"] = float(cfg.get("early_stopping_min_delta", cfg_t.get("vae", {}).get("early_stopping_min_delta", 1e-4)))
    cfg["lr"] = float(cfg.get("lr", cfg_t.get("vae", {}).get("lr", 1e-3)))
    cfg["weight_decay"] = float(cfg.get("weight_decay", cfg_t.get("vae", {}).get("weight_decay", 1e-5)))
    cfg["beta_kl"] = float(cfg.get("beta_kl", cfg_t.get("vae", {}).get("beta_kl", 0.01)))
    cfg["kl_warmup_epochs"] = int(cfg.get("kl_warmup_epochs", 0))
    cfg["recon_weight"] = float(cfg.get("recon_weight", 1.0))
    cfg["batch_size"] = int(cfg.get("batch_size", cfg_t.get("vae", {}).get("batch_size", 64)))
    cfg["fvae_feature_batch"] = int(cfg.get("fvae_feature_batch", cfg_t.get("vae", {}).get("vae_batch_size", 64)))
    return cfg


def _fcvae_cfg(cfg_t: Dict) -> Dict:
    cfg = _fvae_cfg(cfg_t)
    extra = dict(cfg_t.get("fcvae", {}))
    for key in ("h_dim", "z_dim", "epochs", "early_stopping_patience", "early_stopping_min_delta", "lr",
                "weight_decay", "beta_kl", "kl_warmup_epochs", "recon_weight", "batch_size",
                "fvae_feature_batch", "generated_per_class", "latent_pool_noise_std"):
        if key in extra:
            cfg[key] = extra[key]
    cfg["h_dim"] = int(cfg.get("h_dim", 512))
    cfg["z_dim"] = int(cfg.get("z_dim", 128))
    cfg["epochs"] = int(cfg.get("epochs", 200))
    cfg["early_stopping_patience"] = int(cfg.get("early_stopping_patience", 20))
    cfg["early_stopping_min_delta"] = float(cfg.get("early_stopping_min_delta", 1e-4))
    cfg["lr"] = float(cfg.get("lr", 1e-3))
    cfg["weight_decay"] = float(cfg.get("weight_decay", 1e-5))
    cfg["beta_kl"] = float(cfg.get("beta_kl", 0.01))
    cfg["kl_warmup_epochs"] = int(cfg.get("kl_warmup_epochs", 0))
    cfg["recon_weight"] = float(cfg.get("recon_weight", 1.0))
    cfg["batch_size"] = int(cfg.get("batch_size", 64))
    cfg["fvae_feature_batch"] = int(cfg.get("fvae_feature_batch", 64))
    cfg["generated_per_class"] = int(cfg.get("generated_per_class", cfg_t.get("vae_generated_per_class", 200)))
    cfg["latent_pool_noise_std"] = float(cfg.get("latent_pool_noise_std", 0.0))
    return cfg


def _fvqvae_cfg(cfg_t: Dict) -> Dict:
    cfg = dict(cfg_t.get("fvqvae", {}))
    fallback_vqvae = cfg_t.get("vqvae", {})
    fallback_fvae = cfg_t.get("fvae", {})
    cfg["h_dim"] = int(cfg.get("h_dim", fallback_fvae.get("h_dim", 512)))
    cfg["embedding_dim"] = int(cfg.get("embedding_dim", fallback_vqvae.get("embedding_dim", 128)))
    cfg["num_embeddings"] = int(cfg.get("num_embeddings", fallback_vqvae.get("num_embeddings", 512)))
    cfg["commitment_cost"] = float(cfg.get("commitment_cost", fallback_vqvae.get("commitment_cost", 0.25)))
    cfg["codebook_weight"] = float(cfg.get("codebook_weight", fallback_vqvae.get("codebook_weight", 1.0)))
    cfg["ema_decay"] = float(cfg.get("ema_decay", fallback_vqvae.get("ema_decay", 0.99)))
    cfg["epochs"] = int(cfg.get("epochs", fallback_vqvae.get("epochs", 200)))
    cfg["early_stopping_patience"] = int(cfg.get("early_stopping_patience", fallback_vqvae.get("early_stopping_patience", 20)))
    cfg["early_stopping_min_delta"] = float(cfg.get("early_stopping_min_delta", fallback_vqvae.get("early_stopping_min_delta", 1e-4)))
    cfg["lr"] = float(cfg.get("lr", fallback_vqvae.get("lr", 1e-3)))
    cfg["weight_decay"] = float(cfg.get("weight_decay", fallback_vqvae.get("weight_decay", 1e-5)))
    cfg["recon_weight"] = float(cfg.get("recon_weight", fallback_vqvae.get("recon_weight", 1.0)))
    cfg["batch_size"] = int(cfg.get("batch_size", fallback_vqvae.get("batch_size", 64)))
    cfg["fvae_feature_batch"] = int(cfg.get("fvae_feature_batch", fallback_fvae.get("fvae_feature_batch", 64)))
    cfg["generated_per_class"] = int(cfg.get("generated_per_class", fallback_vqvae.get("generated_per_class", cfg_t.get("vae_generated_per_class", 200))))
    return cfg


def _fvae_mlp_taskid_cfg(cfg_t: Dict) -> Dict:
    base_fvae = _fvae_cfg(cfg_t)
    cfg = dict(cfg_t.get("fvae_mlp_taskid", {}))
    cfg["hidden_dim"] = int(cfg.get("hidden_dim", base_fvae.get("h_dim", 512)))
    cfg["hidden_layers"] = int(cfg.get("hidden_layers", 2))
    cfg["dropout"] = float(cfg.get("dropout", 0.0))
    cfg["epochs"] = int(cfg.get("epochs", cfg_t.get("epochs_per_task", 50)))
    cfg["batch_size"] = int(cfg.get("batch_size", base_fvae.get("batch_size", cfg_t.get("batch_size", 64))))
    cfg["lr"] = float(cfg.get("lr", cfg_t.get("lr", 1e-3)))
    cfg["weight_decay"] = float(cfg.get("weight_decay", cfg_t.get("weight_decay", 1e-4)))
    cfg["early_stopping_patience"] = cfg.get("early_stopping_patience", cfg_t.get("early_stopping_patience", 5))
    cfg["early_stopping_min_delta"] = float(cfg.get("early_stopping_min_delta", cfg_t.get("early_stopping_min_delta", 1e-4)))
    cfg["use_cosine_scheduler"] = bool(cfg.get("use_cosine_scheduler", True))
    cfg["continue_from_prev_session"] = bool(cfg.get("continue_from_prev_session", True))
    cfg["generated_per_old_task"] = int(cfg.get("generated_per_old_task", 0))
    cfg["generated_val_per_old_task"] = int(cfg.get("generated_val_per_old_task", 0))
    cfg["current_task_use_generated"] = bool(cfg.get("current_task_use_generated", False))
    cfg["generated_per_current_task"] = int(cfg.get("generated_per_current_task", 0))
    cfg["generated_val_per_current_task"] = int(cfg.get("generated_val_per_current_task", 0))
    cfg["generated_val_ratio"] = float(cfg.get("generated_val_ratio", cfg_t.get("vae_router_aux_val_ratio", 0.3)))
    return cfg


def _fvae_generated_prototype_cfg(cfg_t: Dict) -> Dict:
    base_fvae = _fvae_cfg(cfg_t)
    cfg = dict(cfg_t.get("fvae_generated_prototype", {}))
    cfg["num_samples_per_task"] = int(
        cfg.get(
            "num_samples_per_task",
            int(base_fvae.get("generated_per_class", 600)) * max(1, len(DATA.get("task_splits", [[0]])[0])),
        )
    )
    cfg["aggregation"] = str(cfg.get("aggregation", "mean")).strip().lower()
    cfg["metric"] = str(cfg.get("metric", "cosine")).strip().lower()
    cfg["normalize_features"] = bool(cfg.get("normalize_features", True))
    return cfg


@torch.no_grad()
def _sample_fvae_task_replay_features(
    generator_state: Dict,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    count = max(0, int(num_samples))
    if count <= 0:
        input_dim = int(generator_state.get("input_dim", 0))
        return torch.zeros((0, input_dim), dtype=torch.float32)
    if str(generator_state.get("type", "")).strip().lower() != "fvae":
        raise ValueError("FVAE + MLP task-id replay only supports task-level fVAE generators")
    model = generator_state.get("model", None)
    if model is None:
        raise ValueError("generator_state has no model for feature replay")
    model_device = next(model.parameters()).device
    if model_device != device:
        model = model.to(device)
    if hasattr(model, "sample"):
        local_labels = torch.zeros((count,), dtype=torch.long, device=device)
        replay = model.sample(local_labels, device=device)
    else:
        z_dim = int(getattr(model, "z_dim"))
        z = torch.randn((count, z_dim), device=device)
        replay = model.decode(z)
    if model_device != device:
        model = model.to(model_device)
        generator_state["model"] = model
    return replay.detach().cpu()


@torch.no_grad()
def _build_generated_task_prototype(
    generator_state: Dict,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    samples = _sample_fvae_task_replay_features(generator_state, num_samples, device)
    if int(samples.size(0)) <= 0:
        input_dim = int(generator_state.get("input_dim", 0))
        return torch.zeros((input_dim,), dtype=torch.float32)
    return samples.float().mean(dim=0)


def _build_vae_router_feature_extractor(cfg_t: Dict) -> Optional[FrozenFeatureExtractor]:
    if not bool(cfg_t.get("vae_router_use_feature_space", False)):
        return None
    return FrozenFeatureExtractor(
        backbone=str(cfg_t.get("vae_router_backbone", "resnet18")),
        pretrained=bool(cfg_t.get("vae_router_backbone_pretrained", True)),
        image_size=int(cfg_t.get("vae_router_backbone_image_size", cfg_t.get("taskid_image_size", 224))),
    )


def _resolve_incremental1_generator_cfg(
    cfg_t: Dict,
    generator_type: str,
    task_classes: List[int],
    class_names: List[str],
) -> Dict:
    del task_classes, class_names
    cfg_local = dict(cfg_t)
    if generator_type != "vqvae":
        return cfg_local

    if "vqvae" in cfg_t:
        cfg_local["vqvae_profile"] = "vqvae"
    return cfg_local


def _move_generator_state_to_cpu(generator_state: Dict) -> None:
    gen_type = generator_state.get("type", "")
    if gen_type in {"vae", "cvae", "vqvae", "fvae", "fcvae", "fvqvae"} and generator_state.get("model", None) is not None:
        generator_state["model"] = generator_state["model"].to(torch.device("cpu"))
    if gen_type == "vqvae" and generator_state.get("pixelcnn_prior", None) is not None:
        generator_state["pixelcnn_prior"] = generator_state["pixelcnn_prior"].to(torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_generator_state_to_device(generator_state: Dict, device: torch.device) -> None:
    gen_type = generator_state.get("type", "")
    if gen_type in {"vae", "cvae", "vqvae", "fvae", "fcvae", "fvqvae"} and generator_state.get("model", None) is not None:
        generator_state["model"] = generator_state["model"].to(device)
    if gen_type == "vqvae" and generator_state.get("pixelcnn_prior", None) is not None:
        generator_state["pixelcnn_prior"] = generator_state["pixelcnn_prior"].to(device)


_VAE_ROUTER_LOG2PI = math.log(2.0 * math.pi)


def _imagenet_stats(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return mean, std


def _denormalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    mean, std = _imagenet_stats(images)
    return (images * std + mean).clamp(0.0, 1.0)


def _router_resize_and_unnormalize(images: torch.Tensor, image_size: int) -> torch.Tensor:
    resized = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return _denormalize_imagenet(resized)


def _class_vae_log_likelihood(
    vae_state: Dict,
    images: torch.Tensor,
    eval_importance_samples: int,
    *,
    feature_space: bool = False,
    score_mode: str = "is",
    class_id: Optional[int] = None,
    importance_batch_override: Optional[int] = None,
) -> torch.Tensor:
    gen_type = str(vae_state.get("type", "")).lower()
    vae = vae_state.get("model")
    if vae is None:
        raise ValueError("vae_state must contain a trained model")
    score_mode = str(score_mode).strip().lower()
    if score_mode not in {"is", "recon", "latent", "elbo_single", "elbo_single_mu", "elbo_single_sample", "elbo_k"}:
        score_mode = "is"
    if feature_space:
        x = images
    else:
        image_size = int(vae_state.get("image_size", images.shape[-1]))
        x = _router_resize_and_unnormalize(images, image_size)
    if x.numel() == 0:
        return x.new_zeros((0,), dtype=x.dtype)

    # Backward compatibility: 按输入形状和模型能力自动兜底。
    if (not gen_type or gen_type == "cvae") and hasattr(vae, "estimate_loglikelihood") and x.dim() == 2:
        gen_type = "fvae"
    if not gen_type and x.dim() == 4:
        gen_type = "cvae"

    k = max(1, int(eval_importance_samples))
    bsz = x.shape[0]
    importance_batch = (
        max(1, int(importance_batch_override))
        if importance_batch_override is not None
        else max(1, min(64, k, x.shape[0]))
    )
    class_ids = [int(c) for c in vae_state.get("class_ids", []) if c is not None]
    g2l = {g: i for i, g in enumerate(class_ids)}
    local_label_idx: Optional[int] = None
    if class_id is not None:
        local_label_idx = g2l.get(int(class_id))
    if local_label_idx is None:
        if len(class_ids) == 1:
            local_label_idx = 0
    local_labels = None if local_label_idx is None else torch.full((bsz,), int(local_label_idx), dtype=torch.long, device=x.device)

    if gen_type in {"fvae"}:
        if x.dim() != 2:
            x = x.reshape(x.size(0), -1)
        if not hasattr(vae, "estimate_loglikelihood"):
            raise ValueError(f"Feature VAE missing estimate_loglikelihood: {type(vae)}")
        if score_mode in {"elbo_single", "elbo_single_mu", "elbo_single_sample", "elbo_k"}:
            mu, logvar = vae.encode(x)
            sigma = torch.exp(0.5 * logvar)
            if score_mode == "elbo_single_sample":
                z = vae.reparameterize(mu, logvar)
            elif score_mode in {"elbo_single", "elbo_single_mu"}:
                z = mu
            if score_mode == "elbo_k":
                latent_dim = mu.shape[1]
                chunk = max(1, int(max(4, min(k, 32))))
                all_elbo = []
                for start in range(0, k, chunk):
                    cur = min(chunk, k - start)
                    eps = torch.randn((bsz, cur, latent_dim), device=x.device, dtype=x.dtype)
                    z = mu[:, None, :] + eps * sigma[:, None, :]
                    z_flat = z.reshape(-1, latent_dim)
                    recon = vae.decode(z_flat)
                    recon = recon.view(bsz, cur, -1)
                    recon_loss = F.mse_loss(recon, x[:, None], reduction="none").mean(dim=2)
                    kl_term = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)
                    recon_weight = float(vae_state.get("recon_weight", 1.0))
                    beta = float(vae_state.get("beta_kl", 1.0))
                    all_elbo.append(-(recon_weight * recon_loss + beta * kl_term[:, None]))
                if not all_elbo:
                    return x.new_zeros((bsz,), dtype=x.dtype)
                return torch.cat(all_elbo, dim=1).mean(dim=1)

            recon = vae.decode(z)
            recon_loss = F.mse_loss(recon, x, reduction="none").mean(dim=1)
            kl_term = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)
            recon_weight = float(vae_state.get("recon_weight", 1.0))
            beta = float(vae_state.get("beta_kl", 1.0))
            return -(recon_weight * recon_loss + beta * kl_term)
        if score_mode == "recon":
            mu, logvar = vae.encode(x)
            sigma = torch.exp(0.5 * logvar)
            if k <= 1:
                z = vae.reparameterize(mu, logvar)
                recon = vae.decode(z)
                return -((recon - x).pow(2).mean(dim=1))

            latent_dim = mu.shape[1]
            chunk = max(1, int(max(4, min(k, 32))))
            all_recon = []
            for start in range(0, k, chunk):
                cur = min(chunk, k - start)
                eps = torch.randn((bsz, cur, latent_dim), device=x.device, dtype=x.dtype)
                z = mu[:, None, :] + eps * sigma[:, None, :]
                recon = vae.decode(z.reshape(-1, latent_dim)).view(bsz, cur, -1)
                all_recon.append(-((recon - x[:, None]).pow(2).mean(dim=2)))
            if not all_recon:
                return x.new_zeros((bsz,), dtype=x.dtype)
            return torch.cat(all_recon, dim=1).mean(dim=1)
        if score_mode == "latent":
            mu, logvar = vae.encode(x)
            return -0.5 * (mu.pow(2) + logvar.exp()).sum(dim=1)
        return vae.estimate_loglikelihood(x, S=k, is_batch=importance_batch)

    if gen_type == "fcvae":
        if x.dim() != 2:
            x = x.reshape(x.size(0), -1)
        if not hasattr(vae, "estimate_loglikelihood"):
            raise ValueError(f"Feature conditional VAE missing estimate_loglikelihood: {type(vae)}")

        def _score_with_local_labels(y_local: torch.Tensor) -> torch.Tensor:
            mu, logvar = vae.encode(x, y_local)
            sigma = torch.exp(0.5 * logvar)
            if score_mode in {"elbo_single", "elbo_single_mu", "elbo_single_sample", "elbo_k"}:
                if score_mode == "elbo_single_sample":
                    z = vae.reparameterize(mu, logvar)
                else:
                    z = mu
                if score_mode == "elbo_k":
                    latent_dim = mu.shape[1]
                    chunk = max(1, int(max(4, min(k, 32))))
                    all_elbo = []
                    for start in range(0, k, chunk):
                        cur = min(chunk, k - start)
                        eps = torch.randn((bsz, cur, latent_dim), device=x.device, dtype=x.dtype)
                        z = mu[:, None, :] + eps * sigma[:, None, :]
                        z_flat = z.reshape(-1, latent_dim)
                        y_flat = y_local.repeat_interleave(cur)
                        recon = vae.decode(z_flat, y_flat).view(bsz, cur, -1)
                        recon_loss = F.mse_loss(recon, x[:, None], reduction="none").mean(dim=2)
                        kl_term = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)
                        recon_weight = float(vae_state.get("recon_weight", 1.0))
                        beta = float(vae_state.get("beta_kl", 1.0))
                        all_elbo.append(-(recon_weight * recon_loss + beta * kl_term[:, None]))
                    if not all_elbo:
                        return x.new_zeros((bsz,), dtype=x.dtype)
                    return torch.cat(all_elbo, dim=1).mean(dim=1)

                recon = vae.decode(z, y_local)
                recon_loss = F.mse_loss(recon, x, reduction="none").mean(dim=1)
                kl_term = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)
                recon_weight = float(vae_state.get("recon_weight", 1.0))
                beta = float(vae_state.get("beta_kl", 1.0))
                return -(recon_weight * recon_loss + beta * kl_term)
            if score_mode == "latent":
                return -0.5 * (mu.pow(2) + logvar.exp()).sum(dim=1)
            if score_mode == "recon":
                if k <= 1:
                    z = vae.reparameterize(mu, logvar)
                    recon = vae.decode(z, y_local)
                    return -((recon - x).pow(2).mean(dim=1))

                latent_dim = mu.shape[1]
                chunk = max(1, int(max(4, min(k, 32))))
                all_recon = []
                for start in range(0, k, chunk):
                    cur = min(chunk, k - start)
                    eps = torch.randn((bsz, cur, latent_dim), device=x.device, dtype=x.dtype)
                    z = mu[:, None, :] + eps * sigma[:, None, :]
                    z_flat = z.reshape(-1, latent_dim)
                    y_flat = y_local.repeat_interleave(cur)
                    recon = vae.decode(z_flat, y_flat).view(bsz, cur, -1)
                    all_recon.append(-((recon - x[:, None]).pow(2).mean(dim=2)))
                if not all_recon:
                    return x.new_zeros((bsz,), dtype=x.dtype)
                return torch.cat(all_recon, dim=1).mean(dim=1)
            return vae.estimate_loglikelihood(x, y_local, S=k, is_batch=importance_batch)

        if local_labels is not None:
            return _score_with_local_labels(local_labels)

        if len(class_ids) <= 0:
            return x.new_zeros((bsz,), dtype=x.dtype)
        task_scores = []
        for local_idx in range(len(class_ids)):
            y_local = torch.full((bsz,), local_idx, dtype=torch.long, device=x.device)
            task_scores.append(_score_with_local_labels(y_local))
        return torch.logsumexp(torch.stack(task_scores, dim=1), dim=1) - math.log(float(len(class_ids)))

    if gen_type == "fvqvae":
        if x.dim() != 2:
            x = x.reshape(x.size(0), -1)
        if not hasattr(vae, "estimate_loglikelihood"):
            raise ValueError(f"Feature VQ-VAE missing estimate_loglikelihood: {type(vae)}")

        def _score_vq(y_local: torch.Tensor) -> torch.Tensor:
            if score_mode == "recon":
                recon, _, _, _ = vae(x, y_local)
                return -((recon - x).pow(2).mean(dim=1))
            if score_mode == "latent":
                _, _, vq_loss, _, _ = vae.encode(x, y_local)
                return -torch.full((bsz,), float(vq_loss.detach().item()), device=x.device, dtype=x.dtype)
            return vae.estimate_loglikelihood(x, y_local, S=1, is_batch=importance_batch)

        if local_labels is not None:
            return _score_vq(local_labels)

        if len(class_ids) <= 0:
            return x.new_zeros((bsz,), dtype=x.dtype)
        task_scores = []
        for local_idx in range(len(class_ids)):
            y_local = torch.full((bsz,), local_idx, dtype=torch.long, device=x.device)
            task_scores.append(_score_vq(y_local))
        return torch.logsumexp(torch.stack(task_scores, dim=1), dim=1) - math.log(float(len(class_ids)))

    if gen_type in {"vae", "cvae"}:
        def _score_with_local_labels(y_local: torch.Tensor) -> torch.Tensor:
            mu, logvar = vae.encode(x, y_local)
            sigma = torch.exp(0.5 * logvar)
            if score_mode in {"elbo_single", "elbo_single_mu", "elbo_single_sample", "elbo_k"}:
                if score_mode == "elbo_single_sample":
                    z = vae.reparameterize(mu, logvar)
                else:
                    z = mu
                if score_mode == "elbo_k":
                    latent_dim = mu.shape[1]
                    sigma = torch.exp(0.5 * logvar)
                    chunk = max(1, int(max(4, min(k, 32))))
                    all_elbo = []
                    for start in range(0, k, chunk):
                        cur = min(chunk, k - start)
                        eps = torch.randn((bsz, cur, latent_dim), device=x.device, dtype=x.dtype)
                        z = mu[:, None, :] + eps * sigma[:, None, :]
                        z_flat = z.reshape(-1, latent_dim)
                        y_flat = y_local.repeat_interleave(cur)
                        recon = vae.decode(z_flat, y_flat).view(bsz, cur, *x.shape[1:])
                        recon_loss = ((recon - x[:, None]).pow(2).mean(dim=(2, 3, 4)))
                        kl_term = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)
                        recon_weight = float(vae_state.get("recon_weight", 1.0))
                        beta = float(vae_state.get("beta_kl", 1.0))
                        all_elbo.append(-(recon_weight * recon_loss + beta * kl_term[:, None]))
                    if not all_elbo:
                        return x.new_zeros((bsz,), dtype=x.dtype)
                    return torch.cat(all_elbo, dim=1).mean(dim=1)

                recon = vae.decode(z, y_local)
                recon_loss = ((recon - x).pow(2).mean(dim=(1, 2, 3)))
                kl_term = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)
                recon_weight = float(vae_state.get("recon_weight", 1.0))
                beta = float(vae_state.get("beta_kl", 1.0))
                return -(recon_weight * recon_loss + beta * kl_term)
            if score_mode == "latent":
                return -0.5 * (mu.pow(2) + logvar.exp()).sum(dim=1)
            if score_mode == "recon":
                if k <= 1:
                    z = vae.reparameterize(mu, logvar)
                    recon = vae.decode(z, y_local)
                    return -((recon - x).pow(2).mean(dim=(1, 2, 3)))

                latent_dim = mu.shape[1]
                chunk = max(1, int(max(4, min(k, 32))))
                all_recon = []
                for start in range(0, k, chunk):
                    cur = min(chunk, k - start)
                    eps = torch.randn((bsz, cur, latent_dim), device=x.device, dtype=x.dtype)
                    z = mu[:, None, :] + eps * sigma[:, None, :]
                    z_flat = z.reshape(-1, latent_dim)
                    y_flat = y_local.repeat_interleave(cur)
                    recon = vae.decode(z_flat, y_flat).view(bsz, cur, *x.shape[1:])
                    all_recon.append(-((recon - x[:, None]).pow(2).mean(dim=(2, 3, 4))))
                if not all_recon:
                    return x.new_zeros((bsz,), dtype=x.dtype)
                return torch.cat(all_recon, dim=1).mean(dim=1)

            if k <= 1:
                z = vae.reparameterize(mu, logvar)
                recon = vae.decode(z, y_local)
                recon_score = -((recon - x).pow(2).mean(dim=(1, 2, 3)))
                log_q = -0.5 * (logvar + ((z - mu).pow(2) / sigma.pow(2)).clamp_min(1e-12) + _VAE_ROUTER_LOG2PI).sum(dim=1)
                log_p = -0.5 * (z.pow(2).sum(dim=1) + mu.shape[1] * _VAE_ROUTER_LOG2PI)
                return recon_score + log_p - log_q

            latent_dim = mu.shape[1]
            chunk = max(1, int(max(4, min(k, 32))))
            all_chunks = []
            for start in range(0, k, chunk):
                cur = min(chunk, k - start)
                eps = torch.randn((bsz, cur, latent_dim), device=x.device, dtype=x.dtype)
                z = mu[:, None, :] + eps * sigma[:, None, :]
                z_flat = z.reshape(-1, latent_dim)
                y_flat = y_local.repeat_interleave(cur)
                recon = vae.decode(z_flat, y_flat)
                recon = recon.view(bsz, cur, *x.shape[1:])
                recon_score = -((recon - x[:, None]).pow(2).mean(dim=(2, 3, 4)))

                mu_exp = mu[:, None, :].expand_as(z)
                logvar_exp = logvar[:, None, :].expand_as(z)
                log_q = -0.5 * (logvar_exp + ((z - mu_exp).pow(2) / logvar_exp.exp()).clamp_min(1e-12) + _VAE_ROUTER_LOG2PI).sum(dim=2)
                log_p = -0.5 * (z.pow(2).sum(dim=2) + mu.shape[1] * _VAE_ROUTER_LOG2PI)
                all_chunks.append((recon_score + log_p - log_q).detach())

                del eps, z, z_flat, y_flat, recon, recon_score, mu_exp, logvar_exp, log_p, log_q
            log_weight = torch.cat(all_chunks, dim=1)
            return torch.logsumexp(log_weight, dim=1) - math.log(float(k))

        if local_labels is not None:
            return _score_with_local_labels(local_labels)

        if len(class_ids) <= 0:
            return x.new_zeros((bsz,), dtype=x.dtype)
        task_scores = []
        for local_idx in range(len(class_ids)):
            y_local = torch.full((bsz,), local_idx, dtype=torch.long, device=x.device)
            task_scores.append(_score_with_local_labels(y_local))
        return torch.logsumexp(torch.stack(task_scores, dim=1), dim=1) - math.log(float(len(class_ids)))

    if gen_type == "vqvae":
        if local_labels is None:
            if len(class_ids) <= 0:
                return x.new_zeros((bsz,), dtype=x.dtype)
            task_scores = []
            for local_idx in range(len(class_ids)):
                y_local = torch.full((bsz,), local_idx, dtype=torch.long, device=x.device)
                recon, _, _, _ = vae(x, y_local)
                task_scores.append(-((recon - x).pow(2).mean(dim=(1, 2, 3))))
            return torch.logsumexp(torch.stack(task_scores, dim=1), dim=1) - math.log(float(len(class_ids)))

        recon, _, _, _ = vae(x, local_labels)
        return -((recon - x).pow(2).mean(dim=(1, 2, 3)))

    return torch.full((bsz,), float("-inf"), device=x.device)


def _class_prototype_score(
    features: torch.Tensor,
    prototype: torch.Tensor,
    metric: str,
    normalize: bool = True,
) -> torch.Tensor:
    if features.dim() != 2:
        raise ValueError(f"features must be 2D, got {tuple(features.shape)}")
    if prototype.dim() != 1:
        raise ValueError(f"prototype must be 1D, got {tuple(prototype.shape)}")
    if features.size(1) != prototype.size(0):
        raise ValueError(
            f"feature dim mismatch: features={features.size(1)}, prototype={prototype.size(0)}"
        )

    metric = str(metric).strip().lower()
    if metric in {"cos", "cosine"}:
        if normalize:
            features = F.normalize(features, dim=1)
            prototype = F.normalize(prototype, dim=0)
        return features @ prototype
    if metric in {"euclidean", "l2", "l2_norm"}:
        diff = features - prototype.unsqueeze(0)
        return -diff.pow(2).sum(dim=1)
    raise ValueError(f"unsupported prototype metric: {metric}")


def _extract_router_features(
    feature_extractor: Optional[FrozenFeatureExtractor],
    images: torch.Tensor,
    target_device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    if feature_extractor is None:
        raise ValueError("feature_extractor is required when vae_router_use_feature_space=True")
    if images.numel() == 0:
        return images.new_zeros((0, 0))

    feature_extractor.eval()
    start_device = next(feature_extractor.parameters()).device
    if start_device != target_device:
        feature_extractor.to(target_device)

    feature_parts = []
    with torch.no_grad():
        for start in range(0, images.size(0), batch_size):
            x = images[start:start + batch_size].to(target_device)
            feature_parts.append(feature_extractor(x).detach())

    feats = torch.cat(feature_parts, dim=0)
    if start_device != target_device:
        feature_extractor.to(start_device)
    return feats


def _profile_router_feature_per_sample(
    feature_extractor: Optional[FrozenFeatureExtractor],
    images: torch.Tensor,
    target_device: torch.device,
    batch_size: int = 64,
) -> int:
    if feature_extractor is None or images.numel() == 0:
        return 0
    probe_bs = max(1, min(int(batch_size), int(images.size(0))))
    probe_images = images[:probe_bs]
    flops = profile_module_forward_flops(feature_extractor, (probe_images,), target_device) or 0
    return max(int(flops) // max(probe_bs, 1), 1) if flops else 0


class _FeatureMlpTaskRouter(torch.nn.Module):
    def __init__(
        self,
        feature_extractor: FrozenFeatureExtractor,
        use_feature_space: bool = True,
        task_count: int = 0,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.use_feature_space = bool(use_feature_space)
        self.num_tasks = int(task_count)
        self.feature_classifier: Optional[torch.nn.Module] = None
        self._class_entries = []
        if self.feature_extractor is not None:
            self.feature_extractor.eval()
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def set_num_tasks(self, num_tasks: int) -> None:
        self.num_tasks = max(self.num_tasks, int(num_tasks))
        self._class_entries = [{"task_id": tid} for tid in range(self.num_tasks)] if self.num_tasks > 0 else []

    def set_feature_classifier(self, feature_classifier: torch.nn.Module) -> None:
        self.feature_classifier = feature_classifier
        self.set_num_tasks(int(getattr(feature_classifier, "num_tasks", 0)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        bsz = int(images.shape[0])
        device = images.device
        if bsz <= 0:
            return torch.zeros((0, self.num_tasks), dtype=images.dtype, device=device)
        if self.feature_classifier is None or self.num_tasks <= 0:
            return torch.full((bsz, self.num_tasks), float("-inf"), dtype=images.dtype, device=device)
        if self.use_feature_space:
            feats = _extract_router_features(self.feature_extractor, images, device, batch_size=64)
        else:
            feats = images.reshape(bsz, -1).to(device=device, dtype=torch.float32)

        clf_device = next(self.feature_classifier.parameters()).device
        if clf_device != device:
            self.feature_classifier.to(device)
        return self.feature_classifier(feats.to(device=device, dtype=torch.float32))


class _ClassVaeTaskRouter(torch.nn.Module):
    def __init__(
        self,
        task_count: int,
        eval_importance_samples: int = 200,
        aggregation: str = "logsumexp",
        use_class_prior: bool = False,
        feature_extractor: Optional[FrozenFeatureExtractor] = None,
        use_feature_space: bool = False,
        score_mode: str = "is",
    ):
        super().__init__()
        self.num_tasks = int(task_count)
        self.eval_importance_samples = max(1, int(eval_importance_samples))
        self.aggregation = str(aggregation).strip().lower()
        if self.aggregation not in {"sum", "mean", "max", "logsumexp"}:
            self.aggregation = "logsumexp"
        self.use_class_prior = bool(use_class_prior)
        self.score_mode = str(score_mode).strip().lower()
        if self.score_mode not in {"is", "recon", "latent", "elbo_single", "elbo_single_mu", "elbo_single_sample", "elbo_k"}:
            self.score_mode = "is"

        self.class_models = torch.nn.ModuleDict()
        self._class_entries = []
        self._seen_classes = set()
        self.use_feature_space = bool(use_feature_space)
        self.feature_extractor = feature_extractor
        self.profile_importance_batch: Optional[int] = None
        if self.use_feature_space and self.feature_extractor is None:
            raise ValueError("feature_space routing requires a feature_extractor")
        if self.feature_extractor is not None:
            self.feature_extractor.eval()
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def set_num_tasks(self, num_tasks: int) -> None:
        self.num_tasks = max(int(num_tasks), self.num_tasks)

    @torch.no_grad()
    def add_class_vae(
        self,
        class_id: int,
        task_id: int,
        vae_state: Dict,
        class_prior: float = 1.0,
    ) -> None:
        cls_key = str(int(class_id))
        if cls_key in self._seen_classes:
            return
        self.class_models[cls_key] = vae_state["model"]
        self._class_entries.append({
            "class_id": int(class_id),
            "task_id": int(task_id),
            "state": vae_state,
            "prior": float(class_prior),
            "model_key": cls_key,
        })
        self._seen_classes.add(cls_key)
        self.num_tasks = max(self.num_tasks, int(task_id) + 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        bsz = int(images.shape[0])
        device = images.device
        if bsz <= 0:
            return torch.zeros((0, self.num_tasks), dtype=images.dtype, device=device)

        if not self._class_entries:
            return torch.full((bsz, self.num_tasks), float("-inf"), device=device)
        route_inputs = images
        if self.use_feature_space:
            route_inputs = _extract_router_features(self.feature_extractor, images, device, batch_size=64)

        if self.class_models:
            model_device = next(self.class_models.parameters()).device
            if model_device != device:
                self.to(device)

        if self.aggregation == "mean":
            task_scores = torch.zeros((bsz, self.num_tasks), device=device)
            task_counts = torch.zeros((bsz, self.num_tasks), device=device)
            for entry in self._class_entries:
                score = _class_vae_log_likelihood(
                    entry["state"],
                    route_inputs,
                    self.eval_importance_samples,
                    feature_space=self.use_feature_space,
                    score_mode=self.score_mode,
                    class_id=entry.get("class_id", None),
                    importance_batch_override=self.profile_importance_batch,
                )
                if self.use_class_prior and entry["prior"] > 0:
                    score = score + torch.log(torch.tensor(entry["prior"], device=device))
                tid = int(entry["task_id"])
                if 0 <= tid < self.num_tasks:
                    task_scores[:, tid] += score
                    task_counts[:, tid] += 1.0
            task_scores = torch.where(task_counts > 0, task_scores / task_counts, task_scores.new_full((bsz, self.num_tasks), float("-inf")))
            return task_scores

        task_scores = torch.full((bsz, self.num_tasks), float("-inf"), device=device)
        for entry in self._class_entries:
            score = _class_vae_log_likelihood(
                entry["state"],
                route_inputs,
                self.eval_importance_samples,
                feature_space=self.use_feature_space,
                score_mode=self.score_mode,
                class_id=entry.get("class_id", None),
                importance_batch_override=self.profile_importance_batch,
            )
            if self.use_class_prior and entry["prior"] > 0:
                score = score + torch.log(torch.tensor(entry["prior"], device=device))
            tid = int(entry["task_id"])
            if not (0 <= tid < self.num_tasks):
                continue

            if self.aggregation == "sum":
                prev = task_scores[:, tid]
                task_scores[:, tid] = torch.where(torch.isneginf(prev), score, prev + score)
            elif self.aggregation == "max":
                task_scores[:, tid] = torch.maximum(task_scores[:, tid], score)
            else:
                prev = task_scores[:, tid]
                task_scores[:, tid] = torch.where(
                    torch.isneginf(prev),
                    score,
                    torch.logsumexp(torch.stack([prev, score], dim=0), dim=0),
                )

        return task_scores


class _ClassPrototypeTaskRouter(torch.nn.Module):
    def __init__(
        self,
        task_count: int,
        aggregation: str = "mean",
        feature_extractor: Optional[FrozenFeatureExtractor] = None,
        use_feature_space: bool = False,
        score_metric: str = "cosine",
        use_class_prior: bool = False,
        normalize_features: bool = True,
        use_ema: bool = False,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_tasks = int(task_count)
        self.aggregation = str(aggregation).strip().lower()
        if self.aggregation not in {"sum", "mean", "max", "logsumexp"}:
            self.aggregation = "mean"
        self.score_metric = str(score_metric).strip().lower()
        if self.score_metric not in {"cosine", "cos", "euclidean", "l2", "l2_norm"}:
            self.score_metric = "cosine"
        self.use_class_prior = bool(use_class_prior)
        self.normalize_features = bool(normalize_features)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        if not (0.0 <= self.ema_decay <= 1.0):
            self.ema_decay = 0.99

        self.class_models = torch.nn.ParameterDict()
        self._class_entries = []
        self._seen_classes = set()
        self._class_counts = {}
        self.use_feature_space = bool(use_feature_space)
        self.feature_extractor = feature_extractor
        if self.use_feature_space and self.feature_extractor is None:
            raise ValueError("feature_space routing requires a feature_extractor")
        if self.feature_extractor is not None:
            self.feature_extractor.eval()
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def set_num_tasks(self, num_tasks: int) -> None:
        self.num_tasks = max(int(num_tasks), self.num_tasks)

    @torch.no_grad()
    def add_class_prototype(
        self,
        class_id: int,
        task_id: int,
        prototype: torch.Tensor,
        class_prior: float = 1.0,
        class_count: Optional[int] = None,
    ) -> None:
        cls_key = str(int(class_id))
        proto = prototype.detach().to(dtype=torch.float32).reshape(-1)
        if not torch.isfinite(proto).all():
            return
        count = int(class_count) if class_count is not None and class_count > 0 else 1

        if cls_key in self._seen_classes:
            if cls_key not in self.class_models:
                self._seen_classes.discard(cls_key)
                return

            param = self.class_models[cls_key]
            if proto.device != param.device:
                proto = proto.to(device=param.device)
            if proto.shape != param.shape:
                return
            if proto.dtype != param.dtype:
                proto = proto.to(dtype=param.dtype)

            if self.use_ema:
                beta = float(self.ema_decay)
                param.mul_(beta).add_(proto, alpha=1.0 - beta)
            else:
                old_count = float(self._class_counts.get(cls_key, 1))
                new_count = old_count + float(count)
                param.copy_((param * old_count + proto * float(count)) / max(new_count, 1.0))
                self._class_counts[cls_key] = new_count
            if self.use_class_prior:
                for entry in self._class_entries:
                    if int(entry["class_id"]) == int(class_id):
                        entry["prior"] = float(class_prior)
                        break
            return

        if self.class_models:
            first_param = next(self.class_models.parameters())
            proto = proto.to(device=first_param.device, dtype=first_param.dtype)
        self.class_models[cls_key] = torch.nn.Parameter(proto, requires_grad=False)
        self._class_entries.append({
            "class_id": int(class_id),
            "task_id": int(task_id),
            "prior": float(class_prior),
            "model_key": cls_key,
        })
        self._seen_classes.add(cls_key)
        self._class_counts[cls_key] = float(count)
        self.num_tasks = max(self.num_tasks, int(task_id) + 1)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        bsz = int(images.shape[0])
        device = images.device
        if bsz <= 0:
            return torch.zeros((0, self.num_tasks), dtype=images.dtype, device=device)

        if not self._class_entries:
            return torch.full((bsz, self.num_tasks), float("-inf"), device=device)

        if self.use_feature_space:
            if self.feature_extractor is None:
                raise ValueError("feature_extractor is required for feature space prototype routing")
            route_inputs = _extract_router_features(self.feature_extractor, images, device, batch_size=64)
        else:
            route_inputs = images.reshape(bsz, -1).to(device=device, dtype=torch.float32)

        if self.class_models:
            model_device = next(self.class_models.parameters()).device
            if model_device != device:
                self.to(device)
                route_inputs = route_inputs.to(device=device)
                model_device = device

        if self.aggregation == "mean":
            task_scores = torch.zeros((bsz, self.num_tasks), device=device, dtype=route_inputs.dtype)
            task_counts = torch.zeros((bsz, self.num_tasks), device=device, dtype=route_inputs.dtype)
            for entry in self._class_entries:
                key = str(int(entry["class_id"]))
                if key not in self.class_models:
                    continue
                score = _class_prototype_score(
                    route_inputs,
                    self.class_models[key],
                    self.score_metric,
                    normalize=self.normalize_features,
                )
                if self.use_class_prior and entry["prior"] > 0:
                    score = score + torch.log(torch.tensor(entry["prior"], device=device, dtype=route_inputs.dtype))
                tid = int(entry["task_id"])
                if 0 <= tid < self.num_tasks:
                    task_scores[:, tid] += score
                    task_counts[:, tid] += 1.0
            task_scores = torch.where(
                task_counts > 0,
                task_scores / task_counts.clamp_min(1.0),
                task_scores.new_full((bsz, self.num_tasks), float("-inf")),
            )
            return task_scores

        task_scores = torch.full((bsz, self.num_tasks), float("-inf"), device=device, dtype=route_inputs.dtype)
        for entry in self._class_entries:
            key = str(int(entry["class_id"]))
            if key not in self.class_models:
                continue
            score = _class_prototype_score(
                route_inputs,
                self.class_models[key],
                self.score_metric,
                normalize=self.normalize_features,
            )
            if self.use_class_prior and entry["prior"] > 0:
                score = score + torch.log(torch.tensor(entry["prior"], device=device, dtype=route_inputs.dtype))
            tid = int(entry["task_id"])
            if not (0 <= tid < self.num_tasks):
                continue

            if self.aggregation == "sum":
                prev = task_scores[:, tid]
                task_scores[:, tid] = torch.where(torch.isneginf(prev), score, prev + score)
            elif self.aggregation == "max":
                task_scores[:, tid] = torch.maximum(task_scores[:, tid], score)
            else:
                prev = task_scores[:, tid]
                task_scores[:, tid] = torch.where(
                    torch.isneginf(prev),
                    score,
                    torch.logsumexp(torch.stack([prev, score], dim=0), dim=0),
                )

        return task_scores


def _release_model_train_memory(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.grad = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _task_score_mode_label(mode: str) -> str:
    return str(mode).strip().lower().replace("-", "_")


def _is_feature_prototype_score_mode(mode: str) -> bool:
    return _task_score_mode_label(mode) in {"feature_prototype", "prototype", "real_feature_prototype"}


def _write_task_score_matrix_csv(
    path: str,
    matrix: np.ndarray,
    *,
    row_prefix: str = "candidate_task",
    col_prefix: str = "true_task",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        header = [""] + [f"{col_prefix}{i + 1}" for i in range(matrix.shape[1])]
        f.write(",".join(header) + "\n")
        for row_idx in range(matrix.shape[0]):
            values = [f"{float(v):.10g}" for v in matrix[row_idx].tolist()]
            f.write(",".join([f"{row_prefix}{row_idx + 1}", *values]) + "\n")


def _write_task_score_samples_csv(path: str, rows: List[Dict], num_seen: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_cols = [f"raw_task{i + 1}" for i in range(num_seen)]
    prob_cols = [f"prob_task{i + 1}" for i in range(num_seen)]
    header = ["sample_index", "true_task", "pred_task", *raw_cols, *prob_cols]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            values = [
                str(int(row["sample_index"])),
                str(int(row["true_task"]) + 1),
                str(int(row["pred_task"]) + 1),
            ]
            values.extend(f"{float(v):.10g}" for v in row["raw"])
            values.extend(f"{float(v):.10g}" for v in row["prob"])
            f.write(",".join(values) + "\n")


def _plot_task_score_heatmap(
    matrix: np.ndarray,
    path: str,
    *,
    title: str,
    colorbar_label: str,
    cmap: str = "YlOrRd",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[task-score] skip png plot {path}: matplotlib unavailable ({exc})")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_rows, n_cols = matrix.shape
    fig_w = max(6.0, 0.72 * n_cols + 2.4)
    fig_h = max(5.2, 0.64 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("True task id", fontsize=11)
    ax.set_ylabel("Candidate / predicted task id", fontsize=11)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([str(i + 1) for i in range(n_cols)])
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([str(i + 1) for i in range(n_rows)])
    ax.tick_params(axis="both", labelsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(colorbar_label, rotation=270, labelpad=13)

    if n_rows <= 12 and n_cols <= 12:
        finite = matrix[np.isfinite(matrix)]
        threshold = float(finite.min() + 0.55 * (finite.max() - finite.min())) if finite.size else 0.0
        for i in range(n_rows):
            for j in range(n_cols):
                value = float(matrix[i, j])
                color = "white" if value >= threshold else "#222222"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def _dump_task_router_score_heatmaps(
    run_dir: str,
    task_router,
    test_loaders: List[torch.utils.data.DataLoader],
    *,
    num_seen: int,
    device: torch.device,
    modes: List[str],
    seed: Optional[int] = None,
) -> List[Dict]:
    if task_router is None:
        print("[task-score] skip: task router is None")
        return []
    if num_seen <= 1:
        print("[task-score] skip: num_seen <= 1")
        return []

    out_dir = os.path.join(run_dir, "task_router_score_heatmaps")
    os.makedirs(out_dir, exist_ok=True)
    has_score_mode = hasattr(task_router, "score_mode")
    old_mode = str(getattr(task_router, "score_mode")) if has_score_mode else None
    task_router.eval()
    task_router.to(device)
    report = []

    for mode in modes:
        mode_key = _task_score_mode_label(mode)
        is_feature_prototype = _is_feature_prototype_score_mode(mode_key)
        if is_feature_prototype:
            mode_key = "feature_prototype"
        if mode_key not in {"is", "recon", "elbo_k", "feature_prototype"}:
            print(f"[task-score] skip unsupported mode={mode!r}")
            continue
        if is_feature_prototype and task_router.__class__.__name__ != "_ClassPrototypeTaskRouter":
            print(f"[task-score] skip mode={mode_key}: task router is not a prototype router")
            continue
        if mode_key in {"is", "recon", "elbo_k"} and not has_score_mode:
            print(f"[task-score] skip mode={mode_key}: task router does not expose score_mode")
            continue
        if seed is not None:
            torch.manual_seed(int(seed) + 97 * (len(report) + 1))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed) + 97 * (len(report) + 1))

        if has_score_mode and not is_feature_prototype:
            setattr(task_router, "score_mode", mode_key)
        raw_sum = torch.zeros((num_seen, num_seen), dtype=torch.float64)
        prob_sum = torch.zeros((num_seen, num_seen), dtype=torch.float64)
        confusion = torch.zeros((num_seen, num_seen), dtype=torch.float64)
        counts = torch.zeros(num_seen, dtype=torch.float64)
        sample_rows: List[Dict] = []
        sample_index = 0

        for true_tid in range(num_seen):
            loader = test_loaders[true_tid]
            for images, _labels in loader:
                images = images.to(device)
                scores = task_router(images)[:, :num_seen].detach()
                probs = F.softmax(scores, dim=1)
                pred_tids = scores.argmax(dim=1)
                raw_sum[true_tid] += scores.double().sum(dim=0).cpu()
                prob_sum[true_tid] += probs.double().sum(dim=0).cpu()
                counts[true_tid] += int(images.size(0))
                for pred_tid in pred_tids.cpu().tolist():
                    confusion[int(pred_tid), true_tid] += 1.0

                scores_cpu = scores.cpu()
                probs_cpu = probs.cpu()
                preds_cpu = pred_tids.cpu()
                for row_idx in range(scores_cpu.size(0)):
                    sample_rows.append({
                        "sample_index": sample_index,
                        "true_task": true_tid,
                        "pred_task": int(preds_cpu[row_idx].item()),
                        "raw": [float(v) for v in scores_cpu[row_idx].tolist()],
                        "prob": [float(v) for v in probs_cpu[row_idx].tolist()],
                    })
                    sample_index += 1

        denom = counts.clamp_min(1.0).view(num_seen, 1)
        raw_mean_true_by_candidate = raw_sum / denom
        prob_mean_true_by_candidate = prob_sum / denom
        raw_matrix = raw_mean_true_by_candidate.t().numpy()
        prob_matrix = prob_mean_true_by_candidate.t().numpy()
        confusion_matrix = confusion.numpy()
        taskid_acc = float(confusion.diag().sum().item() / max(confusion.sum().item(), 1.0))

        raw_csv = os.path.join(out_dir, f"task_score_raw_mean_{mode_key}.csv")
        prob_csv = os.path.join(out_dir, f"task_score_prob_mean_{mode_key}.csv")
        confusion_csv = os.path.join(out_dir, f"task_score_confusion_{mode_key}.csv")
        sample_csv = os.path.join(out_dir, f"task_score_samples_{mode_key}.csv")
        raw_png = os.path.join(out_dir, f"task_score_raw_mean_{mode_key}.png")
        prob_png = os.path.join(out_dir, f"task_score_prob_mean_{mode_key}.png")
        confusion_png = os.path.join(out_dir, f"task_score_confusion_{mode_key}.png")

        _write_task_score_matrix_csv(raw_csv, raw_matrix)
        _write_task_score_matrix_csv(prob_csv, prob_matrix)
        _write_task_score_matrix_csv(confusion_csv, confusion_matrix)
        _write_task_score_samples_csv(sample_csv, sample_rows, num_seen)
        _plot_task_score_heatmap(
            raw_matrix,
            raw_png,
            title=f"Task router raw score heatmap ({mode_key})",
            colorbar_label="Mean raw score",
        )
        _plot_task_score_heatmap(
            prob_matrix,
            prob_png,
            title=f"Task router softmax score heatmap ({mode_key})",
            colorbar_label="Mean softmax score",
        )
        _plot_task_score_heatmap(
            confusion_matrix,
            confusion_png,
            title=f"Task router prediction counts ({mode_key})",
            colorbar_label="Count",
            cmap="Blues",
        )

        item = {
            "mode": mode_key,
            "num_seen": int(num_seen),
            "num_samples": int(sum(int(v.item()) for v in counts)),
            "taskid_acc": taskid_acc,
            "raw_mean_csv": raw_csv,
            "prob_mean_csv": prob_csv,
            "confusion_csv": confusion_csv,
            "sample_scores_csv": sample_csv,
            "raw_mean_png": raw_png,
            "prob_mean_png": prob_png,
            "confusion_png": confusion_png,
        }
        report.append(item)
        print(
            f"[task-score] mode={mode_key} samples={item['num_samples']} "
            f"taskid_acc={taskid_acc:.4f} out={out_dir}"
        )

    if has_score_mode and old_mode is not None:
        setattr(task_router, "score_mode", old_mode)
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def _save_incremental_session_model(
    run_dir: str,
    session_id: int,
    model: IncrementalMoEResNet,
    heads: list,
    task_id_clf,
    task_splits: list,
):
    save_path = os.path.join(_profile_models_dir(run_dir), f"incre1_session{session_id + 1}.pt")
    torch.save(
        {
            "session": session_id,
            "model_state": model.state_dict(),
            "heads": [h.state_dict() for h in heads],
            "task_id_classifier": task_id_clf.state_dict() if task_id_clf is not None else None,
            "task_splits": task_splits[: session_id + 1],
        },
        save_path,
    )


def _save_full_model(run_dir: str, filename: str, payload: dict):
    save_path = os.path.join(_profile_models_dir(run_dir), filename)
    torch.save(payload, save_path)


@torch.no_grad()
def _collect_full_routing_stats(model, loader, device):
    per_class = {}
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        layer_weights = {name: weights.cpu() for name, weights in model.routing_layer_weights(images).items()}
        for cls_id in labels.unique():
            cls_int = int(cls_id.item())
            mask = labels == cls_id
            class_layers = per_class.setdefault(cls_int, {})
            for layer_name, weights in layer_weights.items():
                class_layers.setdefault(layer_name, []).append(weights[mask])
    stats = {}
    for cls_id, layer_chunks in per_class.items():
        stats[cls_id] = {}
        for layer_name, chunks in layer_chunks.items():
            all_w = torch.cat(chunks, dim=0)
            stats[cls_id][layer_name] = {
                "mean": all_w.mean(dim=0).tolist(),
                "std": all_w.std(dim=0, unbiased=False).tolist(),
            }
    return stats


@torch.no_grad()
def _collect_full_fixed_routing_stats(model, loader, task_id, task_classes, device):
    per_class = {}
    model.eval()
    for images, labels in loader:
        mask = torch.zeros(len(labels), dtype=torch.bool)
        for cls in task_classes:
            mask |= labels == cls
        if mask.sum() == 0:
            continue
        layer_weights = {
            name: weights.cpu()
            for name, weights in model.routing_layer_weights(images[mask].to(device), task_id).items()
        }
        labels_task = labels[mask]
        for cls_id in labels_task.unique():
            cls_int = int(cls_id.item())
            cls_mask = labels_task == cls_id
            class_layers = per_class.setdefault(cls_int, {})
            for layer_name, weights in layer_weights.items():
                class_layers.setdefault(layer_name, []).append(weights[cls_mask])
    stats = {}
    for cls_id, layer_chunks in per_class.items():
        stats[cls_id] = {}
        for layer_name, chunks in layer_chunks.items():
            all_w = torch.cat(chunks, dim=0)
            stats[cls_id][layer_name] = {
                "mean": all_w.mean(dim=0).tolist(),
                "std": all_w.std(dim=0, unbiased=False).tolist(),
            }
    return stats


def _split_aux_images(images, labels, val_ratio: float, seed: int):
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"aux_val_ratio must be in (0, 1), got {val_ratio}")

    train_indices = []
    val_indices = []
    rng = np.random.default_rng(seed)

    for cls_id in sorted(labels.unique().tolist()):
        cls_idx = torch.nonzero(labels == cls_id, as_tuple=True)[0].cpu().numpy()
        if cls_idx.size == 0:
            continue
        rng.shuffle(cls_idx)
        if cls_idx.size == 1:
            # 单样本类只放入训练集，后续用全局兜底规则补齐验证集
            train_indices.extend(cls_idx.tolist())
            continue

        val_count = int(round(cls_idx.size * val_ratio))
        val_count = max(1, val_count)
        if val_count >= cls_idx.size:
            val_count = cls_idx.size // 2
        if val_count <= 0:
            val_count = 1
        if val_count >= cls_idx.size:
            # 兜底：至少保证训练集和验证集各有一条
            val_count = cls_idx.size - 1

        if val_count <= 0 or cls_idx.size - val_count <= 0:
            # 仅在极端极小类下触发；此处交由外层做全局兜底
            train_indices.extend(cls_idx.tolist())
            continue
        val_indices.extend(cls_idx[:val_count].tolist())
        train_indices.extend(cls_idx[val_count:].tolist())

    if not train_indices and not val_indices:
        raise ValueError("aux split produced empty train and val sets")

    if not val_indices:
        # 全量都落在训练集：从训练集中移一张作为验证样本
        if not train_indices:
            raise ValueError("aux split produced empty train and val sets")
        val_idx = train_indices.pop(0)
        val_indices.append(val_idx)

    if not train_indices:
        # 全量都落在验证集：从验证集中回退一张到训练集
        if not val_indices:
            raise ValueError("aux split produced empty train set")
        train_idx = val_indices.pop(0)
        train_indices.append(train_idx)

    train_idx = torch.tensor(train_indices, dtype=torch.long)
    val_idx = torch.tensor(val_indices, dtype=torch.long)
    return (images[train_idx], labels[train_idx]), (images[val_idx], labels[val_idx])


def _generated_aux_val_num_per_class(train_num_per_class: int, val_ratio: float) -> int:
    if train_num_per_class <= 0:
        raise ValueError(f"train_num_per_class must be > 0, got {train_num_per_class}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"aux_val_ratio must be in (0, 1), got {val_ratio}")
    return max(1, int(round(float(train_num_per_class) * float(val_ratio))))


def _default_zero_metrics(class_ids: List[int]) -> Dict:
    per_class = {}
    for cid in class_ids:
        per_class[int(cid)] = {"acc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "per_class": per_class,
        "macro_acc": 0.0,
        "micro_acc": 0.0,
        "macro_f1": 0.0,
        "micro_f1": 0.0,
    }


def _na_metrics_for_classes(class_ids: List[int], avg_acc: Optional[float] = None) -> Dict:
    per_class = {
        int(cid): {"acc": "NA", "f1": "NA", "precision": "NA", "recall": "NA"}
        for cid in class_ids
    }
    metrics = {
        "per_class": per_class,
        "macro_acc": "NA",
        "micro_acc": "NA",
        "macro_f1": "NA",
        "micro_f1": "NA",
        "old_acc": "NA",
        "new_acc": "NA",
        "total_acc": "NA",
    }
    if avg_acc is not None:
        v = max(0.0, min(1.0, float(avg_acc)))
        metrics["macro_acc"] = v
        metrics["micro_acc"] = v
        metrics["total_acc"] = v
    return metrics


def _infer_external_run_dir_from_summary(summary: Dict) -> Optional[str]:
    run_dir = str(summary.get("run_dir", "")).strip()
    if run_dir:
        return run_dir
    metrics_json = str(summary.get("metrics_json", "")).strip()
    if metrics_json:
        return os.path.dirname(metrics_json)
    # Preferred: non-sequence external run path fields
    p = summary.get("stdout_log", "")
    if p:
        return os.path.dirname(p)
    # Sequence mode: use first stage log path.
    for s in summary.get("stage_results", []):
        p = s.get("stdout_log", "")
        if p:
            return os.path.dirname(p)
    return None


def _load_external_efficiency_json(run_dir: str) -> Optional[Dict]:
    path = os.path.join(run_dir, "external_efficiency.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _load_mrfa_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    path = ""
    if run_dir:
        p = os.path.join(run_dir, "mrfa_session_metrics.json")
        if os.path.isfile(p):
            path = p
    if not path:
        workdir = str(summary.get("workdir", "")).strip()
        if workdir and os.path.isdir(workdir):
            logs_root = os.path.join(workdir, "logs")
            newest = ""
            newest_mtime = -1.0
            if os.path.isdir(logs_root):
                for root, _, files in os.walk(logs_root):
                    if "mrfa_session_metrics.json" not in files:
                        continue
                    cand = os.path.join(root, "mrfa_session_metrics.json")
                    try:
                        mt = os.path.getmtime(cand)
                    except OSError:
                        continue
                    if mt > newest_mtime:
                        newest_mtime = mt
                        newest = cand
            if newest:
                path = newest
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            sid = int(s.get("session", 0))
            if sid <= 0:
                continue
            metrics = s.get("metrics", {})
            per_class = metrics.get("per_class", {})
            normalized_per_class = {}
            for k, v in per_class.items():
                try:
                    kk = int(k)
                except Exception:
                    continue
                normalized_per_class[kk] = v
            metrics["per_class"] = normalized_per_class
            if "old_acc" in s:
                metrics["old_acc"] = s.get("old_acc", "NA")
            if "new_acc" in s:
                metrics["new_acc"] = s.get("new_acc", "NA")
            if "total_acc" in s:
                metrics["total_acc"] = s.get("total_acc", "NA")
            class_ids = s.get("seen_class_ids", [])
            class_ids = sorted(set(int(c) for c in class_ids))
            items.append({
                "title": f"session{sid}",
                "metrics": metrics,
                "class_ids": class_ids,
            })
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _load_tagfex_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    path = ""
    if run_dir:
        p = os.path.join(run_dir, "tagfex_session_metrics.json")
        if os.path.isfile(p):
            path = p
    # Do not fall back to workdir walk: stale metrics from prior TagFex runs
    # would mask a failed seed (return_code != 0) and produce misleading test.txt.
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            try:
                sid = int(s.get("session", 0))
                if sid <= 0:
                    continue
                metrics = s.get("metrics", {})
                per_class = metrics.get("per_class", {})
                normalized_per_class = {}
                for k, v in per_class.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    normalized_per_class[kk] = v
                metrics["per_class"] = normalized_per_class
                for key in (
                    "old_micro_acc",
                    "new_micro_acc",
                    "total_micro_acc",
                    "taskid_acc",
                    "macro_acc",
                    "micro_acc",
                    "macro_f1",
                    "micro_f1",
                ):
                    if key in s:
                        metrics[key] = s.get(key, "NA")
                if "old_acc" in s:
                    metrics["old_acc"] = s.get("old_acc", "NA")
                if "new_acc" in s:
                    metrics["new_acc"] = s.get("new_acc", "NA")
                if "total_acc" in s:
                    metrics["total_acc"] = s.get("total_acc", "NA")
                class_ids = s.get("seen_class_ids", [])
                class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                items.append({
                    "title": f"session{sid}",
                    "metrics": metrics,
                    "class_ids": class_ids,
                })
            except Exception:
                continue
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _load_seed_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    if not run_dir:
        return []
    path = os.path.join(run_dir, "seed_session_metrics.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            try:
                sid = int(s.get("session", 0))
                if sid <= 0:
                    continue
                metrics = s.get("metrics", {})
                per_class = metrics.get("per_class", {})
                normalized_per_class = {}
                for k, v in per_class.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    normalized_per_class[kk] = v
                metrics["per_class"] = normalized_per_class
                if "old_acc" in s:
                    metrics["old_acc"] = s.get("old_acc", "NA")
                if "new_acc" in s:
                    metrics["new_acc"] = s.get("new_acc", "NA")
                if "total_acc" in s:
                    metrics["total_acc"] = s.get("total_acc", "NA")
                class_ids = s.get("seen_class_ids", [])
                class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                items.append({
                    "title": f"session{sid}",
                    "metrics": metrics,
                    "class_ids": class_ids,
                })
            except Exception:
                continue
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _build_seed_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_seed_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _load_tpl_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    if not run_dir:
        return []
    path = os.path.join(run_dir, "tpl_session_metrics.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            try:
                sid = int(s.get("session", 0))
                if sid <= 0:
                    continue
                metrics = s.get("metrics", {})
                per_class = metrics.get("per_class", {})
                normalized_per_class = {}
                for k, v in per_class.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    normalized_per_class[kk] = v
                metrics["per_class"] = normalized_per_class
                if "old_acc" in s:
                    metrics["old_acc"] = s.get("old_acc", "NA")
                if "new_acc" in s:
                    metrics["new_acc"] = s.get("new_acc", "NA")
                if "total_acc" in s:
                    metrics["total_acc"] = s.get("total_acc", "NA")
                class_ids = s.get("seen_class_ids", [])
                class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                items.append({
                    "title": f"session{sid}",
                    "metrics": metrics,
                    "class_ids": class_ids,
                })
            except Exception:
                continue
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _build_tpl_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_tpl_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _load_pec_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    if not run_dir:
        return []
    path = os.path.join(run_dir, "pec_session_metrics.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            try:
                sid = int(s.get("session", 0))
                if sid <= 0:
                    continue
                metrics = s.get("metrics", {})
                per_class = metrics.get("per_class", {})
                normalized_per_class = {}
                for k, v in per_class.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    normalized_per_class[kk] = v
                metrics["per_class"] = normalized_per_class
                if "old_acc" in s:
                    metrics["old_acc"] = s.get("old_acc", "NA")
                if "new_acc" in s:
                    metrics["new_acc"] = s.get("new_acc", "NA")
                if "total_acc" in s:
                    metrics["total_acc"] = s.get("total_acc", "NA")
                class_ids = s.get("seen_class_ids", [])
                class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                items.append({
                    "title": f"session{sid}",
                    "metrics": metrics,
                    "class_ids": class_ids,
                })
            except Exception:
                continue
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _build_pec_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_pec_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _load_move_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    if not run_dir:
        return []
    path = os.path.join(run_dir, "move_session_metrics.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            try:
                sid = int(s.get("session", 0))
                if sid <= 0:
                    continue
                metrics = s.get("metrics", {})
                per_class = metrics.get("per_class", {})
                normalized_per_class = {}
                for k, v in per_class.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    normalized_per_class[kk] = v
                metrics["per_class"] = normalized_per_class
                if "old_acc" in s:
                    metrics["old_acc"] = s.get("old_acc", "NA")
                if "new_acc" in s:
                    metrics["new_acc"] = s.get("new_acc", "NA")
                if "total_acc" in s:
                    metrics["total_acc"] = s.get("total_acc", "NA")
                class_ids = s.get("seen_class_ids", [])
                class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                items.append({
                    "title": f"session{sid}",
                    "metrics": metrics,
                    "class_ids": class_ids,
                })
            except Exception:
                continue
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _build_move_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_move_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _load_sema_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    if not run_dir:
        return []
    path = os.path.join(run_dir, "sema_session_metrics.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            try:
                sid = int(s.get("session", 0))
                if sid <= 0:
                    continue
                metrics = s.get("metrics", {})
                per_class = metrics.get("per_class", {})
                normalized_per_class = {}
                for k, v in per_class.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    normalized_per_class[kk] = v
                metrics["per_class"] = normalized_per_class
                if "old_acc" in s:
                    metrics["old_acc"] = s.get("old_acc", "NA")
                if "new_acc" in s:
                    metrics["new_acc"] = s.get("new_acc", "NA")
                if "total_acc" in s:
                    metrics["total_acc"] = s.get("total_acc", "NA")
                class_ids = s.get("seen_class_ids", [])
                class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                items.append({
                    "title": f"session{sid}",
                    "metrics": metrics,
                    "class_ids": class_ids,
                })
            except Exception:
                continue
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _build_sema_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_sema_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _load_moeadapterspp_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    if not run_dir:
        return []
    path = os.path.join(run_dir, "moeadapterspp_session_metrics.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            try:
                sid = int(s.get("session", 0))
                if sid <= 0:
                    continue
                metrics = s.get("metrics", {})
                per_class = metrics.get("per_class", {})
                normalized_per_class = {}
                for k, v in per_class.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    normalized_per_class[kk] = v
                metrics["per_class"] = normalized_per_class
                if "old_acc" in s:
                    metrics["old_acc"] = s.get("old_acc", "NA")
                if "new_acc" in s:
                    metrics["new_acc"] = s.get("new_acc", "NA")
                if "total_acc" in s:
                    metrics["total_acc"] = s.get("total_acc", "NA")
                class_ids = s.get("seen_class_ids", [])
                class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                items.append({
                    "title": f"session{sid}",
                    "metrics": metrics,
                    "class_ids": class_ids,
                })
            except Exception:
                continue
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _build_moeadapterspp_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_moeadapterspp_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _load_more_session_metrics_json(summary: Dict, filename: str = "more_session_metrics.json") -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    if not run_dir:
        return []
    candidates = [filename]
    if filename == "more_session_metrics.json":
        candidates.insert(0, "more_official_session_metrics.json")
    for cand_name in candidates:
        path = os.path.join(run_dir, cand_name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            sessions = obj.get("sessions", [])
            if not isinstance(sessions, list):
                continue
            items = []
            for s in sessions:
                try:
                    sid = int(s.get("session", 0))
                    if sid <= 0:
                        continue
                    metrics = s.get("metrics", {})
                    per_class = metrics.get("per_class", {})
                    normalized_per_class = {}
                    for k, v in per_class.items():
                        try:
                            kk = int(k)
                        except Exception:
                            continue
                        normalized_per_class[kk] = v
                    metrics["per_class"] = normalized_per_class
                    if "old_acc" in s:
                        metrics["old_acc"] = s.get("old_acc", "NA")
                    if "new_acc" in s:
                        metrics["new_acc"] = s.get("new_acc", "NA")
                    if "total_acc" in s:
                        metrics["total_acc"] = s.get("total_acc", "NA")
                    class_ids = s.get("seen_class_ids", [])
                    class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                    items.append({
                        "title": f"session{sid}",
                        "metrics": metrics,
                        "class_ids": class_ids,
                    })
                except Exception:
                    continue
            items.sort(key=lambda x: int(x["title"].replace("session", "")))
            return items
        except Exception:
            continue
    return []


def _build_more_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_more_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _build_build_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_more_session_metrics_json(summary, filename="build_session_metrics.json")
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _load_itaml_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    if not run_dir:
        return []
    path = os.path.join(run_dir, "itaml_session_metrics.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            try:
                sid = int(s.get("session", 0))
                if sid <= 0:
                    continue
                metrics = s.get("metrics", {})
                per_class = metrics.get("per_class", {})
                normalized_per_class = {}
                for k, v in per_class.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    normalized_per_class[kk] = v
                metrics["per_class"] = normalized_per_class
                if "old_acc" in s:
                    metrics["old_acc"] = s.get("old_acc", "NA")
                if "new_acc" in s:
                    metrics["new_acc"] = s.get("new_acc", "NA")
                if "total_acc" in s:
                    metrics["total_acc"] = s.get("total_acc", "NA")
                class_ids = s.get("seen_class_ids", [])
                class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                items.append({
                    "title": f"session{sid}",
                    "metrics": metrics,
                    "class_ids": class_ids,
                })
            except Exception:
                continue
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _build_itaml_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_itaml_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _load_diva_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    if not run_dir:
        return []
    path = os.path.join(run_dir, "diva_session_metrics.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            try:
                sid = int(s.get("session", 0))
                if sid <= 0:
                    continue
                metrics = s.get("metrics", {})
                per_class = metrics.get("per_class", {})
                normalized_per_class = {}
                for k, v in per_class.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    normalized_per_class[kk] = v
                metrics["per_class"] = normalized_per_class
                if "old_acc" in s:
                    metrics["old_acc"] = s.get("old_acc", "NA")
                if "new_acc" in s:
                    metrics["new_acc"] = s.get("new_acc", "NA")
                if "total_acc" in s:
                    metrics["total_acc"] = s.get("total_acc", "NA")
                class_ids = s.get("seen_class_ids", [])
                class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                items.append({
                    "title": f"session{sid}",
                    "metrics": metrics,
                    "class_ids": class_ids,
                })
            except Exception:
                continue
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _build_diva_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_diva_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _load_paper_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    if not run_dir:
        return []
    path = os.path.join(run_dir, "paper_session_metrics.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            try:
                sid = int(s.get("session", 0))
                if sid <= 0:
                    continue
                metrics = s.get("metrics", {})
                per_class = metrics.get("per_class", {})
                normalized_per_class = {}
                for k, v in per_class.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    normalized_per_class[kk] = v
                metrics["per_class"] = normalized_per_class
                if "old_acc" in s:
                    metrics["old_acc"] = s.get("old_acc", "NA")
                if "new_acc" in s:
                    metrics["new_acc"] = s.get("new_acc", "NA")
                if "total_acc" in s:
                    metrics["total_acc"] = s.get("total_acc", "NA")
                class_ids = s.get("seen_class_ids", [])
                class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                items.append({
                    "title": f"session{sid}",
                    "metrics": metrics,
                    "class_ids": class_ids,
                })
            except Exception:
                continue
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _build_paper_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_paper_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _load_genclassifier_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    if not run_dir:
        return []
    path = os.path.join(run_dir, "genclassifier_session_metrics.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            try:
                sid = int(s.get("session", 0))
                if sid <= 0:
                    continue
                metrics = s.get("metrics", {})
                per_class = metrics.get("per_class", {})
                normalized_per_class = {}
                for k, v in per_class.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    normalized_per_class[kk] = v
                metrics["per_class"] = normalized_per_class
                if "old_acc" in s:
                    metrics["old_acc"] = s.get("old_acc", "NA")
                if "new_acc" in s:
                    metrics["new_acc"] = s.get("new_acc", "NA")
                if "total_acc" in s:
                    metrics["total_acc"] = s.get("total_acc", "NA")
                class_ids = s.get("seen_class_ids", [])
                class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                items.append({
                    "title": f"session{sid}",
                    "metrics": metrics,
                    "class_ids": class_ids,
                })
            except Exception:
                continue
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _build_genclassifier_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_genclassifier_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _load_pycil_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    path = ""
    if run_dir:
        p = os.path.join(run_dir, "pycil_session_metrics.json")
        if os.path.isfile(p):
            path = p
    # Do not fall back to PyCIL workdir walk (stale metrics from old logs/).
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            try:
                sid = int(s.get("session", 0))
                if sid <= 0:
                    continue
                metrics = s.get("metrics", {})
                per_class = metrics.get("per_class", {})
                normalized_per_class = {}
                for k, v in per_class.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    normalized_per_class[kk] = v
                metrics["per_class"] = normalized_per_class
                if "old_acc" in s:
                    metrics["old_acc"] = s.get("old_acc", "NA")
                if "new_acc" in s:
                    metrics["new_acc"] = s.get("new_acc", "NA")
                if "total_acc" in s:
                    metrics["total_acc"] = s.get("total_acc", "NA")
                class_ids = s.get("seen_class_ids", [])
                class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                items.append({
                    "title": f"session{sid}",
                    "metrics": metrics,
                    "class_ids": class_ids,
                })
            except Exception:
                continue
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _build_mrfa_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_mrfa_session_metrics_json(summary)
    if session_items:
        return session_items

    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _build_tagfex_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_tagfex_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _build_pycil_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_pycil_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _load_der_session_metrics_json(summary: Dict) -> List[Dict]:
    run_dir = _infer_external_run_dir_from_summary(summary)
    if not run_dir:
        return []
    path = os.path.join(run_dir, "der_paper_session_metrics.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        sessions = obj.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        items = []
        for s in sessions:
            try:
                sid = int(s.get("session", 0))
                if sid <= 0:
                    continue
                metrics = s.get("metrics", {})
                per_class = metrics.get("per_class", {})
                normalized_per_class = {}
                for k, v in per_class.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    normalized_per_class[kk] = v
                metrics["per_class"] = normalized_per_class
                if "old_acc" in s:
                    metrics["old_acc"] = s.get("old_acc", "NA")
                if "new_acc" in s:
                    metrics["new_acc"] = s.get("new_acc", "NA")
                if "total_acc" in s:
                    metrics["total_acc"] = s.get("total_acc", "NA")
                class_ids = s.get("seen_class_ids", [])
                class_ids = sorted(set(int(c) for c in class_ids if c is not None))
                items.append({
                    "title": f"session{sid}",
                    "metrics": metrics,
                    "class_ids": class_ids,
                })
            except Exception:
                continue
        items.sort(key=lambda x: int(x["title"].replace("session", "")))
        return items
    except Exception:
        return []


def _build_der_report_items_from_logs(summary: Dict) -> List[Dict]:
    session_items = _load_der_session_metrics_json(summary)
    if session_items:
        return session_items
    items = []
    seen = []
    task_splits = DATA.get("task_splits", [])
    for i in range(len(task_splits)):
        seen.extend(task_splits[i])
        class_ids = sorted(set(int(c) for c in seen))
        items.append({
            "title": f"session{i + 1}",
            "metrics": _na_metrics_for_classes(class_ids, avg_acc=None),
            "class_ids": class_ids,
        })
    return items


def _data_root_to_str(data_root) -> str:
    if isinstance(data_root, (list, tuple)):
        return ",".join(str(x) for x in data_root)
    return str(data_root)


def _baseline_cil_dataset_name(active_dataset: str) -> str:
    """Map config ACTIVE_DATASET to the name registered in PyCIL-style baseline repos."""
    name = str(active_dataset).lower()
    baseline_aliases = {
        "neu_xsdd": "neu_xsdd",
        "neu_xsdd_magnetic_cr7": "neu_xsdd_magnetic_cr7",
        "cr7_xsdd_neu_magnetic": "neu_xsdd_magnetic_cr7",
        "magnetic_neu_cr7_xsdd": "neu_xsdd_magnetic_cr7",
        "xsdd_magnetic_cr7_neu": "neu_xsdd_magnetic_cr7",
        "cr7_magnetic_xsdd_neu": "neu_xsdd_magnetic_cr7",
        "neu_xddd_magnetic_cr7": "neu_xddd_magnetic_cr7",
        "neu_xdd_magnetic_cr7": "neu_xdd_magnetic_cr7",
        "dagm_gc10": "dagm_gc10",
        "neu_xsdd_magnetic_cr7_dagm_gc10_kolek_bsd": "neu_xsdd_magnetic_cr7_dagm_gc10_kolek_bsd",
        "dagm_neu_gc10_xsdd_kolek_magnetic_bsd_cr7": "neu_xsdd_magnetic_cr7_dagm_gc10_kolek_bsd",
        "xsdd_bsd_magnetic_dagm_cr7_kolek_neu_gc10": "neu_xsdd_magnetic_cr7_dagm_gc10_kolek_bsd",
        "gc10_cr7_neu_kolek_xsdd_dagm_magnetic_bsd": "neu_xsdd_magnetic_cr7_dagm_gc10_kolek_bsd",
        "magnetic_kolek_dagm_neu_bsd_xsdd_gc10_cr7": "neu_xsdd_magnetic_cr7_dagm_gc10_kolek_bsd",
    }
    if name.startswith("neu_xsdd_magnetic_cr7_order"):
        return "neu_xsdd_magnetic_cr7"
    if name.startswith("neu_xsdd_magnetic_cr7_dagm_gc10_kolek_bsd_order"):
        return "neu_xsdd_magnetic_cr7_dagm_gc10_kolek_bsd"
    return baseline_aliases.get(name, str(active_dataset))


def _cil_neu_xsdd_schedule():
    """Flattened class order, per-task sizes, and order-space splits (contiguous 0..K-1)."""
    ts = DATA.get("task_splits") or []
    if not ts:
        return None
    class_order = [int(c) for g in ts for c in g]
    increments = [len(g) for g in ts]
    off = 0
    order_split = []
    for g in ts:
        order_split.append(list(range(off, off + len(g))))
        off += len(g)
    return class_order, increments, order_split


_FAIR_JSON_SKIP_KEYS = frozenset({
    "seed",
    "image_size",
    "class_names",
    "task_splits",
    "init_cls",
    "increment",
    "data_augmentation",
    "optimizer",
    "scheduler",
    "memory_budget",
    "backbone",
    "epochs_per_task",
    "first_stage",
    "second_stage",
    "yaml_overrides",
    "icarl_ewc_lwf_init_epochs",
    "icarl_ewc_lwf_inc_epochs",
    "icarl_ewc_lwf_early_stop_patience",
})

_FAIR_JSON_KEY_MAP = {
    "backbone": "convnet_type",
    "memory_budget": "memory_size",
}


def _deep_update_dict(base: dict, updates: dict) -> None:
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_update_dict(base[key], val)
        else:
            base[key] = deepcopy(val) if isinstance(val, (dict, list)) else val


def _apply_fair_dict_to_json_cfg(cfg_obj: dict, fair: dict, *, skip: Optional[set] = None) -> None:
    """将 train['fair'] 中可映射字段写入 baseline JSON/YAML（config 优先）。"""
    skip_keys = _FAIR_JSON_SKIP_KEYS if skip is None else skip
    for key, val in fair.items():
        if key in skip_keys:
            continue
        mapped = _FAIR_JSON_KEY_MAP.get(key, key)
        if isinstance(val, dict):
            if mapped not in cfg_obj or not isinstance(cfg_obj.get(mapped), dict):
                cfg_obj[mapped] = {}
            _deep_update_dict(cfg_obj[mapped], val)
        else:
            cfg_obj[mapped] = deepcopy(val) if isinstance(val, (list, dict)) else val


def _apply_icarl_schedule_cfg(cfg_obj: dict, fair: dict) -> None:
    """Task0/init + incremental epochs and early stop for PyCIL-style methods."""
    cfg_obj["init_epochs"] = int(
        fair.get("init_epochs", fair.get("icarl_ewc_lwf_init_epochs", 200))
    )
    cfg_obj["epochs"] = int(fair.get("epochs", fair.get("icarl_ewc_lwf_inc_epochs", 150)))
    cfg_obj["early_stop_patience"] = int(
        fair.get("early_stop_patience", fair.get("icarl_ewc_lwf_early_stop_patience", 20))
    )
    cfg_obj["early_stop_min_delta"] = float(fair.get("early_stop_min_delta", 1e-4))


def _sync_neu_xsdd_json_cfg(cfg_obj: dict, fair: dict, *, write_class_order: bool) -> None:
    ts = DATA.get("task_splits") or []
    if not ts:
        return
    incs = [len(g) for g in ts]
    co = [int(c) for g in ts for c in g]
    cfg_obj["increments"] = incs
    cfg_obj["init_cls"] = int(incs[0])
    cfg_obj["increment"] = int(fair.get("increment", incs[0]))
    if write_class_order:
        cfg_obj["class_order"] = co
    cfg_obj["early_stop_min_delta"] = float(fair.get("early_stop_min_delta", 1e-4))
    if "early_stop_patience" in cfg_obj:
        cfg_obj["early_stop_patience"] = int(fair.get("early_stop_patience", 30))


def _pycil_visible_cuda_devices() -> List[int]:
    """Map visible CUDA devices to PyCIL-style device ids (always use local cuda:0 when one GPU is visible)."""
    device_override = os.getenv("CIL_PYCIL_DEVICES", "").strip()
    visible_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device_override:
        parsed = []
        for tok in device_override.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                did = int(tok)
            except ValueError:
                continue
            if did >= 0 and (visible_cuda <= 0 or did < visible_cuda):
                parsed.append(did)
        if parsed:
            return parsed
    return [0] if visible_cuda > 0 else [-1]


def _replace_cmd_after_flag(cmd: List[str], flag: str, new_val: str) -> List[str]:
    out = list(cmd)
    try:
        i = out.index(flag)
    except ValueError:
        return out
    if i + 1 >= len(out):
        return out
    out[i + 1] = new_val
    return out


def _maybe_patch_mrfa_json_cmd(
    method_cfg: Dict,
    launcher: Dict,
    cmd: List[str],
    run_dir: str,
) -> Tuple[List[str], Optional[str]]:
    if str(method_cfg.get("method_key", "")).lower() != "mrfa":
        return cmd, None
    wd = str(launcher.get("workdir", "")).strip()
    if not wd or "--config" not in cmd:
        return cmd, None
    try:
        i = cmd.index("--config")
    except ValueError:
        return cmd, None
    if i + 1 >= len(cmd):
        return cmd, None
    src_rel = cmd[i + 1]
    src_abs = src_rel if os.path.isabs(src_rel) else os.path.join(wd, src_rel)
    cfg_obj: Optional[dict] = None
    if os.path.isfile(src_abs):
        try:
            with open(src_abs, "r", encoding="utf-8") as f:
                cfg_obj = json.load(f)
        except Exception:
            cfg_obj = None
    if not isinstance(cfg_obj, dict):
        cfg_obj = {
            "prefix": "cil",
            "dataset": _baseline_cil_dataset_name(ACTIVE_DATASET),
            "memory_per_class": 0,
            "fixed_memory": False,
            "shuffle": False,
            "save_task_checkpoints": [],
            "load_ckpt": [],
            "perturb_p": [0.0001, 0.0001, 0.0001, 0.0001],
            "num_augmem": 1,
            "disable_perturb": False,
            "auto_kd": False,
            "model_name": "icarl_mrfa",
            "convnet_type": "resnet18",
            "batch_size": 32,
            "num_workers": 4,
            "init_epochs": 200,
            "epochs": 150,
            "early_stop_patience": 20,
            "early_stop_min_delta": 1e-4,
            "lr": 0.1,
            "weight_decay": 0.0005,
            "gamma": 0.1,
            "topk": 1,
            "device": ["0"],
            "seed": [42],
        }
    fair = method_cfg["train"]["fair"]
    cfg_obj["dataset"] = _baseline_cil_dataset_name(ACTIVE_DATASET)
    _sync_neu_xsdd_json_cfg(cfg_obj, fair, write_class_order=True)
    _apply_fair_dict_to_json_cfg(cfg_obj, fair)
    _apply_icarl_schedule_cfg(cfg_obj, fair)
    cfg_obj["memory_size"] = int(
        method_cfg["train"].get("memory_budget", fair.get("memory_budget", cfg_obj.get("memory_size", 0)))
    )
    cfg_obj["convnet_type"] = str(fair.get("backbone", cfg_obj.get("convnet_type", "resnet18")))
    cfg_obj["image_size"] = int(fair.get("image_size", DATA.get("image_size", 224)))
    cfg_obj["flops_backward_factor"] = float(fair.get("flops_backward_factor", 3.0))
    out_path = os.path.join(os.path.abspath(run_dir), "mrfa_neu_xsdd_fair.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg_obj, f, indent=2, ensure_ascii=False)
    return _replace_cmd_after_flag(cmd, "--config", out_path), out_path


def _maybe_patch_tagfex_yaml_cmd(
    method_cfg: Dict,
    launcher: Dict,
    cmd: List[str],
    run_dir: str,
) -> Tuple[List[str], Optional[str]]:
    if str(method_cfg.get("method_key", "")).lower() != "tagfex":
        return cmd, None
    try:
        import yaml  # type: ignore
    except ImportError:
        print("[warn] PyYAML not installed; skip TagFex yaml sync from DATA[task_splits]")
        return cmd, None
    wd = str(launcher.get("workdir", "")).strip()
    if not wd or "--exp-configs" not in cmd:
        return cmd, None
    try:
        i = cmd.index("--exp-configs")
    except ValueError:
        return cmd, None
    if i + 1 >= len(cmd):
        return cmd, None
    src = cmd[i + 1]
    src_abs = src if os.path.isabs(src) else os.path.join(wd, src)
    if not os.path.isfile(src_abs):
        return cmd, None
    with open(src_abs, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return cmd, None
    sched = _cil_neu_xsdd_schedule()
    if sched is None:
        return cmd, None
    co, inc, _ = sched
    cfg["task_num_cls"] = inc
    cfg["class_order"] = co
    fair = method_cfg["train"]["fair"]
    yaml_overrides = fair.get("yaml_overrides", {})
    if yaml_overrides:
        _deep_update_dict(cfg, yaml_overrides)
    _apply_fair_dict_to_json_cfg(cfg, fair)
    cfg["image_size"] = int(fair.get("image_size", DATA.get("image_size", 224)))
    cfg["flops_backward_factor"] = float(fair.get("flops_backward_factor", 3.0))
    bb = str(fair.get("backbone", "resnet18")).lower()
    if isinstance(cfg.get("backbone_configs"), dict):
        cfg["backbone_configs"]["name"] = bb
    out_path = os.path.join(os.path.abspath(run_dir), "tagfex_neu_xsdd_runtime.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return _replace_cmd_after_flag(cmd, "--exp-configs", out_path), out_path


def _build_external_command_and_env(
    method_cfg: Dict,
    run_dir: str,
    launcher: Optional[Dict] = None,
) -> Tuple[List[str], Dict[str, str], str]:
    if launcher is None:
        launcher = method_cfg["train"].get("external_launcher", {})
    entrypoint = str(launcher.get("entrypoint", "")).strip()
    args = launcher.get("args", [])
    if isinstance(args, str):
        args = [args]

    fair = method_cfg["train"]["fair"]
    num_tasks = len(DATA.get("task_splits", [])) or 6
    sched = _cil_neu_xsdd_schedule()
    if sched is not None:
        _co, inc, _order_splits = sched
        num_tasks = len(inc)

    method_key = str(method_cfg.get("method_key", "")).lower()
    seed_fair = {}
    der_paper_fair = {}
    tpl_fair = {}
    pec_fair = {}
    move_fair = {}
    sema_fair = {}
    moeadapterspp_fair = {}
    more_fair = {}
    build_fair = {}
    itaml_fair = {}
    diva_fair = {}
    paper_gr_fair = {}
    genclassifier_fair = {}
    if method_key == "seed":
        seed_fair = fair
    elif method_key == "der_paper":
        der_paper_fair = fair
    elif method_key == "tpl":
        tpl_fair = fair
    elif method_key == "pec":
        pec_fair = fair
    elif method_key == "move":
        move_fair = fair
    elif method_key == "sema":
        sema_fair = fair
    elif method_key == "moeadapterspp":
        moeadapterspp_fair = fair
    elif method_key == "more":
        more_fair = fair
    elif method_key == "build":
        build_fair = fair
    elif method_key == "itaml":
        itaml_fair = fair
    elif method_key == "diva":
        diva_fair = fair
    elif method_key in {"mfgr_paper", "gfril_paper"}:
        paper_gr_fair = fair
    elif method_key == "genclassifier":
        genclassifier_fair = fair
    gc_input_space = str(genclassifier_fair.get("vae_input_space", "feature")).strip().lower()
    if gc_input_space not in {"feature", "raw"}:
        print(f"[warn] unsupported genclassifier vae_input_space={gc_input_space!r}; fallback to 'feature'")
        gc_input_space = "feature"
    gc_entrypoint = "run_cil_neu_xddd_raw.py" if gc_input_space == "raw" else "run_cil_neu_xddd.py"

    context = {
        "run_dir": run_dir,
        "output_dir": method_cfg["train"].get("output_dir", run_dir),
        "data_root": _data_root_to_str(DATA["data_root"]),
        "active_dataset": str(ACTIVE_DATASET),
        "task_splits_json": json.dumps(DATA.get("task_splits", []), ensure_ascii=False),
        "class_names_json": json.dumps(DATA.get("class_names", []), ensure_ascii=False),
        "seed": fair["seed"],
        "init_cls": fair["init_cls"],
        "increment": fair["increment"],
        "memory_budget": method_cfg["train"].get("memory_budget", fair.get("memory_budget", 0)),
        "epochs": fair.get("epochs_per_task", fair.get("num_epochs", fair.get("epochs", 30))),
        "early_stop_patience": str(fair.get("early_stop_patience", 30)),
        "early_stop_min_delta": str(fair.get("early_stop_min_delta", 1e-4)),
        "batch_size": fair["batch_size"],
        "backbone": fair.get("backbone", "resnet18"),
        "num_classes": DATA["num_classes"],
        "repo_url": method_cfg["repo_url"],
        "method_key": method_cfg["method_key"],
        "num_tasks": str(num_tasks),
        "seed_num_workers": str(seed_fair.get("num_workers", fair.get("num_workers", 4))),
        "seed_network": str(
            seed_fair.get("network", seed_fair.get("backbone", fair.get("backbone", "resnet18")))
        ),
        "seed_pretrained": "1" if bool(seed_fair.get("pretrained", fair.get("pretrained", True))) else "0",
        "seed_nepochs": str(seed_fair.get("nepochs", 200)),
        "seed_ftepochs": str(seed_fair.get("ftepochs", 100)),
        "seed_lr": str(seed_fair.get("lr", 0.05)),
        "seed_weight_decay": str(seed_fair.get("weight_decay", 5e-4)),
        "seed_momentum": str(seed_fair.get("momentum", 0.9)),
        "seed_clipping": str(seed_fair.get("clipping", 1.0)),
        "seed_max_experts": str(seed_fair.get("max_experts", 6)),
        "seed_gmms": str(seed_fair.get("gmms", 1)),
        "seed_alpha": str(seed_fair.get("alpha", 0.99)),
        "seed_tau": str(seed_fair.get("tau", 3.0)),
        "seed_shared": str(seed_fair.get("shared", 0)),
        "seed_extra_aug": str(seed_fair.get("extra_aug", "fetril")),
        "seed_early_stop_patience": str(seed_fair.get("early_stop_patience", fair.get("early_stop_patience", 20))),
        "seed_early_stop_min_delta": str(seed_fair.get("early_stop_min_delta", fair.get("early_stop_min_delta", 1e-4))),
        "der_paper_num_workers": str(der_paper_fair.get("num_workers", fair.get("num_workers", 4))),
        "der_paper_image_size": str(der_paper_fair.get("image_size", DATA.get("image_size", 224))),
        "der_paper_backbone": str(der_paper_fair.get("backbone", fair.get("backbone", "resnet18"))),
        "der_paper_pretrained": "1" if bool(der_paper_fair.get("pretrained", False)) else "0",
        "der_paper_epochs": str(der_paper_fair.get("epochs", fair.get("epochs", 80))),
        "der_paper_learning_rate": str(der_paper_fair.get("lr", fair.get("lr", 0.1))),
        "der_paper_momentum": str(der_paper_fair.get("momentum", fair.get("momentum", 0.9))),
        "der_paper_weight_decay": str(der_paper_fair.get("weight_decay", fair.get("weight_decay", 5e-4))),
        "der_paper_early_stop_patience": str(
            der_paper_fair.get("early_stop_patience", fair.get("early_stop_patience", 20))
        ),
        "der_paper_early_stop_min_delta": str(
            der_paper_fair.get("early_stop_min_delta", fair.get("early_stop_min_delta", 1e-4))
        ),
        "der_paper_memory_budget": str(
            der_paper_fair.get("memory_budget", method_cfg["train"].get("memory_budget", fair.get("memory_budget", 600)))
        ),
        "der_paper_replay_percent": str(der_paper_fair.get("replay_percent", fair.get("replay_percent", 0.05))),
        "der_paper_aux_loss_weight": str(der_paper_fair.get("aux_loss_weight", 1.0)),
        "der_paper_mask_reg_weight": str(der_paper_fair.get("mask_reg_weight", 1e-4)),
        "der_paper_mask_threshold": str(der_paper_fair.get("mask_threshold", 0.5)),
        "der_paper_mask_scale": str(der_paper_fair.get("mask_scale", 10.0)),
        "der_paper_finetune_epochs": str(der_paper_fair.get("finetune_epochs", 50)),
        "der_paper_finetune_lr": str(der_paper_fair.get("finetune_lr", 0.05)),
        "der_paper_finetune_weight_decay": str(der_paper_fair.get("finetune_weight_decay", 5e-4)),
        "der_paper_finetune_patience": str(der_paper_fair.get("finetune_patience", 15)),
        "der_paper_grad_clip_norm": str(der_paper_fair.get("grad_clip_norm", 5.0)),
        "der_paper_nf": str(der_paper_fair.get("nf", 64)),
        "tpl_visual_encoder": str(tpl_fair.get("visual_encoder", "deit_small_patch16_224_in661")),
        "tpl_learning_rate": str(tpl_fair.get("lr", 1e-3)),
        "tpl_num_train_epochs": str(tpl_fair.get("epochs", 40)),
        "tpl_early_stop_patience": str(tpl_fair.get("early_stop_patience", 0)),
        "tpl_early_stop_min_delta": str(tpl_fair.get("early_stop_min_delta", 1e-4)),
        "tpl_latent": str(tpl_fair.get("latent", 64)),
        "tpl_k": str(tpl_fair.get("K", 5)),
        "tpl_alpha": str(tpl_fair.get("alpha", 0.2)),
        "tpl_smax": str(tpl_fair.get("smax", 400)),
        "tpl_clipgrad": str(tpl_fair.get("clipgrad", 1.0)),
        "tpl_thres_cosh": str(tpl_fair.get("thres_cosh", 50)),
        "tpl_weight_decay": str(tpl_fair.get("weight_decay", fair.get("weight_decay", 5e-4))),
        "tpl_warmup_ratio": str(tpl_fair.get("warmup_ratio", 0.0)),
        "tpl_lr_scheduler_type": str(tpl_fair.get("lr_scheduler_type", "cosine")),
        "tpl_num_warmup_steps": str(tpl_fair.get("num_warmup_steps", 0)),
        "tpl_gradient_accumulation_steps": str(tpl_fair.get("gradient_accumulation_steps", 1)),
        "tpl_replay_mode": str(tpl_fair.get("replay_mode", "percent")),
        "tpl_replay_percent": str(tpl_fair.get("replay_percent", 0.05)),
        "tpl_replay_buffer_size": str(tpl_fair.get("replay_buffer_size", 600)),
        "tpl_replay_batch_size": str(tpl_fair.get("replay_batch_size", fair.get("batch_size", 32))),
        "tpl_sequence_file": str(tpl_fair.get("sequence_file", "NEUXSDD_6T")),
        "tpl_base_dir": str(tpl_fair.get("base_dir") or os.path.join(run_dir, "tpl_ckpt")),
        "tpl_pretrained_dir": str(tpl_fair.get("pretrained_dir", "")),
        "tpl_baseline_name": str(
            tpl_fair.get("baseline_name") or "tpl_deit_small_patch16_224_in661_neu_xddd_hat"
        ),
        "pec_num_workers": str(pec_fair.get("num_workers", fair.get("num_workers", 4))),
        "pec_image_size": str(pec_fair.get("image_size", DATA.get("image_size", 224))),
        "pec_num_train_epochs": str(pec_fair.get("epochs", 40)),
        "pec_learning_rate": str(pec_fair.get("lr", 1e-3)),
        "pec_optim_kind": str(pec_fair.get("optimizer", "adam")),
        "pec_lr_scheduler": str(pec_fair.get("lr_scheduler", "linear")),
        "pec_early_stop_patience": str(pec_fair.get("early_stop_patience", 30)),
        "pec_early_stop_min_delta": str(pec_fair.get("early_stop_min_delta", 1e-4)),
        "pec_force_no_augmentations": (
            "1" if bool(pec_fair.get("force_no_augmentations", True)) else "0"
        ),
        "pec_architecture": str(pec_fair.get("pec_architecture", "cnn")),
        "pec_num_layers": str(pec_fair.get("pec_num_layers", 2)),
        "pec_width": str(pec_fair.get("pec_width", 40)),
        "pec_teacher_width_multiplier": str(pec_fair.get("pec_teacher_width_multiplier", 4)),
        "pec_output_dim": str(pec_fair.get("pec_output_dim", 172)),
        "pec_activation": str(pec_fair.get("pec_activation", "relu")),
        "pec_normalize_layers": (
            "1" if bool(pec_fair.get("pec_normalize_layers", True)) else "0"
        ),
        "pec_conv_layers": str(pec_fair.get("pec_conv_layers", "(40, 3, 1)")),
        "pec_conv_reduce_spatial_to": str(pec_fair.get("pec_conv_reduce_spatial_to", 4)),
        "pec_train_chunk_size": str(pec_fair.get("pec_train_chunk_size", 4)),
        "move_num_workers": str(move_fair.get("num_workers", fair.get("num_workers", 4))),
        "move_image_size": str(move_fair.get("image_size", DATA.get("image_size", 224))),
        "move_backbone": str(move_fair.get("backbone", fair.get("backbone", "resnet18"))),
        "move_pretrained": "1" if bool(move_fair.get("pretrained", True)) else "0",
        "move_epochs": str(move_fair.get("epochs", 150)),
        "move_learning_rate": str(move_fair.get("lr", 1e-3)),
        "move_weight_decay": str(move_fair.get("weight_decay", 1e-4)),
        "move_early_stop_patience": str(move_fair.get("early_stop_patience", 30)),
        "move_early_stop_min_delta": str(move_fair.get("early_stop_min_delta", 1e-4)),
        "move_num_experts": str(move_fair.get("num_experts", 2)),
        "move_top_k": str(move_fair.get("top_k", 1)),
        "move_kernel_width": str(move_fair.get("kernel_width", 10.0)),
        "move_hidden_dim": str(move_fair.get("hidden_dim", 256)),
        "move_dropout": str(move_fair.get("dropout", 0.1)),
        "move_scale": str(move_fair.get("scale", COMMON_HEAD.get("scale", 20.0))),
        "move_prior_kl_weight": str(move_fair.get("prior_kl_weight", 0.01)),
        "move_gate_prior_kl_weight": str(move_fair.get("gate_prior_kl_weight", 0.01)),
        "move_gate_entropy_weight": str(move_fair.get("gate_entropy_weight", 0.001)),
        "move_expert_diversity_weight": str(move_fair.get("expert_diversity_weight", 0.01)),
        "move_distill_weight": str(move_fair.get("distill_weight", 1.0)),
        "move_distill_temperature": str(move_fair.get("distill_temperature", 2.0)),
        "move_generator_hidden_dim": str(move_fair.get("generator_hidden_dim", 256)),
        "move_generator_z_dim": str(move_fair.get("generator_z_dim", 64)),
        "move_generator_epochs": str(move_fair.get("generator_epochs", 80)),
        "move_generator_early_stop_patience": str(move_fair.get("generator_early_stop_patience", 15)),
        "move_generator_early_stop_min_delta": str(move_fair.get("generator_early_stop_min_delta", 1e-4)),
        "move_generator_beta_kl": str(move_fair.get("generator_beta_kl", 1.0)),
        "move_generator_recon_weight": str(move_fair.get("generator_recon_weight", 1.0)),
        "move_generator_replay_per_class": str(move_fair.get("generator_replay_per_class", 600)),
        "move_generator_replay_temperature": str(move_fair.get("generator_replay_temperature", 1.0)),
        "move_grad_clip_norm": str(move_fair.get("grad_clip_norm", 5.0)),
        "sema_num_workers": str(sema_fair.get("num_workers", fair.get("num_workers", 4))),
        "sema_image_size": str(sema_fair.get("image_size", DATA.get("image_size", 224))),
        "sema_backbone": str(sema_fair.get("backbone", fair.get("backbone", "vit_b_16"))),
        "sema_pretrained": "1" if bool(sema_fair.get("pretrained", True)) else "0",
        "sema_epochs": str(sema_fair.get("epochs", 50)),
        "sema_adapter_lr": str(sema_fair.get("adapter_lr", 0.005)),
        "sema_rd_lr": str(sema_fair.get("rd_lr", 0.01)),
        "sema_weight_decay": str(sema_fair.get("weight_decay", 0.0)),
        "sema_early_stop_patience": str(sema_fair.get("early_stop_patience", 20)),
        "sema_early_stop_min_delta": str(sema_fair.get("early_stop_min_delta", 1e-4)),
        "sema_adapter_hidden_dim": str(sema_fair.get("adapter_hidden_dim", 16)),
        "sema_rd_hidden_dim": str(sema_fair.get("rd_hidden_dim", 128)),
        "sema_rd_bottleneck_dim": str(sema_fair.get("rd_bottleneck_dim", 64)),
        "sema_expansion_layers": ",".join(
            str(x) for x in sema_fair.get("expansion_layers", [9, 10, 11])
        ),
        "sema_expansion_threshold": str(sema_fair.get("expansion_threshold", 1.0)),
        "sema_expansion_min_fraction": str(sema_fair.get("expansion_min_fraction", 0.05)),
        "sema_rd_loss_weight": str(sema_fair.get("rd_loss_weight", 1.0)),
        "sema_classifier_scale": str(sema_fair.get("classifier_scale", 20.0)),
        "sema_grad_clip_norm": str(sema_fair.get("grad_clip_norm", 5.0)),
        "moeadapterspp_num_workers": str(moeadapterspp_fair.get("num_workers", fair.get("num_workers", 4))),
        "moeadapterspp_image_size": str(moeadapterspp_fair.get("image_size", DATA.get("image_size", 224))),
        "moeadapterspp_backbone": str(moeadapterspp_fair.get("backbone", fair.get("backbone", "clip_vit_b_16"))),
        "moeadapterspp_pretrained": "1" if bool(moeadapterspp_fair.get("pretrained", True)) else "0",
        "moeadapterspp_epochs": str(moeadapterspp_fair.get("epochs", 25)),
        "moeadapterspp_learning_rate": str(moeadapterspp_fair.get("lr", 1e-3)),
        "moeadapterspp_weight_decay": str(moeadapterspp_fair.get("weight_decay", 0.0)),
        "moeadapterspp_early_stop_patience": str(moeadapterspp_fair.get("early_stop_patience", 10)),
        "moeadapterspp_early_stop_min_delta": str(moeadapterspp_fair.get("early_stop_min_delta", 1e-4)),
        "moeadapterspp_router_hidden_dim": str(moeadapterspp_fair.get("router_hidden_dim", 256)),
        "moeadapterspp_expert_hidden_dim": str(moeadapterspp_fair.get("expert_hidden_dim", 16)),
        "moeadapterspp_ae_hidden_dim": str(moeadapterspp_fair.get("ae_hidden_dim", 256)),
        "moeadapterspp_ae_bottleneck_dim": str(moeadapterspp_fair.get("ae_bottleneck_dim", 64)),
        "moeadapterspp_recognition_layer": str(moeadapterspp_fair.get("recognition_layer", 6)),
        "moeadapterspp_subsequent_layers": ",".join(
            str(x) for x in moeadapterspp_fair.get("subsequent_layers", [7, 8, 9, 10, 11])
        ),
        "moeadapterspp_initial_experts": str(moeadapterspp_fair.get("initial_experts", 2)),
        "moeadapterspp_top_k": str(moeadapterspp_fair.get("top_k", 2)),
        "moeadapterspp_expansion_threshold": str(moeadapterspp_fair.get("expansion_threshold", 1.0)),
        "moeadapterspp_expansion_min_fraction": str(moeadapterspp_fair.get("expansion_min_fraction", 0.05)),
        "moeadapterspp_leas_loss_weight": str(moeadapterspp_fair.get("leas_loss_weight", 1.0)),
        "moeadapterspp_deec_loss_weight": str(moeadapterspp_fair.get("deec_loss_weight", 1.0)),
        "moeadapterspp_classifier_scale": str(moeadapterspp_fair.get("classifier_scale", 20.0)),
        "moeadapterspp_grad_clip_norm": str(moeadapterspp_fair.get("grad_clip_norm", 5.0)),
        "moeadapterspp_label_smoothing": str(moeadapterspp_fair.get("label_smoothing", 0.0)),
        "moeadapterspp_paper_mode": "1" if bool(moeadapterspp_fair.get("paper_mode", True)) else "0",
        "moeadapterspp_paper_warmup_epochs": str(moeadapterspp_fair.get("paper_warmup_epochs", 1)),
        "moeadapterspp_paper_preference_window": str(moeadapterspp_fair.get("paper_preference_window", 10)),
        "more_num_workers": str(more_fair.get("num_workers", fair.get("num_workers", 4))),
        "more_image_size": str(more_fair.get("image_size", DATA.get("image_size", 224))),
        "more_backbone": str(more_fair.get("backbone", fair.get("backbone", "resnet18"))),
        "more_pretrained": "1" if bool(more_fair.get("pretrained", True)) else "0",
        "more_epochs": str(more_fair.get("epochs", 40)),
        "more_learning_rate": str(more_fair.get("lr", 1e-3)),
        "more_weight_decay": str(more_fair.get("weight_decay", 5e-4)),
        "more_momentum": str(more_fair.get("momentum", 0.9)),
        "more_early_stop_patience": str(more_fair.get("early_stop_patience", 20)),
        "more_early_stop_min_delta": str(more_fair.get("early_stop_min_delta", 1e-4)),
        "more_hidden_dim": str(more_fair.get("hidden_dim", 256)),
        "more_dropout": str(more_fair.get("dropout", 0.1)),
        "more_replay_percent": str(more_fair.get("replay_percent", 0.05)),
        "more_back_update": "1" if bool(more_fair.get("back_update", True)) else "0",
        "more_back_update_epochs": str(more_fair.get("back_update_epochs", 10)),
        "more_back_update_lr": str(more_fair.get("back_update_lr", 0.01)),
        "more_back_update_batch_size": str(more_fair.get("back_update_batch_size", 16)),
        "more_distance_scale": str(more_fair.get("distance_scale", 20.0)),
        "more_use_distance_coeff": "1" if bool(more_fair.get("use_distance_coeff", True)) else "0",
        "more_grad_clip_norm": str(more_fair.get("grad_clip_norm", 5.0)),
        "more_adapter_dim": str(more_fair.get("adapter_dim", 64)),
        "more_smax": str(more_fair.get("smax", 500)),
        "more_reg_lambda": str(more_fair.get("reg_lambda", 0.75)),
        "build_num_workers": str(build_fair.get("num_workers", fair.get("num_workers", 4))),
        "build_image_size": str(build_fair.get("image_size", DATA.get("image_size", 224))),
        "build_backbone": str(build_fair.get("backbone", fair.get("backbone", "deit_small_patch16_224_in661"))),
        "build_pretrained": "1" if bool(build_fair.get("pretrained", True)) else "0",
        "build_epochs": str(build_fair.get("epochs", 40)),
        "build_learning_rate": str(build_fair.get("lr", 0.005)),
        "build_weight_decay": str(build_fair.get("weight_decay", 5e-4)),
        "build_momentum": str(build_fair.get("momentum", 0.9)),
        "build_early_stop_patience": str(build_fair.get("early_stop_patience", 15)),
        "build_early_stop_min_delta": str(build_fair.get("early_stop_min_delta", 1e-4)),
        "build_adapter_dim": str(build_fair.get("adapter_dim", 64)),
        "build_dropout": str(build_fair.get("dropout", 0.1)),
        "build_smax": str(build_fair.get("smax", 500)),
        "build_mask_reg_weight": str(build_fair.get("mask_reg_weight", 1e-4)),
        "build_detector": str(build_fair.get("detector", "base")),
        "build_scorer": str(build_fair.get("scorer", "smmd")),
        "build_react_percentile": str(build_fair.get("react_percentile", 90)),
        "build_dice_percentile": str(build_fair.get("dice_percentile", 85)),
        "build_scale_percentile": str(build_fair.get("scale_percentile", 85)),
        "build_md_scale": str(build_fair.get("md_scale", 20.0)),
        "build_md_ridge": str(build_fair.get("md_ridge", 1e-3)),
        "build_grad_clip_norm": str(build_fair.get("grad_clip_norm", 5.0)),
        "itaml_num_workers": str(itaml_fair.get("num_workers", fair.get("num_workers", 4))),
        "itaml_image_size": str(itaml_fair.get("image_size", DATA.get("image_size", 224))),
        "itaml_backbone": str(itaml_fair.get("backbone", fair.get("backbone", "resnet18"))),
        "itaml_pretrained": "1" if bool(itaml_fair.get("pretrained", True)) else "0",
        "itaml_epochs": str(itaml_fair.get("epochs", 70)),
        "itaml_learning_rate": str(itaml_fair.get("lr", 0.01)),
        "itaml_optimizer": str(itaml_fair.get("optimizer", "radam")),
        "itaml_weight_decay": str(itaml_fair.get("weight_decay", 5e-4)),
        "itaml_momentum": str(itaml_fair.get("momentum", 0.9)),
        "itaml_early_stop_patience": str(itaml_fair.get("early_stop_patience", 20)),
        "itaml_early_stop_min_delta": str(itaml_fair.get("early_stop_min_delta", 1e-4)),
        "itaml_hidden_dim": str(itaml_fair.get("hidden_dim", 512)),
        "itaml_embed_dim": str(itaml_fair.get("embed_dim", 256)),
        "itaml_dropout": str(itaml_fair.get("dropout", 0.1)),
        "itaml_replay_percent": str(itaml_fair.get("replay_percent", 0.05)),
        "itaml_inner_steps": str(itaml_fair.get("inner_steps", 1)),
        "itaml_beta": str(itaml_fair.get("beta", 1.0)),
        "itaml_continuum_size": str(itaml_fair.get("continuum_size", 1)),
        "itaml_grad_clip_norm": str(itaml_fair.get("grad_clip_norm", 5.0)),
        "diva_num_workers": str(diva_fair.get("num_workers", fair.get("num_workers", 4))),
        "diva_image_size": str(diva_fair.get("image_size", DATA.get("image_size", 224))),
        "diva_backbone": str(diva_fair.get("backbone", fair.get("backbone", "resnet18"))),
        "diva_pretrained": "1" if bool(diva_fair.get("pretrained", True)) else "0",
        "diva_epochs": str(diva_fair.get("epochs", 100)),
        "diva_learning_rate": str(diva_fair.get("lr", 1e-3)),
        "diva_weight_decay": str(diva_fair.get("weight_decay", 0.0)),
        "diva_early_stop_patience": str(diva_fair.get("early_stop_patience", 20)),
        "diva_early_stop_min_delta": str(diva_fair.get("early_stop_min_delta", 1e-4)),
        "diva_hidden_dim": str(diva_fair.get("hidden_dim", 512)),
        "diva_z_dim": str(diva_fair.get("z_dim", 128)),
        "diva_dropout": str(diva_fair.get("dropout", 0.1)),
        "diva_generated_replay_per_class": str(diva_fair.get("generated_replay_per_class", 600)),
        "diva_replay_temperature": str(diva_fair.get("replay_temperature", 1.0)),
        "diva_lambda_cls": str(diva_fair.get("lambda_cls", 10.0)),
        "diva_beta_kl": str(diva_fair.get("beta_kl", 1.0)),
        "diva_recon_weight": str(diva_fair.get("recon_weight", 1.0)),
        "diva_input_noise_std": str(diva_fair.get("input_noise_std", 0.0)),
        "diva_wide_widen_factor": str(diva_fair.get("wide_widen_factor", 4)),
        "diva_domain_translation": "1" if bool(diva_fair.get("domain_translation", False)) else "0",
        "diva_dt_epochs": str(diva_fair.get("dt_epochs", 10)),
        "diva_dt_lr": str(diva_fair.get("dt_lr", 2e-4)),
        "diva_dt_channels": str(diva_fair.get("dt_channels", 32)),
        "diva_dt_cycle_weight": str(diva_fair.get("dt_cycle_weight", 10.0)),
        "diva_dt_identity_weight": str(diva_fair.get("dt_identity_weight", 0.5)),
        "diva_grad_clip_norm": str(diva_fair.get("grad_clip_norm", 5.0)),
        "paper_source_dir": str(paper_gr_fair.get("paper_source_dir", method_key)),
        "paper_num_workers": str(paper_gr_fair.get("num_workers", fair.get("num_workers", 4))),
        "paper_image_size": str(paper_gr_fair.get("image_size", DATA.get("image_size", 224))),
        "paper_backbone": str(paper_gr_fair.get("backbone", fair.get("backbone", "resnet18"))),
        "paper_pretrained": "1" if bool(paper_gr_fair.get("pretrained", True)) else "0",
        "paper_epochs": str(paper_gr_fair.get("epochs", 80)),
        "paper_learning_rate": str(paper_gr_fair.get("lr", 1e-3)),
        "paper_weight_decay": str(paper_gr_fair.get("weight_decay", 1e-4)),
        "paper_early_stop_patience": str(paper_gr_fair.get("early_stop_patience", 15)),
        "paper_early_stop_min_delta": str(paper_gr_fair.get("early_stop_min_delta", 1e-4)),
        "paper_hidden_dim": str(paper_gr_fair.get("hidden_dim", 512)),
        "paper_z_dim": str(paper_gr_fair.get("z_dim", 128)),
        "paper_dropout": str(paper_gr_fair.get("dropout", 0.1)),
        "paper_generated_replay_budget": str(paper_gr_fair.get("generated_replay_budget", 600)),
        "paper_replay_percent": str(paper_gr_fair.get("replay_percent", 0.05)),
        "paper_replay_temperature": str(paper_gr_fair.get("replay_temperature", 1.0)),
        "paper_lambda_cls": str(paper_gr_fair.get("lambda_cls", 10.0)),
        "paper_beta_kl": str(paper_gr_fair.get("beta_kl", 1.0)),
        "paper_recon_weight": str(paper_gr_fair.get("recon_weight", 1.0)),
        "paper_mfgr_aligned": "1" if bool(paper_gr_fair.get("mfgr_aligned", False)) else "0",
        "paper_mfgr_classifier_backbone": str(paper_gr_fair.get("mfgr_classifier_backbone", "resnet18")),
        "paper_mfgr_latent_dim": str(paper_gr_fair.get("mfgr_latent_dim", 1000)),
        "paper_mfgr_generator_base_channels": str(paper_gr_fair.get("mfgr_generator_base_channels", 32)),
        "paper_mfgr_generator_epochs": str(paper_gr_fair.get("mfgr_generator_epochs", 80)),
        "paper_mfgr_generator_steps_per_epoch": str(paper_gr_fair.get("mfgr_generator_steps_per_epoch", 20)),
        "paper_mfgr_generator_batch_size": str(paper_gr_fair.get("mfgr_generator_batch_size", 64)),
        "paper_mfgr_generated_batch_size": str(paper_gr_fair.get("mfgr_generated_batch_size", 64)),
        "paper_mfgr_generator_lr": str(paper_gr_fair.get("mfgr_generator_lr", 1e-3)),
        "paper_mfgr_temperature": str(paper_gr_fair.get("mfgr_temperature", 2.0)),
        "paper_mfgr_momentum": str(paper_gr_fair.get("mfgr_momentum", 0.9)),
        "paper_mfgr_goh_ratio": str(paper_gr_fair.get("mfgr_goh_ratio", 1.0)),
        "paper_mfgr_gie_ratio": str(paper_gr_fair.get("mfgr_gie_ratio", 5.0)),
        "paper_mfgr_ga_ratio": str(paper_gr_fair.get("mfgr_ga_ratio", 0.1)),
        "paper_mfgr_gtv_ratio": str(paper_gr_fair.get("mfgr_gtv_ratio", 0.0)),
        "paper_mfgr_gbn_ratio": str(paper_gr_fair.get("mfgr_gbn_ratio", 1.0)),
        "paper_mfgr_gkl_ratio": str(paper_gr_fair.get("mfgr_gkl_ratio", 0.1)),
        "paper_mfgr_kl_img_sample_num": str(paper_gr_fair.get("mfgr_kl_img_sample_num", 200)),
        "paper_mfgr_o_ce": str(paper_gr_fair.get("mfgr_o_ce", 1.0)),
        "paper_mfgr_n_ce": str(paper_gr_fair.get("mfgr_n_ce", 1.0)),
        "paper_mfgr_o_kd": str(paper_gr_fair.get("mfgr_o_kd", 1.0)),
        "paper_mfgr_n_kd": str(paper_gr_fair.get("mfgr_n_kd", 1.0)),
        "paper_gfril_aligned": "1" if bool(paper_gr_fair.get("gfril_aligned", False)) else "0",
        "paper_gfril_classifier_backbone": str(paper_gr_fair.get("gfril_classifier_backbone", "resnet18")),
        "paper_gfril_latent_dim": str(paper_gr_fair.get("gfril_latent_dim", 200)),
        "paper_gfril_hidden_dim": str(paper_gr_fair.get("gfril_hidden_dim", 2048)),
        "paper_gfril_generator_epochs": str(paper_gr_fair.get("gfril_generator_epochs", 501)),
        "paper_gfril_generator_steps_per_epoch": str(paper_gr_fair.get("gfril_generator_steps_per_epoch", 0)),
        "paper_gfril_gan_lr": str(paper_gr_fair.get("gfril_gan_lr", 1e-4)),
        "paper_gfril_lambda_gp": str(paper_gr_fair.get("gfril_lambda_gp", 10.0)),
        "paper_gfril_n_critic": str(paper_gr_fair.get("gfril_n_critic", 5)),
        "paper_gfril_feature_distill_weight": str(paper_gr_fair.get("gfril_feature_distill_weight", 1.0)),
        "paper_gfril_replay_cls_weight": str(paper_gr_fair.get("gfril_replay_cls_weight", 1.0)),
        "paper_gfril_alignment_weight": str(paper_gr_fair.get("gfril_alignment_weight", 1.0)),
        "paper_gfril_replay_batch_size": str(paper_gr_fair.get("gfril_replay_batch_size", 0)),
        "paper_input_noise_std": str(paper_gr_fair.get("input_noise_std", 0.0)),
        "paper_grad_clip_norm": str(paper_gr_fair.get("grad_clip_norm", 5.0)),
        "gc_num_workers": str(genclassifier_fair.get("num_workers", fair.get("num_workers", 4))),
        "gc_image_size": str(genclassifier_fair.get("image_size", DATA.get("image_size", 224))),
        "gc_entrypoint": gc_entrypoint,
        "gc_vae_epochs": str(genclassifier_fair.get("vae_epochs", 120)),
        "gc_vae_lr": str(genclassifier_fair.get("vae_lr", 1e-3)),
        "gc_z_dim": str(genclassifier_fair.get("z_dim", 128)),
        "gc_h_dim": str(genclassifier_fair.get("h_dim", 512)),
        "gc_eval_s": str(genclassifier_fair.get("eval_importance_samples", 200)),
        "gc_backbone": str(genclassifier_fair.get("backbone", fair.get("backbone", "resnet18"))),
        "gc_early_stop_patience": str(genclassifier_fair.get("early_stop_patience", 30)),
        "gc_early_stop_min_delta": str(genclassifier_fair.get("early_stop_min_delta", 1e-4)),
    }

    command_parts = []
    if entrypoint:
        command_parts.extend(shlex.split(entrypoint.format(**context)))
    for arg in args:
        command_parts.append(str(arg).format(**context))

    env = os.environ.copy()
    env["CIL_DATA_ROOT"] = context["data_root"]
    env["CIL_DATA_ROOTS"] = context["data_root"]
    env["CIL_SEED"] = str(context["seed"])
    env["CIL_INIT_CLS"] = str(context["init_cls"])
    env["CIL_INCREMENT"] = str(context["increment"])
    env["CIL_MEMORY_BUDGET"] = str(context["memory_budget"])
    env["CIL_EPOCHS_PER_TASK"] = str(context["epochs"])
    env["CIL_BATCH_SIZE"] = str(context["batch_size"])
    env["CIL_NUM_TASKS"] = str(num_tasks)
    env["CIL_BACKBONE"] = str(context["backbone"])
    env["CIL_OUTPUT_DIR"] = str(context["output_dir"])
    env["CIL_METHOD_KEY"] = str(context["method_key"])
    env["CIL_RUN_DIR"] = os.path.abspath(run_dir)
    env["CIL_PROJECT_DIR"] = os.path.dirname(os.path.abspath(__file__))
    env["CIL_FLOPS_BACKWARD_FACTOR"] = str(
        method_cfg.get("train", {}).get("fair", {}).get("flops_backward_factor", 3.0)
    )
    env["CIL_PROFILE_FLOPS"] = "1" if bool(EFFICIENCY.get("profile_flops", True)) else "0"
    env["CIL_STRICT_REPRO"] = str(EFFICIENCY.get("strict_reproducibility", "all"))
    env["CIL_CLASS_NAMES_JSON"] = str(context["class_names_json"])
    env["CIL_DATASET_CLASS_NAMES_JSON"] = json.dumps(
        DATA.get("dataset_class_names", []),
        ensure_ascii=False,
    )
    env["CIL_GEN_PREVIEW_PER_CLASS"] = "10"
    if method_key == "seed":
        sf = method_cfg["train"]["fair"]
        env["CIL_SEED_NETWORK"] = str(sf.get("backbone", "resnet18"))
        env["CIL_SEED_NEPOCHS"] = str(sf.get("nepochs", 200))
        env["CIL_SEED_LR"] = str(sf.get("lr", 0.05))
        env["CIL_SEED_WEIGHT_DECAY"] = str(sf.get("weight_decay", 5e-4))
        env["CIL_SEED_MOMENTUM"] = str(sf.get("momentum", 0.9))
        env["CIL_SEED_CLIPPING"] = str(sf.get("clipping", 1.0))
        env["CIL_SEED_MAX_EXPERTS"] = str(sf.get("max_experts", 6))
        env["CIL_SEED_GMMS"] = str(sf.get("gmms", 1))
        env["CIL_SEED_ALPHA"] = str(sf.get("alpha", 0.99))
        env["CIL_SEED_TAU"] = str(sf.get("tau", 3.0))
        env["CIL_SEED_SHARED"] = str(sf.get("shared", 0))
        env["CIL_SEED_EXTRA_AUG"] = str(sf.get("extra_aug", "fetril"))
        env["CIL_SEED_PRETRAINED"] = "1" if bool(sf.get("pretrained", True)) else "0"
        env["CIL_SEED_FTEPOCHS"] = str(sf.get("ftepochs", 100))
        env["CIL_SEED_EARLY_STOP_PATIENCE"] = str(sf.get("early_stop_patience", 30))
        env["CIL_SEED_EARLY_STOP_MIN_DELTA"] = str(sf.get("early_stop_min_delta", 1e-4))
        env["CIL_IMAGE_SIZE"] = str(sf.get("image_size", DATA.get("image_size", 224)))
        env["CIL_NUM_WORKERS"] = str(sf.get("num_workers", 4))
        env["CIL_NUM_TASKS"] = str(num_tasks)
    if method_key == "tpl":
        tf = method_cfg["train"]["fair"]
        env["CIL_TPL_VISUAL_ENCODER"] = str(tf.get("visual_encoder", "deit_small_patch16_224_in661"))
        env["CIL_TPL_LR"] = str(tf.get("lr", 1e-3))
        env["CIL_TPL_EPOCHS"] = str(tf.get("epochs", 40))
        env["CIL_TPL_LATENT"] = str(tf.get("latent", 64))
        env["CIL_TPL_REPLAY_MODE"] = str(tf.get("replay_mode", "percent"))
        env["CIL_TPL_REPLAY_PERCENT"] = str(tf.get("replay_percent", 0.05))
        env["CIL_TPL_REPLAY_BUFFER_SIZE"] = str(tf.get("replay_buffer_size", 600))
        env["CIL_TPL_REPLAY_BATCH_SIZE"] = str(tf.get("replay_batch_size", tf.get("batch_size", 32)))
        env["CIL_TPL_SEQUENCE_FILE"] = str(tf.get("sequence_file", "NEUXSDD_6T"))
        env["CIL_TPL_BASE_DIR"] = str(tf.get("base_dir") or os.path.join(run_dir, "tpl_ckpt"))
        env["CIL_TPL_PRETRAINED_DIR"] = str(tf.get("pretrained_dir", ""))
        env["CIL_TPL_BASELINE_NAME"] = str(
            tf.get("baseline_name") or "tpl_deit_small_patch16_224_in661_neu_xddd_hat"
        )
        env["CIL_TPL_K"] = str(tf.get("K", 5))
        env["CIL_TPL_ALPHA"] = str(tf.get("alpha", 0.2))
        env["CIL_TPL_SMAX"] = str(tf.get("smax", 400))
        env["CIL_TPL_CLIPGRAD"] = str(tf.get("clipgrad", 1.0))
        env["CIL_TPL_THRES_COSH"] = str(tf.get("thres_cosh", 50))
        env["CIL_TPL_WEIGHT_DECAY"] = str(tf.get("weight_decay", 5e-4))
        env["CIL_TPL_WARMUP_RATIO"] = str(tf.get("warmup_ratio", 0.0))
        env["CIL_TPL_LR_SCHEDULER_TYPE"] = str(tf.get("lr_scheduler_type", "cosine"))
        env["CIL_TPL_NUM_WARMUP_STEPS"] = str(tf.get("num_warmup_steps", 0))
        env["CIL_TPL_GRADIENT_ACCUMULATION_STEPS"] = str(tf.get("gradient_accumulation_steps", 1))
        env["CIL_TPL_PRETRAINED"] = "1" if bool(tf.get("pretrained", True)) else "0"
        env["CIL_IMAGE_SIZE"] = str(tf.get("image_size", DATA.get("image_size", 224)))
        env["CIL_NUM_WORKERS"] = str(tf.get("num_workers", 4))
        env["CIL_NUM_TASKS"] = str(num_tasks)
    if method_key == "pec":
        pf = method_cfg["train"]["fair"]
        env["CIL_IMAGE_SIZE"] = str(pf.get("image_size", DATA.get("image_size", 224)))
        env["CIL_NUM_WORKERS"] = str(pf.get("num_workers", 4))
        env["CIL_NUM_TASKS"] = str(num_tasks)
        env["CIL_PEC_EPOCHS"] = str(pf.get("epochs", 40))
        env["CIL_PEC_LR"] = str(pf.get("lr", 1e-3))
        env["CIL_PEC_OPTIM_KIND"] = str(pf.get("optimizer", "adam"))
        env["CIL_PEC_OPTIM_SCHEDULER"] = str(pf.get("lr_scheduler", "linear"))
        env["CIL_PEC_FORCE_NO_AUG"] = "1" if bool(pf.get("force_no_augmentations", True)) else "0"
        env["CIL_PEC_EARLY_STOP_PATIENCE"] = str(pf.get("early_stop_patience", 30))
        env["CIL_PEC_EARLY_STOP_MIN_DELTA"] = str(pf.get("early_stop_min_delta", 1e-4))
        env["CIL_PEC_ARCHITECTURE"] = str(pf.get("pec_architecture", "cnn"))
        env["CIL_PEC_NUM_LAYERS"] = str(pf.get("pec_num_layers", 2))
        env["CIL_PEC_WIDTH"] = str(pf.get("pec_width", 40))
        env["CIL_PEC_TEACHER_WIDTH_MULTIPLIER"] = str(pf.get("pec_teacher_width_multiplier", 4))
        env["CIL_PEC_OUTPUT_DIM"] = str(pf.get("pec_output_dim", 172))
        env["CIL_PEC_ACTIVATION"] = str(pf.get("pec_activation", "relu"))
        env["CIL_PEC_NORMALIZE_LAYERS"] = "1" if bool(pf.get("pec_normalize_layers", True)) else "0"
        env["CIL_PEC_CONV_LAYERS"] = str(pf.get("pec_conv_layers", "(40, 3, 1)"))
        env["CIL_PEC_CONV_REDUCE_SPATIAL_TO"] = str(pf.get("pec_conv_reduce_spatial_to", 4))
        env["CIL_PEC_TRAIN_CHUNK_SIZE"] = str(pf.get("pec_train_chunk_size", 4))
    if method_key == "move":
        mf = method_cfg["train"]["fair"]
        env["CIL_IMAGE_SIZE"] = str(mf.get("image_size", DATA.get("image_size", 224)))
        env["CIL_NUM_WORKERS"] = str(mf.get("num_workers", 4))
        env["CIL_NUM_TASKS"] = str(num_tasks)
        env["CIL_MOVE_EPOCHS"] = str(mf.get("epochs", 150))
        env["CIL_MOVE_LR"] = str(mf.get("lr", 1e-3))
        env["CIL_MOVE_WEIGHT_DECAY"] = str(mf.get("weight_decay", 1e-4))
        env["CIL_MOVE_EARLY_STOP_PATIENCE"] = str(mf.get("early_stop_patience", 30))
        env["CIL_MOVE_EARLY_STOP_MIN_DELTA"] = str(mf.get("early_stop_min_delta", 1e-4))
        env["CIL_MOVE_NUM_EXPERTS"] = str(mf.get("num_experts", 2))
        env["CIL_MOVE_TOP_K"] = str(mf.get("top_k", 1))
        env["CIL_MOVE_HIDDEN_DIM"] = str(mf.get("hidden_dim", 256))
        env["CIL_MOVE_DROPOUT"] = str(mf.get("dropout", 0.1))
        env["CIL_MOVE_SCALE"] = str(mf.get("scale", COMMON_HEAD.get("scale", 20.0)))
        env["CIL_MOVE_PRIOR_KL_WEIGHT"] = str(mf.get("prior_kl_weight", 0.01))
        env["CIL_MOVE_GATE_PRIOR_KL_WEIGHT"] = str(mf.get("gate_prior_kl_weight", 0.01))
        env["CIL_MOVE_GATE_ENTROPY_WEIGHT"] = str(mf.get("gate_entropy_weight", 0.001))
        env["CIL_MOVE_EXPERT_DIVERSITY_WEIGHT"] = str(mf.get("expert_diversity_weight", 0.01))
        env["CIL_MOVE_GENERATOR_HIDDEN_DIM"] = str(mf.get("generator_hidden_dim", 256))
        env["CIL_MOVE_GENERATOR_Z_DIM"] = str(mf.get("generator_z_dim", 64))
        env["CIL_MOVE_GENERATOR_EPOCHS"] = str(mf.get("generator_epochs", 80))
        env["CIL_MOVE_GENERATOR_EARLY_STOP_PATIENCE"] = str(mf.get("generator_early_stop_patience", 15))
        env["CIL_MOVE_GENERATOR_EARLY_STOP_MIN_DELTA"] = str(mf.get("generator_early_stop_min_delta", 1e-4))
        env["CIL_MOVE_GENERATOR_BETA_KL"] = str(mf.get("generator_beta_kl", 1.0))
        env["CIL_MOVE_GENERATOR_RECON_WEIGHT"] = str(mf.get("generator_recon_weight", 1.0))
        env["CIL_MOVE_GENERATOR_REPLAY_PER_CLASS"] = str(mf.get("generator_replay_per_class", 600))
        env["CIL_MOVE_GENERATOR_REPLAY_TEMPERATURE"] = str(mf.get("generator_replay_temperature", 1.0))
        env["CIL_MOVE_GRAD_CLIP_NORM"] = str(mf.get("grad_clip_norm", 5.0))
    if method_key == "sema":
        sf = method_cfg["train"]["fair"]
        env["CIL_IMAGE_SIZE"] = str(sf.get("image_size", DATA.get("image_size", 224)))
        env["CIL_NUM_WORKERS"] = str(sf.get("num_workers", 4))
        env["CIL_NUM_TASKS"] = str(num_tasks)
        env["CIL_SEMA_EPOCHS"] = str(sf.get("epochs", 50))
        env["CIL_SEMA_ADAPTER_LR"] = str(sf.get("adapter_lr", 0.005))
        env["CIL_SEMA_RD_LR"] = str(sf.get("rd_lr", 0.01))
        env["CIL_SEMA_WEIGHT_DECAY"] = str(sf.get("weight_decay", 0.0))
        env["CIL_SEMA_EARLY_STOP_PATIENCE"] = str(sf.get("early_stop_patience", 20))
        env["CIL_SEMA_EARLY_STOP_MIN_DELTA"] = str(sf.get("early_stop_min_delta", 1e-4))
        env["CIL_SEMA_ADAPTER_HIDDEN_DIM"] = str(sf.get("adapter_hidden_dim", 16))
        env["CIL_SEMA_RD_HIDDEN_DIM"] = str(sf.get("rd_hidden_dim", 128))
        env["CIL_SEMA_RD_BOTTLENECK_DIM"] = str(sf.get("rd_bottleneck_dim", 64))
        env["CIL_SEMA_EXPANSION_LAYERS"] = ",".join(str(x) for x in sf.get("expansion_layers", [9, 10, 11]))
        env["CIL_SEMA_EXPANSION_THRESHOLD"] = str(sf.get("expansion_threshold", 1.0))
        env["CIL_SEMA_EXPANSION_MIN_FRACTION"] = str(sf.get("expansion_min_fraction", 0.05))
        env["CIL_SEMA_RD_LOSS_WEIGHT"] = str(sf.get("rd_loss_weight", 1.0))
        env["CIL_SEMA_CLASSIFIER_SCALE"] = str(sf.get("classifier_scale", 20.0))
        env["CIL_SEMA_GRAD_CLIP_NORM"] = str(sf.get("grad_clip_norm", 5.0))
    if method_key == "moeadapterspp":
        mf = method_cfg["train"]["fair"]
        env["CIL_IMAGE_SIZE"] = str(mf.get("image_size", DATA.get("image_size", 224)))
        env["CIL_NUM_WORKERS"] = str(mf.get("num_workers", 4))
        env["CIL_NUM_TASKS"] = str(num_tasks)
        env["CIL_BACKBONE"] = str(mf.get("backbone", "clip_vit_b_16"))
        env["CIL_MOEADAPTERSPP_PRETRAINED"] = "1" if bool(mf.get("pretrained", True)) else "0"
        env["CIL_MOEADAPTERSPP_EPOCHS"] = str(mf.get("epochs", 25))
        env["CIL_MOEADAPTERSPP_LR"] = str(mf.get("lr", 1e-3))
        env["CIL_MOEADAPTERSPP_WEIGHT_DECAY"] = str(mf.get("weight_decay", 0.0))
        env["CIL_MOEADAPTERSPP_EARLY_STOP_PATIENCE"] = str(mf.get("early_stop_patience", 10))
        env["CIL_MOEADAPTERSPP_EARLY_STOP_MIN_DELTA"] = str(mf.get("early_stop_min_delta", 1e-4))
        env["CIL_MOEADAPTERSPP_ROUTER_HIDDEN_DIM"] = str(mf.get("router_hidden_dim", 256))
        env["CIL_MOEADAPTERSPP_EXPERT_HIDDEN_DIM"] = str(mf.get("expert_hidden_dim", 16))
        env["CIL_MOEADAPTERSPP_AE_HIDDEN_DIM"] = str(mf.get("ae_hidden_dim", 256))
        env["CIL_MOEADAPTERSPP_AE_BOTTLENECK_DIM"] = str(mf.get("ae_bottleneck_dim", 64))
        env["CIL_MOEADAPTERSPP_RECOGNITION_LAYER"] = str(mf.get("recognition_layer", 6))
        env["CIL_MOEADAPTERSPP_SUBSEQUENT_LAYERS"] = ",".join(
            str(x) for x in mf.get("subsequent_layers", [7, 8, 9, 10, 11])
        )
        env["CIL_MOEADAPTERSPP_INITIAL_EXPERTS"] = str(mf.get("initial_experts", 2))
        env["CIL_MOEADAPTERSPP_TOP_K"] = str(mf.get("top_k", 2))
        env["CIL_MOEADAPTERSPP_EXPANSION_THRESHOLD"] = str(mf.get("expansion_threshold", 1.0))
        env["CIL_MOEADAPTERSPP_EXPANSION_MIN_FRACTION"] = str(mf.get("expansion_min_fraction", 0.05))
        env["CIL_MOEADAPTERSPP_LEAS_LOSS_WEIGHT"] = str(mf.get("leas_loss_weight", 1.0))
        env["CIL_MOEADAPTERSPP_DEEC_LOSS_WEIGHT"] = str(mf.get("deec_loss_weight", 1.0))
        env["CIL_MOEADAPTERSPP_CLASSIFIER_SCALE"] = str(mf.get("classifier_scale", 20.0))
        env["CIL_MOEADAPTERSPP_GRAD_CLIP_NORM"] = str(mf.get("grad_clip_norm", 5.0))
        env["CIL_MOEADAPTERSPP_LABEL_SMOOTHING"] = str(mf.get("label_smoothing", 0.0))
        env["CIL_MOEADAPTERSPP_PAPER_MODE"] = "1" if bool(mf.get("paper_mode", True)) else "0"
        env["CIL_MOEADAPTERSPP_PAPER_WARMUP_EPOCHS"] = str(mf.get("paper_warmup_epochs", 1))
        env["CIL_MOEADAPTERSPP_PAPER_PREFERENCE_WINDOW"] = str(mf.get("paper_preference_window", 10))
    if method_key == "more":
        mf = method_cfg["train"]["fair"]
        env["CIL_IMAGE_SIZE"] = str(mf.get("image_size", DATA.get("image_size", 224)))
        env["CIL_NUM_WORKERS"] = str(mf.get("num_workers", 4))
        env["CIL_NUM_TASKS"] = str(num_tasks)
        env["CIL_MORE_EPOCHS"] = str(mf.get("epochs", 40))
        env["CIL_MORE_LR"] = str(mf.get("lr", 1e-3))
        env["CIL_MORE_WEIGHT_DECAY"] = str(mf.get("weight_decay", 5e-4))
        env["CIL_MORE_MOMENTUM"] = str(mf.get("momentum", 0.9))
        env["CIL_MORE_EARLY_STOP_PATIENCE"] = str(mf.get("early_stop_patience", 20))
        env["CIL_MORE_EARLY_STOP_MIN_DELTA"] = str(mf.get("early_stop_min_delta", 1e-4))
        env["CIL_MORE_HIDDEN_DIM"] = str(mf.get("hidden_dim", 256))
        env["CIL_MORE_DROPOUT"] = str(mf.get("dropout", 0.1))
        env["CIL_MORE_REPLAY_PERCENT"] = str(mf.get("replay_percent", 0.05))
        env["CIL_MORE_BACK_UPDATE"] = "1" if bool(mf.get("back_update", True)) else "0"
        env["CIL_MORE_BACK_UPDATE_EPOCHS"] = str(mf.get("back_update_epochs", 10))
        env["CIL_MORE_BACK_UPDATE_LR"] = str(mf.get("back_update_lr", 0.01))
        env["CIL_MORE_BACK_UPDATE_BATCH_SIZE"] = str(mf.get("back_update_batch_size", 16))
        env["CIL_MORE_DISTANCE_SCALE"] = str(mf.get("distance_scale", 20.0))
        env["CIL_MORE_USE_DISTANCE_COEFF"] = "1" if bool(mf.get("use_distance_coeff", True)) else "0"
        env["CIL_MORE_GRAD_CLIP_NORM"] = str(mf.get("grad_clip_norm", 5.0))
    if method_key == "itaml":
        itf = method_cfg["train"]["fair"]
        env["CIL_IMAGE_SIZE"] = str(itf.get("image_size", DATA.get("image_size", 224)))
        env["CIL_NUM_WORKERS"] = str(itf.get("num_workers", 4))
        env["CIL_NUM_TASKS"] = str(num_tasks)
        env["CIL_ITAML_EPOCHS"] = str(itf.get("epochs", 70))
        env["CIL_ITAML_LR"] = str(itf.get("lr", 0.01))
        env["CIL_ITAML_OPTIMIZER"] = str(itf.get("optimizer", "radam"))
        env["CIL_ITAML_WEIGHT_DECAY"] = str(itf.get("weight_decay", 5e-4))
        env["CIL_ITAML_MOMENTUM"] = str(itf.get("momentum", 0.9))
        env["CIL_ITAML_EARLY_STOP_PATIENCE"] = str(itf.get("early_stop_patience", 20))
        env["CIL_ITAML_EARLY_STOP_MIN_DELTA"] = str(itf.get("early_stop_min_delta", 1e-4))
        env["CIL_ITAML_HIDDEN_DIM"] = str(itf.get("hidden_dim", 512))
        env["CIL_ITAML_EMBED_DIM"] = str(itf.get("embed_dim", 256))
        env["CIL_ITAML_DROPOUT"] = str(itf.get("dropout", 0.1))
        env["CIL_ITAML_REPLAY_PERCENT"] = str(itf.get("replay_percent", 0.05))
        env["CIL_ITAML_INNER_STEPS"] = str(itf.get("inner_steps", 1))
        env["CIL_ITAML_BETA"] = str(itf.get("beta", 1.0))
        env["CIL_ITAML_CONTINUUM_SIZE"] = str(itf.get("continuum_size", 1))
        env["CIL_ITAML_GRAD_CLIP_NORM"] = str(itf.get("grad_clip_norm", 5.0))
    if method_key == "diva":
        df = method_cfg["train"]["fair"]
        env["CIL_IMAGE_SIZE"] = str(df.get("image_size", DATA.get("image_size", 224)))
        env["CIL_NUM_WORKERS"] = str(df.get("num_workers", 4))
        env["CIL_NUM_TASKS"] = str(num_tasks)
        env["CIL_DIVA_EPOCHS"] = str(df.get("epochs", 100))
        env["CIL_DIVA_LR"] = str(df.get("lr", 1e-3))
        env["CIL_DIVA_WEIGHT_DECAY"] = str(df.get("weight_decay", 0.0))
        env["CIL_DIVA_EARLY_STOP_PATIENCE"] = str(df.get("early_stop_patience", 20))
        env["CIL_DIVA_EARLY_STOP_MIN_DELTA"] = str(df.get("early_stop_min_delta", 1e-4))
        env["CIL_DIVA_HIDDEN_DIM"] = str(df.get("hidden_dim", 512))
        env["CIL_DIVA_Z_DIM"] = str(df.get("z_dim", 128))
        env["CIL_DIVA_DROPOUT"] = str(df.get("dropout", 0.1))
        env["CIL_DIVA_GENERATED_REPLAY_PER_CLASS"] = str(df.get("generated_replay_per_class", 600))
        env["CIL_DIVA_REPLAY_TEMPERATURE"] = str(df.get("replay_temperature", 1.0))
        env["CIL_DIVA_LAMBDA_CLS"] = str(df.get("lambda_cls", 10.0))
        env["CIL_DIVA_BETA_KL"] = str(df.get("beta_kl", 1.0))
        env["CIL_DIVA_RECON_WEIGHT"] = str(df.get("recon_weight", 1.0))
        env["CIL_DIVA_INPUT_NOISE_STD"] = str(df.get("input_noise_std", 0.0))
        env["CIL_DIVA_WIDE_WIDEN_FACTOR"] = str(df.get("wide_widen_factor", 4))
        env["CIL_DIVA_DOMAIN_TRANSLATION"] = "1" if bool(df.get("domain_translation", False)) else "0"
        env["CIL_DIVA_DT_EPOCHS"] = str(df.get("dt_epochs", 10))
        env["CIL_DIVA_DT_LR"] = str(df.get("dt_lr", 2e-4))
        env["CIL_DIVA_DT_CHANNELS"] = str(df.get("dt_channels", 32))
        env["CIL_DIVA_DT_CYCLE_WEIGHT"] = str(df.get("dt_cycle_weight", 10.0))
        env["CIL_DIVA_DT_IDENTITY_WEIGHT"] = str(df.get("dt_identity_weight", 0.5))
        env["CIL_DIVA_GRAD_CLIP_NORM"] = str(df.get("grad_clip_norm", 5.0))
    if method_key == "genclassifier":
        gf = method_cfg["train"]["fair"]
        env["CIL_IMAGE_SIZE"] = str(gf.get("image_size", DATA.get("image_size", 224)))
        env["CIL_NUM_WORKERS"] = str(gf.get("num_workers", 4))
        env["CIL_GC_VAE_EPOCHS"] = str(gf.get("vae_epochs", 120))
        env["CIL_GC_VAE_LR"] = str(gf.get("vae_lr", 1e-3))
        env["CIL_GC_Z_DIM"] = str(gf.get("z_dim", 128))
        env["CIL_GC_H_DIM"] = str(gf.get("h_dim", 512))
        env["CIL_GC_EVAL_S"] = str(gf.get("eval_importance_samples", 200))
        env["CIL_GC_BACKBONE"] = str(gf.get("backbone", "resnet18"))
        env["CIL_GC_EARLY_STOP_PATIENCE"] = str(gf.get("early_stop_patience", 30))
        env["CIL_GC_EARLY_STOP_MIN_DELTA"] = str(gf.get("early_stop_min_delta", 1e-4))
        env["CIL_NUM_TASKS"] = str(num_tasks)
    extra_env = launcher.get("env", {})
    if isinstance(extra_env, dict):
        for k, v in extra_env.items():
            env[str(k)] = str(v).format(**context)

    # Shared efficiency helper + sibling repos (e.g., MRFA -> PyCIL).
    baselines_common = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines_0610", "common")
    )
    py_paths = []
    if os.path.isdir(baselines_common):
        py_paths.append(baselines_common)
    repo_dir = os.path.abspath(str(launcher.get("repo_dir", "")).strip()) if launcher else ""
    if repo_dir:
        repo_parent = os.path.dirname(repo_dir)
        py_paths.extend([repo_parent, repo_dir])
    if py_paths:
        existing = env.get("PYTHONPATH", "")
        if existing:
            py_paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(py_paths)

    sched_env = _cil_neu_xsdd_schedule()
    if sched_env is not None:
        co, inc, _gs = sched_env
        env["CIL_CLASS_ORDER"] = json.dumps(co, ensure_ascii=False)
        env["CIL_INCREMENTS"] = json.dumps(inc, ensure_ascii=False)
        env["CIL_TASK_SPLITS"] = json.dumps(DATA["task_splits"], ensure_ascii=False)

    env = merge_deterministic_env(env, seed=context["seed"])

    command_preview = " ".join(shlex.quote(x) for x in command_parts)
    return command_parts, env, command_preview


def _merge_launcher(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for k, v in override.items():
        merged[k] = v
    return merged


def _format_seconds(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _infer_max_tasks_from_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None) -> int:
    for i, tok in enumerate(cmd):
        if tok in ("--CI_task_count", "--test_max_task_count", "--max_task_count", "--num-tasks") and i + 1 < len(cmd):
            try:
                v = int(cmd[i + 1])
                if v > 0:
                    return v
            except ValueError:
                pass
    if env:
        raw_num_tasks = str(env.get("CIL_NUM_TASKS", "")).strip()
        if raw_num_tasks:
            try:
                v = int(raw_num_tasks)
                if v > 0:
                    return v
            except ValueError:
                pass
        raw_task_splits = str(env.get("CIL_TASK_SPLITS", "")).strip()
        if raw_task_splits:
            try:
                parsed = json.loads(raw_task_splits)
                if isinstance(parsed, list) and parsed:
                    return max(1, len(parsed))
            except Exception:
                pass
    return 6


def _progress_percent(state: Dict) -> float:
    p = state.get("percent")
    if p is None:
        return 0.0
    try:
        return float(p)
    except (TypeError, ValueError):
        return 0.0


def _set_progress_percent(state: Dict, value: float) -> float:
    p = min(99.9, max(0.0, float(value)))
    state["percent"] = p
    return p


def _task_start_percent(task_idx: int, max_tasks: int, intra: float = 0.02) -> float:
    max_tasks = max(1, int(max_tasks))
    task_idx = max(0, int(task_idx))
    intra = min(0.99, max(0.0, float(intra)))
    return min(99.9, 100.0 * (task_idx + intra) / max_tasks)


def _pycil_combined_percent(state: Dict) -> float:
    """Overall progress for PyCIL-style trainers across tasks."""
    max_tasks = max(1, int(state.get("max_tasks", 6)))
    task_idx = max(0, int(state.get("task_idx", 0)))
    epoch_cur = max(0, int(state.get("epoch_cur", 0)))
    epoch_total = max(1, int(state.get("epoch_total", 1)))
    return min(99.9, 100.0 * (task_idx + epoch_cur / float(epoch_total)) / max_tasks)


def _parse_log_step_value(token: str) -> Optional[int]:
    try:
        return int(float(str(token).strip()))
    except (TypeError, ValueError):
        return None


def _update_progress_from_line(line: str, state: Dict) -> Optional[float]:
    m = re.search(r"Task\s+(\d+)\s*,\s*Epoch\s+(\d+)\s*/\s*(\d+)", line)
    if m:
        task_idx = int(m.group(1))
        epoch_cur = int(m.group(2))
        epoch_total = max(1, int(m.group(3)))
        max_tasks = max(1, int(state.get("max_tasks", 6)))
        state["task_idx"] = task_idx
        state["epoch_cur"] = epoch_cur
        state["epoch_total"] = epoch_total
        state["detail"] = (
            f"task {task_idx + 1}/{max_tasks} epoch {epoch_cur}/{epoch_total}"
        )
        return _set_progress_percent(state, _pycil_combined_percent(state))

    m = re.search(r"Task\s+(\d+)\s+Epoch\s+(\d+)\s*/\s*(\d+)", line)
    if m and "BackUpdate" not in line:
        task_idx = int(m.group(1))
        epoch_cur = int(m.group(2))
        epoch_total = max(1, int(m.group(3)))
        max_tasks = max(1, int(state.get("max_tasks", 6)))
        state["task_idx"] = task_idx
        state["epoch_cur"] = epoch_cur
        state["epoch_total"] = epoch_total
        state["detail"] = (
            f"task {task_idx + 1}/{max_tasks} epoch {epoch_cur}/{epoch_total}"
        )
        return _set_progress_percent(state, _pycil_combined_percent(state))

    m = re.search(r"\[PEC\]\s*task\s+(\d+)\s*/\s*(\d+)\s*epoch\s+(\d+)\s*/\s*(\d+)", line, re.I)
    if m:
        task_idx = max(0, int(m.group(1)) - 1)
        max_tasks = max(1, int(m.group(2)))
        epoch_cur = int(m.group(3))
        epoch_total = max(1, int(m.group(4)))
        state["tracker"] = "pec"
        state["task_idx"] = task_idx
        state["epoch_cur"] = epoch_cur
        state["epoch_total"] = epoch_total
        state["max_tasks"] = max_tasks
        state["detail"] = f"task {task_idx + 1}/{max_tasks} epoch {epoch_cur}/{epoch_total}"
        return _set_progress_percent(state, _pycil_combined_percent(state))

    m = re.search(r"SEED task\s+(\d+)\s*/\s*(\d+)", line, re.I)
    if m:
        task_idx = max(0, int(m.group(1)) - 1)
        max_tasks = max(1, int(m.group(2)))
        state["tracker"] = "seed"
        state["task_idx"] = task_idx
        state["max_tasks"] = max_tasks
        state["epoch_cur"] = 0
        state["epoch_total"] = int(state.get("seed_epoch_total", state.get("epoch_total", 1)))
        state["detail"] = f"task {task_idx + 1}/{max_tasks}"
        return _set_progress_percent(state, _task_start_percent(task_idx, max_tasks))

    m = re.search(r"Training backbone on task\s+(\d+)", line, re.I)
    if m:
        task_idx = int(m.group(1))
        max_tasks = max(1, int(state.get("max_tasks", 6)))
        state["tracker"] = "seed"
        state["task_idx"] = task_idx
        state["epoch_cur"] = 0
        state["epoch_total"] = int(state.get("seed_nepochs", state.get("epoch_total", 1)))
        state["detail"] = f"task {task_idx + 1}/{max_tasks} backbone"
        return _set_progress_percent(state, _task_start_percent(task_idx, max_tasks))

    m = re.search(r"Finetuning backbone\s+\d+\s+on task\s+(\d+)", line, re.I)
    if m:
        task_idx = int(m.group(1))
        max_tasks = max(1, int(state.get("max_tasks", 6)))
        state["tracker"] = "seed"
        state["task_idx"] = task_idx
        state["epoch_cur"] = 0
        state["epoch_total"] = int(state.get("seed_ftepochs", state.get("epoch_total", 1)))
        state["detail"] = f"task {task_idx + 1}/{max_tasks} finetune"
        return _set_progress_percent(state, _task_start_percent(task_idx, max_tasks))

    m = re.search(r"Epoch:\s*(\d+)(?:\s*/\s*(\d+))?\b", line)
    if m and state.get("tracker") == "seed":
        epoch_cur = int(m.group(1))
        epoch_total = max(1, int(m.group(2) or state.get("epoch_total", state.get("seed_nepochs", 1))))
        max_tasks = max(1, int(state.get("max_tasks", 6)))
        task_idx = max(0, int(state.get("task_idx", 0)))
        state["epoch_cur"] = epoch_cur
        state["epoch_total"] = epoch_total
        state["detail"] = f"task {task_idx + 1}/{max_tasks} epoch {epoch_cur}/{epoch_total}"
        return _set_progress_percent(state, _pycil_combined_percent(state))

    m = re.search(r"SEMA\s+task\s+(\d+)\s*:\s*epoch\s+(\d+)\s*/\s*(\d+)", line, re.I)
    if m:
        task_idx = max(0, int(m.group(1)) - 1)
        epoch_cur = int(m.group(2))
        epoch_total = max(1, int(m.group(3)))
        max_tasks = max(1, int(state.get("max_tasks", 6)))
        state["tracker"] = "sema"
        state["task_idx"] = task_idx
        state["max_tasks"] = max_tasks
        state["epoch_cur"] = epoch_cur
        state["epoch_total"] = epoch_total
        state["detail"] = f"task {task_idx + 1}/{max_tasks} epoch {epoch_cur}/{epoch_total}"
        return _set_progress_percent(state, _pycil_combined_percent(state))

    m = re.search(r"MoE-Adapters\+\+\s+task\s+(\d+)\s*:\s*epoch\s+(\d+)\s*/\s*(\d+)", line, re.I)
    if m:
        task_idx = max(0, int(m.group(1)) - 1)
        epoch_cur = int(m.group(2))
        epoch_total = max(1, int(m.group(3)))
        max_tasks = max(1, int(state.get("max_tasks", 6)))
        state["tracker"] = "moeadapterspp"
        state["task_idx"] = task_idx
        state["max_tasks"] = max_tasks
        state["epoch_cur"] = epoch_cur
        state["epoch_total"] = epoch_total
        state["detail"] = f"task {task_idx + 1}/{max_tasks} epoch {epoch_cur}/{epoch_total}"
        return _set_progress_percent(state, _pycil_combined_percent(state))

    m = re.search(r"\btask\s+(\d+)\s*/\s*(\d+)\s+epoch\s+(\d+)\s*/\s*(\d+)", line, re.I)
    if m:
        task_idx = max(0, int(m.group(1)) - 1)
        max_tasks = max(1, int(m.group(2)))
        epoch_cur = int(m.group(3))
        epoch_total = max(1, int(m.group(4)))
        state["tracker"] = "task_epoch"
        state["task_idx"] = task_idx
        state["max_tasks"] = max_tasks
        state["epoch_cur"] = epoch_cur
        state["epoch_total"] = epoch_total
        state["detail"] = f"task {task_idx + 1}/{max_tasks} epoch {epoch_cur}/{epoch_total}"
        return _set_progress_percent(state, _pycil_combined_percent(state))

    m = re.search(r"\btask\s+(\d+)\s*,?\s*:?\s*(?:Epoch|epoch)\s+(\d+)\s*/\s*(\d+)", line, re.I)
    if m:
        task_idx = max(0, int(m.group(1)))
        epoch_cur = int(m.group(2))
        epoch_total = max(1, int(m.group(3)))
        max_tasks = max(1, int(state.get("max_tasks", 6)))
        state["tracker"] = "task_epoch"
        state["task_idx"] = task_idx
        state["max_tasks"] = max_tasks
        state["epoch_cur"] = epoch_cur
        state["epoch_total"] = epoch_total
        state["detail"] = f"task {task_idx + 1}/{max_tasks} epoch {epoch_cur}/{epoch_total}"
        return _set_progress_percent(state, _pycil_combined_percent(state))

    m = re.search(r"\[DER\]\s*task\s+(\d+)\s*/\s*(\d+)\s+train\s+(\d+)\s*/\s*(\d+)", line, re.I)
    if m:
        task_idx = max(0, int(m.group(1)) - 1)
        max_tasks = max(1, int(m.group(2)))
        cur = int(m.group(3))
        total = max(1, int(m.group(4)))
        state["tracker"] = "der"
        state["task_idx"] = task_idx
        state["max_tasks"] = max_tasks
        state["detail"] = f"task {task_idx + 1}/{max_tasks} train {cur}/{total}"
        return _set_progress_percent(
            state,
            min(99.9, 100.0 * (task_idx + cur / float(total)) / max_tasks),
        )

    m = re.search(r"\[TPL\]\s+running:\s+.*--task\s+(\d+).*--num_train_epochs\s+(\d+)", line, re.I)
    if m:
        task_idx = int(m.group(1))
        epoch_total = max(1, int(m.group(2)))
        max_tasks = max(1, int(state.get("max_tasks", 6)))
        state["tracker"] = "tpl"
        state["task_idx"] = task_idx
        state["epoch_cur"] = 0
        state["epoch_total"] = epoch_total
        state["detail"] = f"task {task_idx + 1}/{max_tasks} train"
        return _set_progress_percent(state, _task_start_percent(task_idx, max_tasks))

    m = re.search(r"R\d+T\[(\d+)\s*/\s*(\d+)\]E\[(\d+)\s*/\s*(\d+)\]", line, re.I)
    if m:
        task_idx = max(0, int(m.group(1)) - 1)
        max_tasks = max(1, int(m.group(2)))
        epoch_cur = int(m.group(3))
        epoch_total = max(1, int(m.group(4)))
        state["tracker"] = "tagfex"
        state["task_idx"] = task_idx
        state["max_tasks"] = max_tasks
        state["epoch_cur"] = epoch_cur
        state["epoch_total"] = epoch_total
        state["detail"] = f"task {task_idx + 1}/{max_tasks} epoch {epoch_cur}/{epoch_total}"
        return _set_progress_percent(state, _pycil_combined_percent(state))

    m = re.search(r"Task\s+(\d+)\s*/\s*(\d+):\s*(\d{1,3})%\|.*\|\s*(\d+)\s*/\s*(\d+)", line)
    if m and "updmem" not in line and "rdcmem" not in line:
        task_idx = max(0, int(m.group(1)) - 1)
        max_tasks = max(1, int(m.group(2)))
        epoch_cur = int(m.group(4))
        epoch_total = max(1, int(m.group(5)))
        state["tracker"] = "tagfex"
        state["task_idx"] = task_idx
        state["max_tasks"] = max_tasks
        state["epoch_cur"] = epoch_cur
        state["epoch_total"] = epoch_total
        state["detail"] = f"task {task_idx + 1}/{max_tasks} epoch {epoch_cur}/{epoch_total}"
        return _set_progress_percent(state, _pycil_combined_percent(state))

    m = re.search(r"GenClassifier\s+session\s+(\d+)\s*/\s*(\d+)", line, re.I)
    if m:
        task_idx = max(0, int(m.group(1)) - 1)
        max_tasks = max(1, int(m.group(2)))
        state["tracker"] = "genclassifier"
        state["task_idx"] = task_idx
        state["max_tasks"] = max_tasks
        state["detail"] = f"task {task_idx + 1}/{max_tasks}"
        return _set_progress_percent(state, _task_start_percent(task_idx, max_tasks))

    m = re.search(r"(DiVA|MoVE)\s+session\s+(\d+)\s*/\s*(\d+)", line, re.I)
    if m:
        task_idx = max(0, int(m.group(2)) - 1)
        max_tasks = max(1, int(m.group(3)))
        state["tracker"] = str(m.group(1)).lower()
        state["task_idx"] = task_idx
        state["max_tasks"] = max_tasks
        state["detail"] = f"task {task_idx + 1}/{max_tasks}"
        return _set_progress_percent(state, _task_start_percent(task_idx, max_tasks))

    m = re.search(r"DER-paper\s+session\s+(\d+)\s*/\s*(\d+)", line, re.I)
    if m:
        task_idx = max(0, int(m.group(1)) - 1)
        max_tasks = max(1, int(m.group(2)))
        state["tracker"] = "der"
        state["task_idx"] = task_idx
        state["max_tasks"] = max_tasks
        state["detail"] = f"task {task_idx + 1}/{max_tasks}"
        return _set_progress_percent(state, _task_start_percent(task_idx, max_tasks))

    m = re.search(r"(\d{1,3})%\|[^|]*\|\s*(\d+)\s*/\s*(\d+)", line)
    if m and state.get("epoch_total") is None:
        try:
            cur, total = int(m.group(2)), max(1, int(m.group(3)))
            state["step_frac"] = cur / float(total)
        except ValueError:
            pass
        else:
            state["detail"] = state.get("detail") or f"training {cur}/{total}"
            sub_p = float(m.group(1))
            return _set_progress_percent(state, sub_p)

    if m and state.get("tracker") == "tpl":
        try:
            cur, total = int(m.group(2)), max(1, int(m.group(3)))
        except ValueError:
            pass
        else:
            task_idx = max(0, int(state.get("task_idx", 0)))
            max_tasks = max(1, int(state.get("max_tasks", 6)))
            state["detail"] = f"task {task_idx + 1}/{max_tasks} train {cur}/{total}"
            return _set_progress_percent(
                state,
                min(99.9, 100.0 * (task_idx + cur / float(total)) / max_tasks),
            )

    cur = _progress_percent(state)
    return cur if cur > 0 else None


def _run_external_process_with_progress(
    cmd: List[str],
    cwd: str,
    env: Dict[str, str],
    stdout_path: str,
    stderr_path: str,
    progress_label: str,
    report_interval_sec: float = 20.0,
) -> Dict:
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )

    sel = selectors.DefaultSelector()
    assert proc.stdout is not None
    assert proc.stderr is not None
    sel.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    sel.register(proc.stderr, selectors.EVENT_READ, data="stderr")

    last_report = 0.0
    progress_percent = None
    progress_state = {
        "max_tasks": _infer_max_tasks_from_cmd(cmd, env),
        "task_idx": 0,
        "phase": 1,
        "sample_frac": 0.0,
        "stage_frac": 0.0,
        "detail": "",
        "percent": 0.0,
        "seed_nepochs": int(str(env.get("CIL_SEED_NEPOCHS", "0")).strip() or 0),
        "seed_ftepochs": int(str(env.get("CIL_SEED_FTEPOCHS", "0")).strip() or 0),
    }
    tail_stdout = deque(maxlen=120)
    tail_stderr = deque(maxlen=120)

    with open(stdout_path, "w", encoding="utf-8") as f_out, open(stderr_path, "w", encoding="utf-8") as f_err:
        while True:
            events = sel.select(timeout=1.0)
            for key, _ in events:
                stream_name = key.data
                stream = key.fileobj
                line = stream.readline()
                if line == "":
                    try:
                        sel.unregister(stream)
                    except Exception:
                        pass
                    continue

                if stream_name == "stdout":
                    f_out.write(line)
                    f_out.flush()
                    tail_stdout.append(line.rstrip("\n"))
                else:
                    f_err.write(line)
                    f_err.flush()
                    tail_stderr.append(line.rstrip("\n"))

                p = _update_progress_from_line(line, progress_state)
                if p is not None:
                    progress_percent = p

            now = time.time()
            if now - last_report >= report_interval_sec:
                elapsed = now - t0
                detail = str(progress_state.get("detail") or "").strip()
                detail_suffix = f" ({detail})" if detail else ""
                if progress_percent is not None and progress_percent > 0:
                    eta = elapsed * (100.0 / progress_percent - 1.0)
                    print(
                        f"[{progress_label}] elapsed={_format_seconds(elapsed)} "
                        f"progress={progress_percent:.1f}%{detail_suffix} "
                        f"eta={_format_seconds(eta)}"
                    )
                else:
                    wait_detail = detail or "waiting_for_epoch_log"
                    print(f"[{progress_label}] elapsed={_format_seconds(elapsed)} {wait_detail}")
                last_report = now

            if proc.poll() is not None and not sel.get_map():
                break

    rc = int(proc.wait())
    elapsed = time.time() - t0
    if progress_percent is not None and progress_percent > 0 and progress_percent < 100:
        eta = elapsed * (100.0 / progress_percent - 1.0)
        print(
            f"[{progress_label}] finished(rc={rc}) elapsed={_format_seconds(elapsed)} "
            f"last_progress={progress_percent:.1f}% eta_at_last={_format_seconds(eta)}"
        )
    else:
        print(f"[{progress_label}] finished(rc={rc}) elapsed={_format_seconds(elapsed)}")

    return {
        "return_code": rc,
        "elapsed_sec": elapsed,
        "tail_stdout": list(tail_stdout),
        "tail_stderr": list(tail_stderr),
        "progress_percent_last": progress_percent,
    }


def _resolve_repo_dir(repo_dir: str) -> Tuple[Optional[str], List[str]]:
    raw = str(repo_dir).strip()
    if not raw:
        return None, []

    tried = []
    candidates = [raw]

    base_env = os.getenv("CIL_BASELINES_ROOT", "").strip()
    if base_env:
        candidates.append(os.path.join(base_env, os.path.basename(raw.rstrip("/"))))

    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)  # .../Incre_neu
    candidates.append(os.path.join(project_root, "baselines_0610", os.path.basename(raw.rstrip("/"))))

    cwd = os.getcwd()
    candidates.append(os.path.join(cwd, "..", "baselines_0610", os.path.basename(raw.rstrip("/"))))

    normalized = []
    seen = set()
    for c in candidates:
        c_abs = os.path.abspath(c)
        if c_abs in seen:
            continue
        seen.add(c_abs)
        normalized.append(c_abs)

    for c in normalized:
        tried.append(c)
        if os.path.isdir(c):
            return c, tried
    return None, tried


def _ensure_mrfa_neu_xsdd_support(repo_dir: str) -> None:
    repo_dir = os.path.abspath(str(repo_dir))
    dm_path = os.path.join(repo_dir, "utils", "data_manager.py")
    data_path = os.path.join(repo_dir, "utils", "data.py")
    cfg_path = os.path.join(repo_dir, "exps", "icarl", "mrfa", "neu_xsdd.json")

    # 1) Ensure config file exists.
    if not os.path.isfile(cfg_path):
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        cfg_obj = {
            "prefix": "cil",
            "dataset": "neu_xsdd",
            "memory_size": 600,
            "memory_per_class": 0,
            "fixed_memory": False,
            "shuffle": False,
            "class_order": [0, 3, 4, 5, 1, 2, 6, 9, 12, 7, 8, 10, 11],
            "increments": [2, 2, 2, 3, 2, 2],
            "init_cls": 2,
            "increment": 2,
            "save_task_checkpoints": [],
            "load_ckpt": [],
            "perturb_p": [0.0001, 0.0001, 0.0001, 0.0001],
            "num_augmem": 1,
            "disable_perturb": False,
            "auto_kd": False,
            "model_name": "icarl_mrfa",
            "convnet_type": "resnet18",
            "batch_size": 32,
            "num_workers": 4,
            "init_epochs": 200,
            "epochs": 150,
            "early_stop_patience": 20,
            "early_stop_min_delta": 1e-4,
            "lr": 0.1,
            "weight_decay": 0.0005,
            "gamma": 0.1,
            "topk": 1,
            "device": ["0"],
            "seed": [42],
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_obj, f, indent=2, ensure_ascii=False)

    # 2) Ensure data_manager registers neu_xsdd -> iNEUXSDD.
    if os.path.isfile(dm_path):
        with open(dm_path, "r", encoding="utf-8") as f:
            dm_text = f.read()

        if "from utils.data import" in dm_text and "iNEUXSDD" not in dm_text:
            dm_lines = dm_text.splitlines()
            for i, line in enumerate(dm_lines):
                if line.strip().startswith("from utils.data import"):
                    if "iNEUXSDD" not in line:
                        dm_lines[i] = line.rstrip() + ", iNEUXSDD"
                    break
            dm_text = "\n".join(dm_lines) + ("\n" if dm_text.endswith("\n") else "")

        if 'elif name == "neu_xsdd":' not in dm_text and "def _get_idata(dataset_name):" in dm_text:
            dm_lines = dm_text.splitlines()
            def_idx = -1
            for i, line in enumerate(dm_lines):
                if line.strip().startswith("def _get_idata(dataset_name):"):
                    def_idx = i
                    break
            if def_idx >= 0:
                else_idx = -1
                for i in range(def_idx + 1, len(dm_lines)):
                    if dm_lines[i].startswith("def "):
                        break
                    if dm_lines[i].strip().startswith("else:"):
                        else_idx = i
                        break
                if else_idx >= 0:
                    dm_lines.insert(else_idx, '    elif name == "neu_xsdd":')
                    dm_lines.insert(else_idx + 1, "        return iNEUXSDD()")
                    dm_text = "\n".join(dm_lines) + "\n"

        with open(dm_path, "w", encoding="utf-8") as f:
            f.write(dm_text)

    # 3) Ensure iNEUXSDD exists in utils/data.py (older MRFA snapshots may not have it).
    if os.path.isfile(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data_text = f.read()

        changed = False
        if "import os" not in data_text:
            data_text = "import os\n" + data_text
            changed = True
        if "import re" not in data_text:
            if "import os\n" in data_text:
                data_text = data_text.replace("import os\n", "import os\nimport re\n", 1)
            else:
                data_text = "import re\n" + data_text
            changed = True
        if "import json" not in data_text:
            if "import re\n" in data_text:
                data_text = data_text.replace("import re\n", "import re\nimport json\n", 1)
            elif "import os\n" in data_text:
                data_text = data_text.replace("import os\n", "import os\nimport json\n", 1)
            else:
                data_text = "import json\n" + data_text
            changed = True

        if "class iNEUXSDD(iData):" not in data_text:
            data_text += """

def _normalize_token(text):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text.strip().lower())).strip("_")


def _collect_split_samples(root_dir, split_name):
    image_dir = os.path.join(root_dir, "images", split_name)
    if not os.path.isdir(image_dir):
        return []
    samples = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            samples.append(os.path.join(image_dir, fname))
    return samples


def _parse_label_file(root_dir, split_name, stem, root_kind, class_name_to_gid):
    label_path = os.path.join(root_dir, "labels", split_name, f"{stem}.txt")
    if not os.path.isfile(label_path):
        return None
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            tok = parts[0]
            try:
                cid = int(tok)
                if root_kind == "neu" and 0 <= cid <= 5:
                    return cid
                if root_kind == "xsdd" and 0 <= cid <= 6:
                    return cid + 6
            except ValueError:
                pass
            key = _normalize_token(tok)
            if key in class_name_to_gid:
                return class_name_to_gid[key]
            full = _normalize_token(line.strip())
            if full in class_name_to_gid:
                return class_name_to_gid[full]
    return None


def _parse_label_name(stem, class_name_to_gid):
    m = re.match(r"^(.+?)_(\\d+)$", stem)
    if m is None:
        return None
    return class_name_to_gid.get(_normalize_token(m.group(1)), None)


class iNEUXSDD(iData):
    use_path = True
    train_trsf = [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip()]
    test_trsf = [transforms.Resize((224, 224))]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = [0, 3, 4, 5, 1, 2, 6, 9, 12, 7, 8, 10, 11]

    def download_data(self):
        raw_co = os.getenv("CIL_CLASS_ORDER", "").strip()
        if raw_co:
            try:
                parsed = json.loads(raw_co)
                if isinstance(parsed, list) and parsed:
                    self.class_order = [int(x) for x in parsed]
            except Exception:
                pass

        roots_env = os.getenv("CIL_DATA_ROOTS", "").strip()
        roots_raw = [x.strip() for x in roots_env.split(",") if x.strip()] if roots_env else []
        if len(roots_raw) < 2:
            fallback = os.getenv("CIL_DATA_ROOT", "").strip()
            if "," in fallback:
                roots_raw = [x.strip() for x in fallback.split(",") if x.strip()]
        if len(roots_raw) >= 2:
            roots = [("neu", roots_raw[0]), ("xsdd", roots_raw[1])]
        else:
            raise RuntimeError("iNEUXSDD requires two roots via CIL_DATA_ROOTS=neu_root,xsdd_root")

        for _, r in roots:
            if not os.path.isdir(r):
                raise FileNotFoundError(f"dataset root not found: {r}")

        class_names = [
            "Crazing",
            "Inclusion",
            "Patches",
            "Pitted_Surface",
            "Rolled-in_Scale",
            "Scratches",
            "finishing_roll_printing",
            "iron_sheet_ash",
            "oxide_scale_of_plate_system",
            "oxide_scale_of_temperature_system",
            "red_iron",
            "slag_inclusion",
            "surface_scratch",
        ]
        class_name_to_gid = {_normalize_token(n): i for i, n in enumerate(class_names)}
        valid_gid = set(self.class_order)

        train_paths, train_labels = [], []
        test_paths, test_labels = [], []
        for root_kind, root_dir in roots:
            for p in _collect_split_samples(root_dir, "train"):
                stem = os.path.splitext(os.path.basename(p))[0]
                gid = _parse_label_file(root_dir, "train", stem, root_kind, class_name_to_gid)
                if gid is None:
                    gid = _parse_label_name(stem, class_name_to_gid)
                if gid is None or gid not in valid_gid:
                    continue
                train_paths.append(p)
                train_labels.append(gid)

            split_eval = "test"
            eval_paths = _collect_split_samples(root_dir, split_eval)
            if not eval_paths:
                split_eval = "val"
                eval_paths = _collect_split_samples(root_dir, split_eval)
            for p in eval_paths:
                stem = os.path.splitext(os.path.basename(p))[0]
                gid = _parse_label_file(root_dir, split_eval, stem, root_kind, class_name_to_gid)
                if gid is None:
                    gid = _parse_label_name(stem, class_name_to_gid)
                if gid is None or gid not in valid_gid:
                    continue
                test_paths.append(p)
                test_labels.append(gid)

        if len(train_paths) == 0 or len(test_paths) == 0:
            raise RuntimeError(
                f"iNEUXSDD empty dataset: train={len(train_paths)} test={len(test_paths)} roots={roots}"
            )

        self.train_data = np.array(train_paths, dtype=object)
        self.train_targets = np.array(train_labels, dtype=np.int64)
        self.test_data = np.array(test_paths, dtype=object)
        self.test_targets = np.array(test_labels, dtype=np.int64)
        print(f"[iNEUXSDD] train={len(self.train_data)} test={len(self.test_data)}")
"""
            changed = True

        if changed:
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(data_text)


def _ensure_tagfex_backbone_support(repo_dir: str, backbone: str) -> None:
    """Ensure resnet34/resnet50 use modules/backbones/resnet.py (dataset_name API)."""
    backbone = str(backbone).lower()
    if backbone not in ("resnet50", "resnet34", "resnet18"):
        return
    repo_dir = os.path.abspath(str(repo_dir))
    init_path = os.path.join(repo_dir, "modules", "backbones", "__init__.py")
    if not os.path.isfile(init_path):
        return
    with open(init_path, "r", encoding="utf-8") as f:
        text = f.read()
    changed = False
    desired_import = "from .resnet import resnet18, resnet10, resnet34, resnet50\n"
    if "from .resnet_pycil import" in text:
        text = text.replace(
            "from .resnet_pycil import resnet18, resnet34, resnet50\n", ""
        )
        changed = True
    if "from .resnet import resnet18, resnet10, resnet34, resnet50" not in text:
        if "from .cifar_resnet import" in text:
            text = text.replace(
                "from .cifar_resnet import resnet32, resnet20\n",
                "from .cifar_resnet import resnet32, resnet20\n" + desired_import,
                1,
            )
        else:
            text = desired_import + text
        changed = True
    for name in ("resnet34", "resnet50"):
        entry = f"    '{name}': {name},\n"
        if f"'{name}':" not in text:
            if "backbone_dict = {" in text:
                text = text.replace(
                    "backbone_dict = {\n",
                    "backbone_dict = {\n" + entry,
                    1,
                )
                changed = True
    if changed:
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(text)

    tagfex_path = os.path.join(repo_dir, "methods", "tagfex", "tagfex.py")
    if os.path.isfile(tagfex_path):
        with open(tagfex_path, "r", encoding="utf-8") as f:
            ttext = f.read()
        if "('resnet18', 'resnet50')" not in ttext and "backbone_configs['name'] == 'resnet18'" in ttext:
            ttext = ttext.replace(
                "if backbone_configs['name'] == 'resnet18':",
                "if backbone_configs['name'] in ('resnet18', 'resnet50', 'resnet34'):",
                1,
            )
            with open(tagfex_path, "w", encoding="utf-8") as f:
                f.write(ttext)


def _ensure_external_method_repo_compatibility(
    method_name: str,
    selected_launcher: Dict,
    method_cfg: Dict,
) -> None:
    if method_name == "incre_mrfa":
        repo_dir = str(selected_launcher.get("repo_dir", "")).strip()
        if repo_dir:
            _ensure_mrfa_neu_xsdd_support(repo_dir)
    if method_name == "incre_tagfex":
        repo_dir = str(selected_launcher.get("repo_dir", "")).strip()
        if repo_dir:
            fair = method_cfg["train"]["fair"]
            _ensure_tagfex_backbone_support(
                repo_dir, str(fair.get("backbone", "resnet18"))
            )


def _maybe_patch_pycil_command(
    method_cfg: Dict,
    launcher: Dict,
    cmd: List[str],
    run_dir: str,
) -> Tuple[List[str], Optional[str]]:
    repo_dir = str(launcher.get("repo_dir", "")).strip()
    if not repo_dir or os.path.basename(repo_dir).lower() != "pycil":
        return cmd, None
    if "--config" not in cmd:
        return cmd, None

    cfg_idx = cmd.index("--config")
    if cfg_idx + 1 >= len(cmd):
        return cmd, None
    cfg_path_raw = cmd[cfg_idx + 1]
    src_cfg_path = cfg_path_raw
    if not os.path.isabs(src_cfg_path):
        src_cfg_path = os.path.join(str(launcher.get("workdir", repo_dir)), cfg_path_raw)
    if not os.path.isfile(src_cfg_path):
        return cmd, None

    try:
        with open(src_cfg_path, "r", encoding="utf-8") as f:
            cfg_obj = json.load(f)
    except Exception:
        return cmd, None

    fair = method_cfg["train"]["fair"]
    method_key = str(method_cfg.get("method_key", "")).lower()
    cfg_obj["seed"] = [int(fair["seed"])]
    cfg_obj["dataset"] = _baseline_cil_dataset_name(ACTIVE_DATASET)
    cfg_obj["shuffle"] = False
    cfg_obj["class_order"] = [int(c) for group in DATA["task_splits"] for c in group]
    cfg_obj["increments"] = [len(group) for group in DATA["task_splits"]]
    cfg_obj["init_cls"] = int(cfg_obj["increments"][0]) if cfg_obj["increments"] else int(fair["init_cls"])
    cfg_obj["increment"] = int(fair["increment"])
    _apply_fair_dict_to_json_cfg(cfg_obj, fair)
    cfg_obj["memory_size"] = int(
        method_cfg["train"].get("memory_budget", fair.get("memory_budget", cfg_obj.get("memory_size", 0)))
    )
    uses_real_replay = not bool(method_cfg.get("train", {}).get("exemplar_free", False))
    if uses_real_replay and "replay_percent" in fair:
        cfg_obj["replay_percent"] = float(fair["replay_percent"])
    elif uses_real_replay:
        cfg_obj["replay_percent"] = 0.05
    if method_key == "beef":
        cfg_obj["fixed_memory"] = bool(fair.get("fixed_memory", False))
        cfg_obj["memory_per_class"] = int(fair.get("memory_per_class", 0))
        cfg_obj["init_epochs"] = int(fair.get("init_epochs", 200))
        cfg_obj["expansion_epochs"] = int(fair.get("expansion_epochs", 170))
        cfg_obj["fusion_epochs"] = int(fair.get("fusion_epochs", 60))
        cfg_obj["epochs"] = int(fair.get("expansion_epochs", 170))
    elif method_key == "der":
        cfg_obj["init_epochs"] = int(fair.get("init_epochs", 200))
        cfg_obj["epochs"] = int(fair.get("epochs", 170))
        cfg_obj["val_ratio"] = float(fair.get("val_ratio", 0.0))
    elif method_key in ("icarl", "ewc"):
        _apply_icarl_schedule_cfg(cfg_obj, fair)
        cfg_obj["val_ratio"] = float(fair.get("val_ratio", 0.0))
    if cfg_obj.get("fixed_memory", False):
        num_classes = max(1, int(DATA["num_classes"]))
        cfg_obj["memory_per_class"] = int(cfg_obj["memory_size"] // num_classes)
    cfg_obj["convnet_type"] = str(fair.get("backbone", cfg_obj.get("convnet_type", "resnet18")))
    cfg_obj["batch_size"] = int(fair.get("batch_size", cfg_obj.get("batch_size", 32)))
    cfg_obj["num_workers"] = int(fair.get("num_workers", cfg_obj.get("num_workers", 4)))
    cfg_obj["image_size"] = int(fair.get("image_size", DATA.get("image_size", 224)))
    cfg_obj["flops_backward_factor"] = float(fair.get("flops_backward_factor", 3.0))
    if "topk" in fair or "topk" in cfg_obj:
        cfg_obj["topk"] = int(fair.get("topk", cfg_obj.get("topk", 1)))
    cfg_obj["device"] = _pycil_visible_cuda_devices()
    if "weight_decay" in fair:
        cfg_obj["weight_decay"] = float(fair["weight_decay"])
    if "init_weight_decay" in fair:
        cfg_obj["init_weight_decay"] = float(fair["init_weight_decay"])
    elif "init_weight_decay" in cfg_obj and "weight_decay" in fair:
        cfg_obj["init_weight_decay"] = float(fair["weight_decay"])
    if "lr" in fair and "lr" in cfg_obj:
        cfg_obj["lr"] = float(fair["lr"])
    if "init_lr" in fair and "init_lr" in cfg_obj:
        cfg_obj["init_lr"] = float(fair["init_lr"])
    elif "init_lr" in cfg_obj and "lr" in fair:
        cfg_obj["init_lr"] = float(fair["lr"])
    if "gamma" in fair or "gamma" in cfg_obj:
        cfg_obj["gamma"] = float(fair.get("gamma", cfg_obj.get("gamma", 0.1)))

    run_dir_abs = os.path.abspath(run_dir)
    patched_cfg_path = os.path.join(run_dir_abs, f"pycil_{method_cfg['method_key']}_fair.json")
    with open(patched_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_obj, f, indent=2, ensure_ascii=False)

    new_cmd = list(cmd)
    new_cmd[cfg_idx + 1] = patched_cfg_path
    return new_cmd, patched_cfg_path


def _method_cfg_for_seed(method_cfg: Dict, seed: int) -> Dict:
    cfg = deepcopy(method_cfg)
    fair = deepcopy(cfg["train"]["fair"])
    fair["seed"] = int(seed)
    cfg["train"]["fair"] = fair
    return cfg


def _record_section_run(
    sections_by_name: Dict[str, List[dict]],
    section_order: List[str],
    section: dict,
    seed: int,
    *,
    run_dir: Optional[str] = None,
    filename: str = "test.txt",
) -> None:
    name = str(section.get("name", ""))
    if name not in sections_by_name:
        sections_by_name[name] = []
        section_order.append(name)
    sections_by_name[name].append({"seed": int(seed), "section": section})
    if run_dir:
        write_seed_test_report(run_dir, section, seed, filename=filename)


def _record_efficiency_run(
    efficiency_by_name: Dict[str, List[dict]],
    efficiency_order: List[str],
    efficiency: Optional[dict],
    section_name: str,
    seed: int,
) -> None:
    if not efficiency:
        return
    name = str(section_name)
    if name not in efficiency_by_name:
        efficiency_by_name[name] = []
        if name not in efficiency_order:
            efficiency_order.append(name)
    row = dict(efficiency)
    row["seed"] = int(seed)
    efficiency_by_name[name].append(row)


def _all_generator_modules(task_generators: Dict) -> List:
    mods = []
    for state in task_generators.values():
        if isinstance(state, dict) and isinstance(state.get("model"), torch.nn.Module):
            mods.append(state["model"])
    return mods


def run_external_incremental_method(
    root_run_dir: str,
    method_name: str,
    seed: int,
    *,
    multi_seed: bool = False,
):
    if method_name not in EXTERNAL_INCREMENTAL_METHODS:
        raise KeyError(f"Unknown external method: {method_name}")

    method_cfg = _method_cfg_for_seed(EXTERNAL_INCREMENTAL_METHODS[method_name], seed)
    run_dir = create_profile_run_dir(
        root_run_dir, profile_subdir(method_name, seed, multi_seed)
    )
    save_config(run_dir, {
        "data": DATA,
        "fair": deepcopy(method_cfg["train"]["fair"]),
        "profile": method_cfg,
    })

    launcher_candidates = method_cfg["train"].get("launcher_candidates", None)
    if not launcher_candidates:
        launcher_candidates = [method_cfg["train"].get("external_launcher", {})]

    summary = {
        "profile": method_name,
        "paper": method_cfg["paper"],
        "year": method_cfg["year"],
        "category": method_cfg["category"],
        "repo_url": method_cfg["repo_url"],
        "status": "configured",
        "fair": method_cfg["train"]["fair"],
        "memory_budget": method_cfg["train"]["memory_budget"],
        "generator_replay": method_cfg["train"]["generator_replay"],
        "exemplar_free": method_cfg["train"]["exemplar_free"],
    }

    selected_launcher = None
    selection_notes = []
    for cand in launcher_candidates:
        if not bool(cand.get("enabled", True)):
            selection_notes.append({"candidate": cand, "status": "disabled"})
            continue
        repo_dir = str(cand.get("repo_dir", "")).strip()
        if not repo_dir:
            selection_notes.append({"candidate": cand, "status": "repo_dir_empty"})
            continue
        repo_resolved, repo_tried = _resolve_repo_dir(repo_dir)
        if repo_resolved is None:
            selection_notes.append({
                "candidate": cand,
                "status": "repo_dir_not_found",
                "tried_paths": repo_tried,
            })
            continue
        workdir_raw = str(cand.get("workdir", "")).strip()
        if workdir_raw:
            if os.path.isabs(workdir_raw):
                workdir = workdir_raw
            else:
                workdir = os.path.join(repo_resolved, workdir_raw)
        else:
            workdir = repo_resolved
        if not os.path.isdir(workdir):
            fallback_workdir = repo_resolved
            if os.path.isdir(fallback_workdir):
                workdir = fallback_workdir
            else:
                selection_notes.append({
                    "candidate": cand,
                    "status": "workdir_not_found",
                    "resolved_repo_dir": repo_resolved,
                    "workdir": workdir,
                })
                continue
        selected_launcher = dict(cand)
        selected_launcher["repo_dir"] = repo_resolved
        selected_launcher["workdir"] = workdir
        selection_notes.append({"candidate": cand, "status": "selected"})
        break

    summary["launcher_selection"] = selection_notes

    class_ids = list(range(DATA["num_classes"]))
    metrics = _default_zero_metrics(class_ids)

    if selected_launcher is None:
        summary["status"] = "skipped"
        summary["reason"] = "no valid launcher candidate found"
        save_summary(run_dir, summary)
        return {
            "metrics": metrics,
            "report_section": {"name": method_name, "items": [{"title": "launcher_not_found", "metrics": metrics, "class_ids": class_ids}]},
        }

    _ensure_external_method_repo_compatibility(method_name, selected_launcher, method_cfg)

    cmd, env, cmd_preview = _build_external_command_and_env(method_cfg, run_dir, selected_launcher)
    patched_cfg_path = None
    cmd, patched_cfg_path = _maybe_patch_pycil_command(method_cfg, selected_launcher, cmd, run_dir)
    patched_repo_json = None
    patched_tagfex_yaml = None
    mk = str(method_cfg.get("method_key", "")).lower()
    has_sequence = bool(
        selected_launcher.get("sequence") and isinstance(selected_launcher.get("sequence"), list)
    )
    if mk == "mrfa":
        cmd, patched_repo_json = _maybe_patch_mrfa_json_cmd(
            method_cfg, selected_launcher, cmd, run_dir
        )
    elif mk == "tagfex":
        cmd, patched_tagfex_yaml = _maybe_patch_tagfex_yaml_cmd(
            method_cfg, selected_launcher, cmd, run_dir
        )
    cmd_preview = " ".join(shlex.quote(x) for x in cmd)
    summary["selected_launcher"] = selected_launcher
    summary["command_preview"] = cmd_preview
    summary["workdir"] = selected_launcher["workdir"]
    if patched_cfg_path is not None:
        summary["patched_pycil_config"] = patched_cfg_path
    if patched_repo_json is not None:
        summary["patched_repo_json"] = patched_repo_json
    if patched_tagfex_yaml is not None:
        summary["patched_tagfex_yaml"] = patched_tagfex_yaml

    if not cmd:
        summary["status"] = "skipped"
        summary["reason"] = "entrypoint/args empty; no command to run"
        save_summary(run_dir, summary)
        return {
            "metrics": metrics,
            "report_section": {"name": method_name, "items": [{"title": "command_missing", "metrics": metrics, "class_ids": class_ids}]},
        }

    sequence = selected_launcher.get("sequence", None)
    stage_results = []
    final_rc = 0
    total_elapsed = 0.0

    if sequence and isinstance(sequence, list):
        for stage_idx, stage in enumerate(sequence):
            stage_name = str(stage.get("name", f"stage{stage_idx + 1}"))
            stage_launcher = _merge_launcher(selected_launcher, stage)
            stage_cmd, stage_env, _ = _build_external_command_and_env(method_cfg, run_dir, stage_launcher)
            stage_cmd, stage_patch = _maybe_patch_pycil_command(method_cfg, stage_launcher, stage_cmd, run_dir)
            stage_repo_json = None
            stage_preview = " ".join(shlex.quote(x) for x in stage_cmd)

            stdout_path = os.path.join(run_dir, f"external_stdout_{stage_name}.log")
            stderr_path = os.path.join(run_dir, f"external_stderr_{stage_name}.log")
            exec_info = _run_external_process_with_progress(
                cmd=stage_cmd,
                cwd=stage_launcher["workdir"],
                env=stage_env,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                progress_label=f"{method_name}:{stage_name}",
            )
            elapsed = float(exec_info["elapsed_sec"])
            total_elapsed += elapsed

            stage_info = {
                "stage": stage_name,
                "command_preview": stage_preview,
                "return_code": int(exec_info["return_code"]),
                "elapsed_sec": elapsed,
                "stdout_log": stdout_path,
                "stderr_log": stderr_path,
            }
            if exec_info.get("progress_percent_last", None) is not None:
                stage_info["progress_percent_last"] = exec_info["progress_percent_last"]
            if stage_patch is not None:
                stage_info["patched_pycil_config"] = stage_patch
            if stage_repo_json is not None:
                stage_info["patched_repo_json"] = stage_repo_json
            if exec_info["return_code"] != 0:
                tail = exec_info["tail_stderr"] or exec_info["tail_stdout"]
                stage_info["error_tail"] = tail[-20:]
            stage_results.append(stage_info)

            if exec_info["return_code"] != 0:
                final_rc = int(exec_info["return_code"])
                break
        summary["stage_results"] = stage_results
    else:
        stdout_path = os.path.join(run_dir, "external_stdout.log")
        stderr_path = os.path.join(run_dir, "external_stderr.log")
        exec_info = _run_external_process_with_progress(
            cmd=cmd,
            cwd=selected_launcher["workdir"],
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            progress_label=method_name,
        )
        elapsed = float(exec_info["elapsed_sec"])
        total_elapsed = elapsed
        final_rc = int(exec_info["return_code"])

        summary["return_code"] = int(exec_info["return_code"])
        summary["stdout_log"] = stdout_path
        summary["stderr_log"] = stderr_path
        if exec_info.get("progress_percent_last", None) is not None:
            summary["progress_percent_last"] = exec_info["progress_percent_last"]
        if exec_info["return_code"] != 0:
            tail = exec_info["tail_stderr"] or exec_info["tail_stdout"]
            summary["error_tail"] = tail[-20:]

    summary["elapsed_sec"] = total_elapsed
    summary["return_code"] = int(final_rc)
    summary["status"] = "finished" if final_rc == 0 else "failed"
    summary["run_dir"] = run_dir

    save_summary(run_dir, summary)
    title = "finished" if final_rc == 0 else "failed"
    report_items = [{"title": title, "metrics": metrics, "class_ids": class_ids}]
    if method_name == "incre_mrfa":
        parsed_items = _build_mrfa_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name == "incre_tagfex":
        parsed_items = _build_tagfex_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name == "seed_paper":
        parsed_items = _build_seed_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name == "incre_tpl":
        parsed_items = _build_tpl_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name == "incre_pec":
        parsed_items = _build_pec_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name == "move_paper":
        parsed_items = _build_move_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name == "incre_sema":
        parsed_items = _build_sema_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name == "moeadapterspp_paper":
        parsed_items = _build_moeadapterspp_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name in {"more_paper", "more_paper_resnet18"}:
        parsed_items = _build_more_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name == "build_paper":
        parsed_items = _build_build_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name == "itaml_paper":
        parsed_items = _build_itaml_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name == "incre_diva":
        parsed_items = _build_diva_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name in {"mfgr_paper", "gfril_paper"}:
        parsed_items = _build_paper_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name == "incre_genclassifier":
        parsed_items = _build_genclassifier_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name in (
        "incre_beef",
        "incre_ewc",
        "incre_icarl",
    ):
        parsed_items = _build_pycil_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items
    elif method_name == "der_paper":
        parsed_items = _build_der_report_items_from_logs(summary)
        if parsed_items:
            report_items = parsed_items

    measured = _load_external_efficiency_json(run_dir)
    if measured:
        efficiency = dict(measured)
        efficiency.setdefault("flops_train_only", efficiency.get("flops_train_total"))
        efficiency.setdefault("flops_eval_total", None)
        efficiency.setdefault("time_train_pipeline_sec", efficiency.get("time_train_total_sec"))
        efficiency.setdefault("time_test_eval_sec", efficiency.get("time_test_total_sec"))
        efficiency.setdefault(
            "note",
            "external_subprocess; measured in baseline via cil_efficiency",
        )
    else:
        efficiency = {
            "params_total": None,
            "params_ever_trained": None,
            "params_never_trained": None,
            "flops_forward_per_image": None,
            "flops_train_only": None,
            "flops_eval_total": None,
            "flops_train_total": None,
            "gpu_peak_mb": None,
            "time_train_pipeline_sec": round(float(total_elapsed), 3),
            "time_test_eval_sec": None,
            "time_train_total_sec": round(float(total_elapsed), 3),
            "time_test_total_sec": None,
            "note": (
                "external_subprocess; missing external_efficiency.json — "
                "check PyCIL trainer cil_efficiency import (PYTHONPATH=baselines/common) and thop"
            ),
        }
        print(
            f"[{method_name}] warning: external_efficiency.json not found under {run_dir}; "
            "FLOPs/GPU/params unavailable (wall time only).",
            flush=True,
        )
    save_efficiency(run_dir, {"seed": seed, "method": method_name, **efficiency})

    return {
        "metrics": metrics,
        "report_section": {"name": method_name, "items": report_items},
        "efficiency": efficiency,
        "run_dir": run_dir,
    }


def _sample_taskid_real_subset(images, labels, ratio: float, seed: int):
    if ratio <= 0.0:
        return images[:0], labels[:0]
    if ratio >= 1.0:
        return images, labels

    rng = np.random.default_rng(seed)
    keep_indices = []
    for cls_id in sorted(labels.unique().tolist()):
        cls_idx = torch.nonzero(labels == cls_id, as_tuple=True)[0].cpu().numpy()
        if cls_idx.size == 0:
            continue
        rng.shuffle(cls_idx)
        keep_count = max(1, int(round(cls_idx.size * ratio)))
        keep_indices.extend(cls_idx[:keep_count].tolist())

    if not keep_indices:
        return images[:0], labels[:0]

    keep_idx = torch.tensor(keep_indices, dtype=torch.long)
    return images[keep_idx], labels[keep_idx]


def run_incremental(root_run_dir, datasets, *, seed: int = 42, multi_seed: bool = False):
    cfg_m = INCREMENTAL_1["model"]
    cfg_t = INCREMENTAL_1["train"]
    d = DATA
    device = torch.device(cfg_t["device"])
    run_dir = create_profile_run_dir(root_run_dir, profile_subdir("incre1", seed, multi_seed))
    save_config(run_dir, {"data": d, "profile": INCREMENTAL_1})
    tlog = TrainLogger(run_dir)

    model = IncrementalMoEResNet(
        backbone_name=cfg_m["backbone"],
        pretrained=cfg_m["pretrained"],
        moe_layers=cfg_m["moe_layers"],
        moe_channels=cfg_m["moe_channels"],
        bottleneck_ratios=cfg_m["bottleneck_ratio"],
        allow_old_expert_reuse=bool(cfg_m.get("allow_old_expert_reuse", False)),
        old_expert_top_k=int(cfg_m.get("old_expert_top_k", 1)),
    ).to(device)

    heads, test_loaders = [], []
    task_generators = {}
    task_train_taskid_images = {}
    task_val_images = {}
    results = []
    router_sessions = []
    task_id_clf = None
    defect_head_imprint_init = bool(COMMON_HEAD.get("imprint_init", True))
    head_type = _resolve_head_type(cfg_t)
    prototype_mu_mode = _resolve_prototype_mu_mode(cfg_t)
    use_oracle_taskid = cfg_t.get("oracle_taskid", False)
    use_generated_replay = (not use_oracle_taskid) and cfg_t["taskid_replay_source"] == "generated"
    use_taskid_classifier = not use_oracle_taskid
    use_generated_current_task = use_generated_replay and cfg_t.get("taskid_use_generated_current_task", True)
    generator_type = cfg_t.get("generator_type", "cvae")
    generator_aux_val_source = cfg_t.get(
        "generator_aux_val_source",
        cfg_t.get("aux_val_source", "split_train"),
    )
    if generator_aux_val_source not in {"split_train", "dataset_val"}:
        raise ValueError(f"unsupported generator_aux_val_source: {generator_aux_val_source}")
    generator_aux_val_ratio = float(
        cfg_t.get("generator_aux_val_ratio", cfg_t.get("aux_val_ratio", 0.4))
    )
    taskid_aux_val_source = cfg_t.get(
        "taskid_aux_val_source",
        cfg_t.get("aux_val_source", "split_train"),
    )
    if taskid_aux_val_source not in {"split_train", "dataset_val"}:
        raise ValueError(f"unsupported taskid_aux_val_source: {taskid_aux_val_source}")
    taskid_aux_val_ratio = float(
        cfg_t.get("taskid_aux_val_ratio", cfg_t.get("aux_val_ratio", 0.4))
    )

    eff = EfficiencyTracker(device, image_size=int(d["image_size"]))
    batch_size = int(cfg_t["batch_size"])
    backbone_train_first_session_if_not_pretrained = bool(
        cfg_t.get("backbone_train_first_session_if_not_pretrained", True)
    )

    for task_id, task_classes in enumerate(d["task_splits"]):
        task_cfg_t = _resolve_incremental1_generator_cfg(
            cfg_t=cfg_t,
            generator_type=generator_type,
            task_classes=task_classes,
            class_names=d["class_names"],
        )
        generator_image_size = _generator_image_size(task_cfg_t, generator_type)
        generated_num_per_class = _generator_num_per_class(task_cfg_t, generator_type)
        loaders = build_task_loaders(datasets, task_classes, cfg_t["batch_size"], d["num_workers"])
        test_loaders.append(loaders["test"])

        if task_id > 0:
            model.freeze_task(task_id - 1)
        model.add_task(task_id, cfg_m["experts_per_task"])
        train_backbone_now = (
            (not bool(cfg_m.get("pretrained", True)))
            and backbone_train_first_session_if_not_pretrained
            and task_id == 0
        )
        model.set_backbone_trainable(train_backbone_now)
        model.to(device)

        head = _build_defect_head(cfg_m, cfg_t, len(task_classes), device)
        if defect_head_imprint_init and hasattr(head, "imprint"):
            model.eval()
            feats_init, labels_init = [], []
            with torch.no_grad():
                for imgs, lbls in loaders["train_eval"]:
                    feats_init.append(model(imgs.to(device), task_id).cpu())
                    labels_init.append(lbls)
            _imprint_or_fail(head, feats_init, labels_init, task_classes, f"incremental task {task_id}")

        with eff.train_timer():
            eff.record_training_modules(model, head)
            best_val, best_ep, model_logs = train_task_experts(
                model, head, loaders["train"], loaders["val"], task_id, task_classes,
                lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"], epochs=cfg_t["epochs_per_task"],
                patience=cfg_t["early_stopping_patience"], min_delta=cfg_t["early_stopping_min_delta"], device=device,
                log_prefix=f"[incre1][session{task_id + 1}][model]",
                train_backbone=train_backbone_now,
                class_internal_loss_cfg=cfg_t.get("class_internal_loss", None),
            )
            if prototype_mu_mode == "post_train_imprint" and hasattr(head, "imprint"):
                _refresh_head_prototypes_from_loader(
                    model,
                    head,
                    loaders["train_eval"],
                    task_id,
                    task_classes,
                    device,
                    f"incremental task {task_id} post_train_imprint",
                )
            eff.record_training_modules(model, head)
            if is_profile_flops_enabled():
                n_epochs_run = max(len(model_logs), 1)
                n_train = count_loader_samples(loaders["train"]) * n_epochs_run
                n_val = count_loader_samples(loaders["val"]) * n_epochs_run
                fwd_batch = measure_incremental1_train_step_flops(
                    model, head, task_id, batch_size, int(d["image_size"]), device
                )
                if fwd_batch > 0:
                    per_sample = max(fwd_batch // max(batch_size, 1), 1)
                    eff.add_train_only_flops(
                        estimate_train_flops_from_steps(per_sample, n_train)
                    )
                    eff.add_eval_flops(
                        estimate_eval_flops_from_steps(per_sample, n_val)
                    )

            need_generator = use_generated_replay or (use_generated_current_task and task_id > 0)
            model_was_offloaded = False

            if need_generator:
                _release_model_train_memory(model)
                if device.type == "cuda":
                    model.to(torch.device("cpu"))
                    model_was_offloaded = True
                    torch.cuda.empty_cache()
                aux_imgs_vae, aux_labels_vae = collect_task_images(loaders["train_eval"], generator_image_size)
                if generator_aux_val_source == "split_train":
                    (train_imgs_vae, train_labels_vae), (val_imgs_vae, val_labels_vae) = _split_aux_images(
                        aux_imgs_vae, aux_labels_vae, generator_aux_val_ratio, d["seed"] + task_id
                    )
                else:
                    train_imgs_vae, train_labels_vae = aux_imgs_vae, aux_labels_vae
                    val_imgs_vae, val_labels_vae = collect_task_images(loaders["val"], generator_image_size)

            if use_taskid_classifier:
                aux_imgs_taskid, aux_labels_taskid = collect_task_images(
                    loaders["train_eval"], cfg_t["taskid_image_size"]
                )
                if taskid_aux_val_source == "split_train":
                    (train_imgs_taskid, train_labels_taskid), (val_imgs, val_labels) = _split_aux_images(
                        aux_imgs_taskid, aux_labels_taskid, taskid_aux_val_ratio, d["seed"] + 100 + task_id
                    )
                else:
                    train_imgs_taskid, train_labels_taskid = aux_imgs_taskid, aux_labels_taskid
                    val_imgs, val_labels = collect_task_images(loaders["val"], cfg_t["taskid_image_size"])
                task_train_taskid_images[task_id] = (train_imgs_taskid, train_labels_taskid)
                task_val_images[task_id] = (val_imgs, val_labels)

            if need_generator:
                try:
                    generator_state = train_generator(
                        generator_type=generator_type,
                        train_images=train_imgs_vae,
                        train_labels=train_labels_vae,
                        val_images=val_imgs_vae,
                        val_labels=val_labels_vae,
                        class_ids=task_classes,
                        class_names=d["class_names"],
                        task_id=task_id,
                        cfg_t=task_cfg_t,
                        run_dir=run_dir,
                        device=device,
                        log_prefix=f"[incre1][session{task_id + 1}][generator]",
                    )
                    generator_state["generated_per_class"] = generated_num_per_class
                    generator_state["batch_size"] = batch_size
                    generator_state["image_size"] = generator_image_size
                    generator_state["num_train_samples"] = int(train_imgs_vae.size(0))
                    generator_state["num_val_samples"] = int(val_imgs_vae.size(0))
                    eff.record_training_modules(generator_state.get("model"))
                    if is_profile_flops_enabled():
                        train_flops, eval_flops = accumulate_generator_flops_split(generator_state, device)
                        eff.add_train_only_flops(train_flops)
                        eff.add_eval_flops(eval_flops)
                    _move_generator_state_to_cpu(generator_state)
                    task_generators[task_id] = generator_state
                    tlog.log_epoch_table("generator", task_id, generator_state.get("epoch_logs", []))
                    plot_loss_curves(
                        run_dir, generator_state.get("epoch_logs", []), "incremental_1", "generator", session=task_id
                    )
                finally:
                    if device.type == "cuda" and model_was_offloaded:
                        model.to(device)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            if task_id > 0 and use_taskid_classifier:
                train_sets = []
                val_sets = []
                for tid in range(task_id + 1):
                    use_generated_for_tid = False
                    if tid < task_id and use_generated_replay:
                        use_generated_for_tid = True
                    if tid == task_id and use_generated_current_task:
                        use_generated_for_tid = True

                    if use_generated_for_tid:
                        tid_num_per_class = int(
                            task_generators[tid].get("generated_per_class", generated_num_per_class)
                        )
                        tid_val_num_per_class = _generated_aux_val_num_per_class(
                            tid_num_per_class,
                            taskid_aux_val_ratio,
                        )
                        _move_generator_state_to_device(task_generators[tid], device)
                        try:
                            gen_imgs, gen_labels = sample_generator_images(
                                task_generators[tid],
                                num_per_class=tid_num_per_class,
                                target_image_size=cfg_t["taskid_image_size"],
                                device=device,
                                save_preview_dir=os.path.join(
                                    root_run_dir,
                                    "false_image",
                                    "incre1",
                                    f"session{task_id + 1}",
                                    f"task{tid + 1}",
                                ),
                                class_names=d["class_names"],
                                max_save_per_class=20,
                                save_seed=d["seed"] + task_id * 100 + tid,
                            )
                            gen_val_imgs, gen_val_labels = sample_generator_images(
                                task_generators[tid],
                                num_per_class=tid_val_num_per_class,
                                target_image_size=cfg_t["taskid_image_size"],
                                device=device,
                                save_seed=d["seed"] + 5000 + task_id * 100 + tid,
                            )
                        finally:
                            _move_generator_state_to_cpu(task_generators[tid])
                        if tid < task_id and use_generated_replay:
                            real_imgs, real_labels = task_train_taskid_images[tid]
                            mix_real_imgs, mix_real_labels = _sample_taskid_real_subset(
                                real_imgs,
                                real_labels,
                                cfg_t.get("taskid_replay_source_generated_real_replay_ratio", 0.05),
                                d["seed"] + 1000 + tid,
                            )
                            if mix_real_imgs.numel() > 0:
                                gen_imgs = torch.cat([gen_imgs, mix_real_imgs], dim=0)
                                gen_labels = torch.cat([gen_labels, mix_real_labels], dim=0)
                        train_sets.append((gen_imgs, tid, gen_labels))
                        val_sets.append((gen_val_imgs, tid, gen_val_labels))
                    else:
                        imgs, labels_real = task_train_taskid_images[tid]
                        use_real_ratio = (
                            cfg_t["taskid_replay_source"] == "real"
                            and (
                                tid < task_id
                                or (
                                    tid == task_id
                                    and cfg_t.get("taskid_replay_source_real_use_current_task_ratio", False)
                                )
                            )
                        )
                        if use_real_ratio:
                            imgs, labels_real = _sample_taskid_real_subset(
                                imgs,
                                labels_real,
                                cfg_t.get("taskid_replay_source_real_ratio", 0.05),
                                d["seed"] + 2000 + tid,
                            )
                        train_sets.append((imgs, tid, labels_real))
                        val_sets.append((task_val_images[tid][0], tid, task_val_images[tid][1]))

                n_epochs = cfg_t["taskid_epochs_initial"] + task_id * cfg_t["taskid_epochs_per_task_add"]
                taskid_init_state = (
                    task_id_clf.state_dict()
                    if task_id_clf is not None and bool(cfg_t.get("taskid_continue_from_prev_session", True))
                    else None
                )
                task_id_clf, taskid_logs = train_task_id_classifier(
                    train_sets=train_sets,
                    val_sets=val_sets,
                    num_tasks=task_id + 1,
                    pretrained=cfg_t["taskid_pretrained"],
                    lr=cfg_t["taskid_lr"],
                    batch_size=cfg_t["taskid_batch_size"],
                    epochs=n_epochs,
                    device=device,
                    early_stopping_patience=cfg_t["taskid_early_stopping_patience"],
                    early_stopping_min_delta=cfg_t["taskid_early_stopping_min_delta"],
                    weight_decay=float(cfg_t.get("taskid_weight_decay", 1e-4)),
                    use_cosine_scheduler=bool(cfg_t.get("taskid_use_cosine_scheduler", True)),
                    log_prefix=f"[incre1][session{task_id + 1}][taskid]",
                    contrastive_cfg=cfg_t.get("taskid_contrastive", None),
                    margin_loss_cfg=cfg_t.get("taskid_margin_loss", None),
                    init_state_dict=taskid_init_state,
                )
                eff.record_training_modules(task_id_clf)
                if is_profile_flops_enabled() and task_id_clf is not None:
                    tid_epochs = max(len(taskid_logs), 1)
                    tid_bs = int(cfg_t["taskid_batch_size"])
                    tid_samples = sum(t[0].size(0) for t in train_sets) * tid_epochs
                    tid_val_samples = sum(v[0].size(0) for v in val_sets) * tid_epochs
                    if tid_samples > 0:
                        x_tid = torch.randn(
                            min(tid_bs, 4), 3, cfg_t["taskid_image_size"], cfg_t["taskid_image_size"]
                        )
                        fwd_tid = profile_module_forward_flops(task_id_clf, (x_tid,), device) or 0
                        per_tid = max(fwd_tid // max(tid_bs, 1), 1)
                        eff.add_train_only_flops(
                            estimate_train_flops_from_steps(per_tid, tid_samples)
                        )
                        eff.add_eval_flops(
                            estimate_eval_flops_from_steps(per_tid, tid_val_samples)
                        )
                tlog.log_epoch_table("taskid", task_id, taskid_logs)
                plot_loss_curves(run_dir, taskid_logs, "incremental_1", "taskid", session=task_id)

        heads.append(head)
        tlog.log_epoch_table("model", task_id, model_logs)
        plot_loss_curves(run_dir, model_logs, "incremental_1", "model", session=task_id)

        with eff.test_timer():
            eval_result = evaluate_all(
                model=model,
                heads=heads,
                task_id_classifier=task_id_clf,
                test_loaders=test_loaders,
                task_splits=d["task_splits"][: task_id + 1],
                class_names=d["class_names"],
                device=device,
                taskid_image_size=cfg_t["taskid_image_size"],
                oracle_taskid=use_oracle_taskid,
                task_router_inference=cfg_t.get("task_router_inference", "top1"),
                task_router_alpha=float(cfg_t.get("task_router_alpha", 0.4)),
                task_router_class_prob_mode=cfg_t.get("task_router_class_prob_mode", "softmax"),
            )
        if is_profile_flops_enabled():
            if task_router_inference == "top1":
                eval_per_image = measure_incremental_top1_forward_per_image(
                    model,
                    heads,
                    task_id_clf if (len(heads) > 1 and not use_oracle_taskid) else None,
                    device,
                    int(d["image_size"]),
                    int(cfg_t["taskid_image_size"]),
                )
            else:
                eval_per_image = measure_incremental1_forward_per_image(
                    model,
                    heads,
                    task_id_clf if (len(heads) > 1 and not use_oracle_taskid) else None,
                    device,
                    int(d["image_size"]),
                    int(cfg_t["taskid_image_size"]),
                    len(heads),
                    use_oracle_taskid,
                )
            eval_samples = sum(count_loader_samples(loader) for loader in test_loaders)
            if eval_per_image:
                eff.add_eval_flops(
                    estimate_eval_flops_from_steps(eval_per_image, eval_samples)
                )
        eval_result["task_id"] = task_id
        results.append(eval_result)
        _save_incremental_session_model(
            run_dir=run_dir,
            session_id=task_id,
            model=model,
            heads=heads,
            task_id_clf=task_id_clf,
            task_splits=d["task_splits"],
        )

        router_sessions.append({
            "session": task_id,
            "num_experts": cfg_m["experts_per_task"],
            "train": collect_incremental_routing_stats(model, loaders["train_eval"], task_id, device),
            "test": collect_incremental_routing_stats(model, loaders["test"], task_id, device),
            "taskid_acc": eval_result.get("taskid_acc"),
            "task_router_dual_rate": eval_result.get("task_router_dual_rate"),
            "task_router_debug": eval_result.get("task_router_debug"),
        })

    save_router_stats(run_dir, "incremental_1", router_sessions)
    save_summary(run_dir, {"profile": "incremental_1", "sessions": results})

    report_items = []
    seen = []
    for idx, metrics in enumerate(results):
        seen.extend(d["task_splits"][idx])
        report_items.append({
            "title": f"session{idx + 1}",
            "metrics": metrics,
            "class_ids": sorted(set(seen)),
        })
    num_seen = len(heads)
    if is_profile_flops_enabled():
        if task_router_inference == "top1":
            eff.set_flops_forward_per_image(
                measure_incremental_top1_forward_per_image(
                    model,
                    heads,
                    task_id_clf if (num_seen > 1 and not use_oracle_taskid) else None,
                    device,
                    int(d["image_size"]),
                    int(cfg_t["taskid_image_size"]),
                )
            )
        else:
            eff.set_flops_forward_per_image(
                measure_incremental1_forward_per_image(
                    model,
                    heads,
                    task_id_clf if (num_seen > 1 and not use_oracle_taskid) else None,
                    device,
                    int(d["image_size"]),
                    int(cfg_t["taskid_image_size"]),
                    num_seen,
                    use_oracle_taskid,
                )
            )
    param_modules = [model, *heads, task_id_clf, *_all_generator_modules(task_generators)]
    efficiency = eff.to_dict(param_modules)
    save_efficiency(run_dir, {"seed": seed, "method": "incremental_1", **efficiency})

    return {
        "results": results,
        "report_section": {"name": "incremental_1", "items": report_items},
        "efficiency": efficiency,
        "run_dir": run_dir,
    }


def _format_router_alpha_label(alpha: float) -> str:
    text = f"{float(alpha):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _incremental_router_grid_name(profile_name: str, alpha: float, score_mode: str) -> str:
    mode = str(score_mode).strip().lower()
    return f"{profile_name}_alpha{_format_router_alpha_label(alpha)}_score{mode}"


def _incremental_router_grid_test_filename(profile_name: str, alpha: float, score_mode: str) -> str:
    mode = str(score_mode).strip().lower()
    if profile_name in ("incremental_2", "HiDMoA"):
        return f"test_alpha{_format_router_alpha_label(alpha)}_score{mode}.txt"
    suffix = profile_name.replace("incremental_", "incre")
    return f"test_{suffix}_alpha{_format_router_alpha_label(alpha)}_score{mode}.txt"


def _build_incremental_session_report_items(results: List[Dict], task_splits: List[List[int]]) -> List[Dict]:
    report_items = []
    seen = []
    for idx, metrics in enumerate(results):
        seen.extend(task_splits[idx])
        report_items.append({
            "title": f"session{idx + 1}",
            "metrics": metrics,
            "class_ids": sorted(set(seen)),
        })
    return report_items


def _run_incremental_vae_router_profile(
    root_run_dir,
    datasets,
    *,
    seed: int = 42,
    multi_seed: bool = False,
    profile: dict,
    profile_name: str,
    profile_dir: str,
):
    cfg_m = profile["model"]
    cfg_t = profile["train"]
    d = DATA
    device = torch.device(cfg_t["device"])
    log_prefix_name = profile_dir
    run_dir = create_profile_run_dir(root_run_dir, profile_subdir(profile_dir, seed, multi_seed))
    save_config(run_dir, {"data": d, "profile": profile})
    tlog = TrainLogger(run_dir)

    model = IncrementalMoEResNet(
        backbone_name=cfg_m["backbone"],
        pretrained=cfg_m["pretrained"],
        moe_layers=cfg_m["moe_layers"],
        moe_channels=cfg_m["moe_channels"],
        bottleneck_ratios=cfg_m["bottleneck_ratio"],
        allow_old_expert_reuse=bool(cfg_m.get("allow_old_expert_reuse", False)),
        old_expert_top_k=int(cfg_m.get("old_expert_top_k", 1)),
    ).to(device)

    heads, test_loaders = [], []
    val_loaders = []
    results = []
    router_sessions = []
    class_generators = {}
    defect_head_imprint_init = bool(COMMON_HEAD.get("imprint_init", True))
    head_type = _resolve_head_type(cfg_t)
    prototype_mu_mode = _resolve_prototype_mu_mode(cfg_t)
    use_oracle_taskid = cfg_t.get("oracle_taskid", False)
    task_router_inference = cfg_t.get("task_router_inference", "top1")
    task_router_class_prob_mode = cfg_t.get(
        "task_router_class_score_mode",
        cfg_t.get("task_router_class_prob_mode", "raw"),
    )
    task_router_alpha_default = float(cfg_t.get("task_router_alpha", 0.4))
    task_router_alpha_search_enabled = bool(cfg_t.get("task_router_alpha_search_enabled", True))
    task_router_alpha_search_trials = int(cfg_t.get("task_router_alpha_search_trials", 20))
    task_router_alpha_search_seed = int(cfg_t.get("task_router_alpha_search_seed", d["seed"]))
    generator_type = str(cfg_t.get("vae_router_type", "vae"))
    feature_space_router = bool(cfg_t.get("vae_router_use_feature_space", generator_type in {"fvae", "fcvae", "fvqvae"}))
    vae_router_feature_extractor = _build_vae_router_feature_extractor(cfg_t) if feature_space_router else None
    if generator_type in {"fvae", "fcvae", "fvqvae"} and not feature_space_router:
        raise ValueError(f"{generator_type} requires vae_router_use_feature_space=True")
    aux_val_source = cfg_t.get("vae_router_aux_val_source", cfg_t.get("aux_val_source", "split_train"))
    if aux_val_source not in {"split_train", "dataset_val"}:
        raise ValueError(f"unsupported vae_router_aux_val_source: {aux_val_source}")
    aux_val_ratio = float(cfg_t.get("vae_router_aux_val_ratio", cfg_t.get("aux_val_ratio", 0.4)))
    vae_router_grouping = str(cfg_t.get("vae_router_grouping", "class")).strip().lower()
    if vae_router_grouping not in {"class", "task"}:
        raise ValueError(f"unsupported vae_router_grouping: {vae_router_grouping}")
    use_prototype_router = bool(cfg_t.get("vae_router_use_prototype", False))
    vae_router_decision_mode = str(cfg_t.get("vae_router_decision_mode", "score")).strip().lower()
    if vae_router_decision_mode not in {"score", "mlpcls", "genproto"}:
        raise ValueError(f"unsupported vae_router_decision_mode: {vae_router_decision_mode}")
    feature_mlp_router_enabled = vae_router_decision_mode == "mlpcls"
    generated_prototype_router_enabled = vae_router_decision_mode == "genproto"
    if feature_mlp_router_enabled:
        if use_prototype_router:
            raise ValueError("vae_router_decision_mode=mlpcls does not support prototype routing")
        if generator_type != "fvae":
            raise ValueError("vae_router_decision_mode=mlpcls currently requires vae_router_type=fvae")
        if vae_router_grouping != "task":
            raise ValueError("vae_router_decision_mode=mlpcls requires vae_router_grouping=task")
        if not feature_space_router or vae_router_feature_extractor is None:
            raise ValueError("vae_router_decision_mode=mlpcls requires feature-space routing with a frozen feature extractor")
        feature_mlp_taskid_cfg = _fvae_mlp_taskid_cfg(cfg_t)
    else:
        feature_mlp_taskid_cfg = {}
    if generated_prototype_router_enabled:
        if use_prototype_router:
            raise ValueError("vae_router_decision_mode=genproto should not be combined with vae_router_use_prototype=True")
        if generator_type != "fvae":
            raise ValueError("vae_router_decision_mode=genproto currently requires vae_router_type=fvae")
        if vae_router_grouping != "task":
            raise ValueError("vae_router_decision_mode=genproto requires vae_router_grouping=task")
        if not feature_space_router or vae_router_feature_extractor is None:
            raise ValueError("vae_router_decision_mode=genproto requires feature-space routing with a frozen feature extractor")
        generated_prototype_cfg = _fvae_generated_prototype_cfg(cfg_t)
    else:
        generated_prototype_cfg = {}
    vae_router_score_mode = str(cfg_t.get("vae_router_score_mode", "is")).strip().lower()
    if vae_router_score_mode not in {"is", "recon", "latent", "elbo_single", "elbo_single_mu", "elbo_single_sample", "elbo_k"}:
        raise ValueError(f"unsupported vae_router_score_mode: {vae_router_score_mode}")
    if feature_mlp_router_enabled:
        task_router = _FeatureMlpTaskRouter(
            task_count=0,
            feature_extractor=vae_router_feature_extractor,
            use_feature_space=feature_space_router,
        )
    elif generated_prototype_router_enabled:
        task_router = _ClassPrototypeTaskRouter(
            task_count=0,
            aggregation=generated_prototype_cfg.get("aggregation", "mean"),
            feature_extractor=vae_router_feature_extractor,
            use_feature_space=feature_space_router,
            score_metric=generated_prototype_cfg.get("metric", "cosine"),
            use_class_prior=False,
            normalize_features=generated_prototype_cfg.get("normalize_features", True),
            use_ema=False,
            ema_decay=0.99,
        )
    elif use_prototype_router:
        task_router = _ClassPrototypeTaskRouter(
            task_count=0,
            aggregation=cfg_t.get("vae_router_prototype_aggregation", "mean"),
            feature_extractor=vae_router_feature_extractor,
            use_feature_space=feature_space_router,
            score_metric=cfg_t.get("vae_router_prototype_metric", "cosine"),
            use_class_prior=bool(cfg_t.get("vae_router_use_class_prior", False)),
            normalize_features=cfg_t.get("vae_router_prototype_normalize", True),
            use_ema=bool(cfg_t.get("vae_router_prototype_use_ema", False)),
            ema_decay=cfg_t.get("vae_router_prototype_ema_decay", 0.99),
        )
    else:
        task_router = _ClassVaeTaskRouter(
            task_count=0,
            eval_importance_samples=cfg_t.get("vae_router_eval_importance_samples", 200),
            aggregation=cfg_t.get("vae_router_aggregation", "logsumexp"),
            use_class_prior=bool(cfg_t.get("vae_router_use_class_prior", False)),
            feature_extractor=vae_router_feature_extractor,
            use_feature_space=feature_space_router,
            score_mode=vae_router_score_mode,
        )

    eff = EfficiencyTracker(device, image_size=int(d["image_size"]))
    batch_size = int(cfg_t["batch_size"])
    backbone_train_first_session_if_not_pretrained = bool(
        cfg_t.get("backbone_train_first_session_if_not_pretrained", True)
    )

    for task_id, task_classes in enumerate(d["task_splits"]):
        task_cfg_t = _resolve_incremental1_generator_cfg(
            cfg_t=cfg_t,
            generator_type=generator_type,
            task_classes=task_classes,
            class_names=d["class_names"],
        )
        generator_image_size = _generator_image_size(task_cfg_t, generator_type)
        generated_num_per_class = _generator_num_per_class(task_cfg_t, generator_type)
        loaders = build_task_loaders(datasets, task_classes, cfg_t["batch_size"], d["num_workers"])
        test_loaders.append(loaders["test"])
        val_loaders.append(loaders["val"])
        if generator_type not in {"vae", "cvae", "vqvae", "fvae", "fcvae", "fvqvae"}:
            raise ValueError(
                f"{profile_name} only supports vae_router_type=vae/cvae/vqvae/fvae/fcvae/fvqvae; got {generator_type}"
            )

        if task_id > 0:
            model.freeze_task(task_id - 1)
        model.add_task(task_id, cfg_m["experts_per_task"])
        train_backbone_now = (
            (not bool(cfg_m.get("pretrained", True)))
            and backbone_train_first_session_if_not_pretrained
            and task_id == 0
        )
        model.set_backbone_trainable(train_backbone_now)
        model.to(device)

        head = _build_defect_head(cfg_m, cfg_t, len(task_classes), device)
        if defect_head_imprint_init and hasattr(head, "imprint"):
            model.eval()
            feats_init, labels_init = [], []
            with torch.no_grad():
                for imgs, lbls in loaders["train_eval"]:
                    feats_init.append(model(imgs.to(device), task_id).cpu())
                    labels_init.append(lbls)
            _imprint_or_fail(head, feats_init, labels_init, task_classes, f"incremental task {task_id}")

        with eff.train_timer():
            eff.record_training_modules(model, head)
            best_val, best_ep, model_logs = train_task_experts(
                model, head, loaders["train"], loaders["val"], task_id, task_classes,
                lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"], epochs=cfg_t["epochs_per_task"],
                patience=cfg_t["early_stopping_patience"], min_delta=cfg_t["early_stopping_min_delta"], device=device,
                log_prefix=f"[{log_prefix_name}][session{task_id + 1}][model]",
                train_backbone=train_backbone_now,
                class_internal_loss_cfg=cfg_t.get("class_internal_loss", None),
            )
            if prototype_mu_mode == "post_train_imprint" and hasattr(head, "imprint"):
                _refresh_head_prototypes_from_loader(
                    model,
                    head,
                    loaders["train_eval"],
                    task_id,
                    task_classes,
                    device,
                    f"incremental task {task_id} post_train_imprint",
                )
            eff.record_training_modules(model, head)
            if is_profile_flops_enabled():
                n_epochs_run = max(len(model_logs), 1)
                n_train = count_loader_samples(loaders["train"]) * n_epochs_run
                n_val = count_loader_samples(loaders["val"]) * n_epochs_run
                fwd_batch = measure_incremental1_train_step_flops(
                    model, head, task_id, batch_size, int(d["image_size"]), device
                )
                if fwd_batch > 0:
                    per_sample = max(fwd_batch // max(batch_size, 1), 1)
                    eff.add_train_only_flops(
                        estimate_train_flops_from_steps(per_sample, n_train)
                    )
                    eff.add_eval_flops(
                        estimate_eval_flops_from_steps(per_sample, n_val)
                    )

            aux_imgs_vae, aux_labels_vae = collect_task_images(loaders["train_eval"], generator_image_size)
            if aux_val_source == "split_train":
                (train_imgs_vae, train_labels_vae), (val_imgs_vae, val_labels_vae) = _split_aux_images(
                    aux_imgs_vae, aux_labels_vae, aux_val_ratio, d["seed"] + task_id
                )
            else:
                train_imgs_vae, train_labels_vae = aux_imgs_vae, aux_labels_vae
                val_imgs_vae, val_labels_vae = collect_task_images(loaders["val"], generator_image_size)

            model_was_offloaded = False
            if device.type == "cuda":
                model.to(torch.device("cpu"))
                model_was_offloaded = True
                task_router.to(torch.device("cpu"))
                torch.cuda.empty_cache()

            if vae_router_grouping == "task":
                task_groups = [("task", int(task_id), train_imgs_vae, train_labels_vae, val_imgs_vae, val_labels_vae)]
            else:
                task_groups = []
                for cls_id in task_classes:
                    cls_mask_train = train_labels_vae == cls_id
                    cls_mask_val = val_labels_vae == cls_id
                    if cls_mask_train.sum() <= 0:
                        continue
                    cls_train_imgs = train_imgs_vae[cls_mask_train]
                    cls_train_labels = train_labels_vae[cls_mask_train]
                    if cls_mask_val.sum() > 0:
                        cls_val_imgs = val_imgs_vae[cls_mask_val]
                        cls_val_labels = val_labels_vae[cls_mask_val]
                    else:
                        cls_val_imgs = cls_train_imgs[:1]
                        cls_val_labels = cls_train_labels[:1]
                    task_groups.append(("class", int(cls_id), cls_train_imgs, cls_train_labels, cls_val_imgs, cls_val_labels))

            current_task_train_features = None
            current_task_val_features = None
            current_task_input_dim = 0
            for group_kind, group_id, cls_train_imgs, cls_train_labels, cls_val_imgs, cls_val_labels in task_groups:
                if cls_train_imgs.numel() <= 0:
                    continue
                group_name = f"{group_kind}_{group_id}"
                if use_prototype_router:
                    if feature_space_router:
                        if vae_router_feature_extractor is None:
                            raise ValueError("prototype routing requires a feature extractor when vae_router_use_feature_space=True")
                        fvae_cfg = _fvae_cfg(task_cfg_t)
                        router_feature_train_per = 0
                        if is_profile_flops_enabled():
                            router_feature_train_per = _profile_router_feature_per_sample(
                                vae_router_feature_extractor,
                                cls_train_imgs,
                                device,
                                batch_size=int(fvae_cfg.get("fvae_feature_batch", 64)),
                            )
                        cls_train_features = _extract_router_features(
                            vae_router_feature_extractor,
                            cls_train_imgs,
                            device,
                            batch_size=int(fvae_cfg.get("fvae_feature_batch", 64)),
                        )
                        if router_feature_train_per > 0:
                            eff.add_train_only_flops(
                                estimate_eval_flops_from_steps(
                                    router_feature_train_per,
                                    int(cls_train_imgs.size(0)),
                                )
                            )
                    else:
                        cls_train_features = cls_train_imgs.reshape(cls_train_imgs.size(0), -1).to(device=device, dtype=torch.float32)

                    if group_kind == "task":
                        task_router.add_class_prototype(
                            class_id=int(group_id),
                            task_id=task_id,
                            prototype=cls_train_features.mean(dim=0),
                            class_prior=float(cls_train_features.size(0)),
                            class_count=int(cls_train_features.size(0)),
                        )
                    else:
                        for cls_id in sorted(int(v) for v in cls_train_labels.unique().tolist()):
                            cls_mask = cls_train_labels == cls_id
                            if not cls_mask.any():
                                continue
                            cls_proto = cls_train_features[cls_mask].mean(dim=0)
                            task_router.add_class_prototype(
                                class_id=int(cls_id),
                                task_id=task_id,
                                prototype=cls_proto,
                                class_prior=float(int(cls_mask.sum().item())),
                                class_count=int(cls_mask.sum().item()),
                            )
                    class_logs = []
                else:
                    if generator_type in {"fvae", "fcvae"}:
                        fvae_cfg = _fvae_cfg(task_cfg_t) if generator_type == "fvae" else _fcvae_cfg(task_cfg_t)
                        router_feature_train_per = 0
                        router_feature_val_per = 0
                        if is_profile_flops_enabled():
                            router_feature_train_per = _profile_router_feature_per_sample(
                                vae_router_feature_extractor,
                                cls_train_imgs,
                                device,
                                batch_size=fvae_cfg["fvae_feature_batch"],
                            )
                            router_feature_val_per = _profile_router_feature_per_sample(
                                vae_router_feature_extractor,
                                cls_val_imgs,
                                device,
                                batch_size=fvae_cfg["fvae_feature_batch"],
                            )
                        cls_train_features = _extract_router_features(
                            vae_router_feature_extractor,
                            cls_train_imgs,
                            device,
                            batch_size=fvae_cfg["fvae_feature_batch"],
                        )
                        cls_val_features = _extract_router_features(
                            vae_router_feature_extractor,
                            cls_val_imgs,
                            device,
                            batch_size=fvae_cfg["fvae_feature_batch"],
                        )
                        if feature_mlp_router_enabled and group_kind == "task":
                            current_task_train_features = cls_train_features.detach().cpu()
                            current_task_val_features = cls_val_features.detach().cpu()
                            current_task_input_dim = int(cls_train_features.shape[1])
                        if router_feature_train_per > 0:
                            eff.add_train_only_flops(
                                estimate_eval_flops_from_steps(
                                    router_feature_train_per,
                                    int(cls_train_imgs.size(0)),
                                )
                            )
                        if router_feature_val_per > 0:
                            eff.add_eval_flops(
                                estimate_eval_flops_from_steps(
                                    router_feature_val_per,
                                    int(cls_val_imgs.size(0)),
                                )
                            )
                        class_ids_for_vae = task_classes if group_kind == "task" else [int(group_id)]
                        if generator_type == "fvae":
                            generator_state = train_feature_vae(
                                train_features=cls_train_features,
                                train_labels=cls_train_labels,
                                val_features=cls_val_features,
                                val_labels=cls_val_labels,
                                class_ids=class_ids_for_vae,
                                input_dim=int(cls_train_features.shape[1]),
                                h_dim=fvae_cfg["h_dim"],
                                z_dim=fvae_cfg["z_dim"],
                                epochs=fvae_cfg["epochs"],
                                early_stopping_patience=fvae_cfg["early_stopping_patience"],
                                early_stopping_min_delta=fvae_cfg["early_stopping_min_delta"],
                                lr=fvae_cfg["lr"],
                                weight_decay=fvae_cfg["weight_decay"],
                                beta_kl=fvae_cfg["beta_kl"],
                                kl_warmup_epochs=fvae_cfg["kl_warmup_epochs"],
                                recon_weight=fvae_cfg["recon_weight"],
                                batch_size=fvae_cfg["batch_size"],
                                device=device,
                                log_prefix=f"[{log_prefix_name}][session{task_id + 1}][{group_name}]",
                                model_name="fVAE",
                            )
                        else:
                            generator_state = train_feature_cvae(
                                train_features=cls_train_features,
                                train_labels=cls_train_labels,
                                val_features=cls_val_features,
                                val_labels=cls_val_labels,
                                class_ids=class_ids_for_vae,
                                input_dim=int(cls_train_features.shape[1]),
                                h_dim=fvae_cfg["h_dim"],
                                z_dim=fvae_cfg["z_dim"],
                                epochs=fvae_cfg["epochs"],
                                early_stopping_patience=fvae_cfg["early_stopping_patience"],
                                early_stopping_min_delta=fvae_cfg["early_stopping_min_delta"],
                                lr=fvae_cfg["lr"],
                                weight_decay=fvae_cfg["weight_decay"],
                                beta_kl=fvae_cfg["beta_kl"],
                                kl_warmup_epochs=fvae_cfg["kl_warmup_epochs"],
                                recon_weight=fvae_cfg["recon_weight"],
                                latent_pool_noise_std=float(fvae_cfg.get("latent_pool_noise_std", 0.0)),
                                batch_size=fvae_cfg["batch_size"],
                                device=device,
                                log_prefix=f"[{log_prefix_name}][session{task_id + 1}][{group_name}]",
                                model_name="fCVAE",
                            )
                    elif generator_type == "fvqvae":
                        fvqvae_cfg = _fvqvae_cfg(task_cfg_t)
                        router_feature_train_per = 0
                        router_feature_val_per = 0
                        if is_profile_flops_enabled():
                            router_feature_train_per = _profile_router_feature_per_sample(
                                vae_router_feature_extractor,
                                cls_train_imgs,
                                device,
                                batch_size=int(fvqvae_cfg["fvae_feature_batch"]),
                            )
                            router_feature_val_per = _profile_router_feature_per_sample(
                                vae_router_feature_extractor,
                                cls_val_imgs,
                                device,
                                batch_size=int(fvqvae_cfg["fvae_feature_batch"]),
                            )
                        cls_train_features = _extract_router_features(
                            vae_router_feature_extractor,
                            cls_train_imgs,
                            device,
                            batch_size=int(fvqvae_cfg["fvae_feature_batch"]),
                        )
                        cls_val_features = _extract_router_features(
                            vae_router_feature_extractor,
                            cls_val_imgs,
                            device,
                            batch_size=int(fvqvae_cfg["fvae_feature_batch"]),
                        )
                        if router_feature_train_per > 0:
                            eff.add_train_only_flops(
                                estimate_eval_flops_from_steps(
                                    router_feature_train_per,
                                    int(cls_train_imgs.size(0)),
                                )
                            )
                        if router_feature_val_per > 0:
                            eff.add_eval_flops(
                                estimate_eval_flops_from_steps(
                                    router_feature_val_per,
                                    int(cls_val_imgs.size(0)),
                                )
                            )
                        class_ids_for_vae = task_classes if group_kind == "task" else [int(group_id)]
                        generator_state = train_feature_vqvae(
                            train_features=cls_train_features,
                            train_labels=cls_train_labels,
                            val_features=cls_val_features,
                            val_labels=cls_val_labels,
                            class_ids=class_ids_for_vae,
                            input_dim=int(cls_train_features.shape[1]),
                            h_dim=fvqvae_cfg["h_dim"],
                            embedding_dim=fvqvae_cfg["embedding_dim"],
                            num_embeddings=fvqvae_cfg["num_embeddings"],
                            commitment_cost=fvqvae_cfg["commitment_cost"],
                            codebook_weight=fvqvae_cfg["codebook_weight"],
                            ema_decay=fvqvae_cfg["ema_decay"],
                            epochs=fvqvae_cfg["epochs"],
                            early_stopping_patience=fvqvae_cfg["early_stopping_patience"],
                            early_stopping_min_delta=fvqvae_cfg["early_stopping_min_delta"],
                            lr=fvqvae_cfg["lr"],
                            weight_decay=fvqvae_cfg["weight_decay"],
                            recon_weight=fvqvae_cfg["recon_weight"],
                            batch_size=fvqvae_cfg["batch_size"],
                            device=device,
                            log_prefix=f"[{log_prefix_name}][session{task_id + 1}][{group_name}]",
                            model_name="fVQ-VAE",
                        )
                    else:
                        generator_state = train_generator(
                            generator_type=generator_type,
                            train_images=cls_train_imgs,
                            train_labels=cls_train_labels,
                            val_images=cls_val_imgs,
                            val_labels=cls_val_labels,
                            class_ids=(task_classes if group_kind == "task" else [int(group_id)]),
                            class_names=d["class_names"],
                            task_id=task_id,
                            cfg_t=task_cfg_t,
                            run_dir=run_dir,
                            device=device,
                            log_prefix=f"[{log_prefix_name}][session{task_id + 1}][{group_name}]",
                        )
                    generator_state["generated_per_class"] = generated_num_per_class
                    generator_state["batch_size"] = batch_size
                    generator_state["image_size"] = generator_image_size
                    generator_state["num_train_samples"] = int(cls_train_imgs.size(0))
                    generator_state["num_val_samples"] = int(cls_val_imgs.size(0))
                    class_generators[group_id] = generator_state
                    if not feature_mlp_router_enabled and not generated_prototype_router_enabled:
                        task_router.add_class_vae(
                            class_id=int(group_id),
                            task_id=task_id,
                            vae_state=generator_state,
                            class_prior=float(cls_train_imgs.size(0)),
                        )
                    if generated_prototype_router_enabled:
                        proto = _build_generated_task_prototype(
                            generator_state=generator_state,
                            num_samples=int(generated_prototype_cfg["num_samples_per_task"]),
                            device=device,
                        )
                        task_router.add_class_prototype(
                            class_id=int(group_id),
                            task_id=task_id,
                            prototype=proto,
                            class_prior=1.0,
                            class_count=int(generated_prototype_cfg["num_samples_per_task"]),
                        )
                    eff.record_training_modules(generator_state.get("model"))
                    if is_profile_flops_enabled():
                        train_flops, eval_flops = accumulate_generator_flops_split(generator_state, device)
                        eff.add_train_only_flops(train_flops)
                        eff.add_eval_flops(eval_flops)
                    class_logs = generator_state.get("epoch_logs", [])
                tlog.log_epoch_table("class_vae", task_id, class_logs)
                plot_loss_curves(
                    run_dir,
                    class_logs,
                    profile_name,
                    f"{group_kind}_vae_{group_id}",
                    session=task_id,
                )

            if feature_mlp_router_enabled:
                taskid_train_sets = []
                taskid_val_sets = []
                for old_task_id in range(task_id):
                    old_state = class_generators.get(old_task_id, None)
                    if old_state is None:
                        continue
                    old_task_class_count = max(1, len(d["task_splits"][old_task_id]))
                    replay_train_count = int(feature_mlp_taskid_cfg["generated_per_old_task"])
                    if replay_train_count <= 0:
                        replay_train_count = int(old_state.get("generated_per_class", generated_num_per_class)) * old_task_class_count
                    replay_val_count = int(feature_mlp_taskid_cfg["generated_val_per_old_task"])
                    if replay_val_count <= 0:
                        replay_val_count = max(
                            1,
                            int(round(replay_train_count * float(feature_mlp_taskid_cfg["generated_val_ratio"]))),
                        )
                    taskid_train_sets.append((
                        _sample_fvae_task_replay_features(old_state, replay_train_count, device),
                        old_task_id,
                    ))
                    taskid_val_sets.append((
                        _sample_fvae_task_replay_features(old_state, replay_val_count, device),
                        old_task_id,
                    ))

                if bool(feature_mlp_taskid_cfg.get("current_task_use_generated", False)):
                    current_state = class_generators.get(task_id, None)
                    if current_state is None:
                        raise ValueError("FVAE + MLP task-id requires the current task fVAE state before generated replay")
                    current_task_train_count = int(feature_mlp_taskid_cfg.get("generated_per_current_task", 0))
                    if current_task_train_count <= 0:
                        current_task_train_count = int(feature_mlp_taskid_cfg.get("generated_per_old_task", 0))
                    if current_task_train_count <= 0:
                        current_task_train_count = 200
                    current_task_val_count = int(feature_mlp_taskid_cfg.get("generated_val_per_current_task", 0))
                    if current_task_val_count <= 0:
                        current_task_val_count = max(
                            1,
                            int(round(current_task_train_count * float(feature_mlp_taskid_cfg["generated_val_ratio"]))),
                        )
                    current_task_train_source = _sample_fvae_task_replay_features(
                        current_state,
                        current_task_train_count,
                        device,
                    )
                    current_task_val_source = _sample_fvae_task_replay_features(
                        current_state,
                        current_task_val_count,
                        device,
                    )
                else:
                    if current_task_train_features is None or current_task_train_features.numel() <= 0:
                        raise ValueError("FVAE + MLP task-id requires current task train features")
                    if current_task_val_features is None or current_task_val_features.numel() <= 0:
                        current_task_val_features = current_task_train_features[:1].clone()
                    current_task_train_source = current_task_train_features
                    current_task_val_source = current_task_val_features

                taskid_train_sets.append((current_task_train_source, task_id))
                taskid_val_sets.append((current_task_val_source, task_id))
                prev_taskid_state = None
                if (
                    getattr(task_router, "feature_classifier", None) is not None
                    and bool(feature_mlp_taskid_cfg.get("continue_from_prev_session", True))
                ):
                    prev_taskid_state = deepcopy(task_router.feature_classifier.state_dict())

                feature_taskid_clf, feature_taskid_logs = train_feature_task_id_classifier(
                    train_sets=taskid_train_sets,
                    val_sets=taskid_val_sets,
                    num_tasks=task_id + 1,
                    input_dim=int(current_task_input_dim),
                    hidden_dim=int(feature_mlp_taskid_cfg["hidden_dim"]),
                    hidden_layers=int(feature_mlp_taskid_cfg["hidden_layers"]),
                    dropout=float(feature_mlp_taskid_cfg["dropout"]),
                    lr=float(feature_mlp_taskid_cfg["lr"]),
                    batch_size=int(feature_mlp_taskid_cfg["batch_size"]),
                    epochs=int(feature_mlp_taskid_cfg["epochs"]),
                    device=device,
                    early_stopping_patience=feature_mlp_taskid_cfg["early_stopping_patience"],
                    early_stopping_min_delta=float(feature_mlp_taskid_cfg["early_stopping_min_delta"]),
                    weight_decay=float(feature_mlp_taskid_cfg["weight_decay"]),
                    use_cosine_scheduler=bool(feature_mlp_taskid_cfg["use_cosine_scheduler"]),
                    log_prefix=f"[{log_prefix_name}][session{task_id + 1}][taskid_mlp]",
                    init_state_dict=prev_taskid_state,
                )
                task_router.set_feature_classifier(feature_taskid_clf)
                tlog.log_epoch_table("taskid_mlp", task_id, feature_taskid_logs)
                plot_loss_curves(run_dir, feature_taskid_logs, profile_name, "taskid_mlp", session=task_id)
                eff.record_training_modules(feature_taskid_clf)
                if is_profile_flops_enabled() and taskid_train_sets:
                    probe_source = next((feats for feats, _tid in taskid_train_sets if int(feats.size(0)) > 0), None)
                    if probe_source is not None:
                        probe_bs = max(1, min(int(feature_mlp_taskid_cfg["batch_size"]), int(probe_source.size(0))))
                        probe_feats = probe_source[:probe_bs].to(device)
                        fwd_tid = profile_module_forward_flops(feature_taskid_clf, (probe_feats,), device) or 0
                        if fwd_tid > 0:
                            per_sample = max(int(fwd_tid) // max(probe_bs, 1), 1)
                            n_epochs_run = max(len(feature_taskid_logs), 1)
                            n_train = sum(int(feats.size(0)) for feats, _tid in taskid_train_sets) * n_epochs_run
                            n_val = sum(int(feats.size(0)) for feats, _tid in taskid_val_sets) * n_epochs_run
                            eff.add_train_only_flops(
                                estimate_train_flops_from_steps(per_sample, n_train)
                            )
                            eff.add_eval_flops(
                                estimate_eval_flops_from_steps(per_sample, n_val)
                            )

            if model_was_offloaded:
                model.to(device)
                torch.cuda.empty_cache()

        heads.append(head)
        tlog.log_epoch_table("model", task_id, model_logs)
        plot_loss_curves(run_dir, model_logs, profile_name, "model", session=task_id)

        with eff.test_timer():
            eval_result = evaluate_all(
                model=model,
                heads=heads,
                task_id_classifier=task_router,
                test_loaders=test_loaders,
                task_splits=d["task_splits"][: task_id + 1],
                class_names=d["class_names"],
                device=device,
                taskid_image_size=cfg_t["taskid_image_size"],
                oracle_taskid=use_oracle_taskid,
                task_router_inference=task_router_inference,
                task_router_alpha=task_router_alpha_default if get_incremental_2_alpha_cache() is None else float(get_incremental_2_alpha_cache()),
                task_router_class_prob_mode=task_router_class_prob_mode,
            )
        if is_profile_flops_enabled():
            if task_router_inference == "top1":
                eval_per_image = measure_incremental_top1_forward_per_image(
                    model,
                    heads,
                    task_router if task_router._class_entries else None,
                    device,
                    int(d["image_size"]),
                    int(cfg_t["taskid_image_size"]),
                )
            else:
                eval_per_image = measure_incremental1_forward_per_image(
                    model,
                    heads,
                    task_router if task_router._class_entries else None,
                    device,
                    int(d["image_size"]),
                    int(cfg_t["taskid_image_size"]),
                    len(heads),
                    use_oracle_taskid,
                )
            eval_samples = sum(count_loader_samples(loader) for loader in test_loaders)
            if eval_per_image:
                eff.add_eval_flops(
                    estimate_eval_flops_from_steps(eval_per_image, eval_samples)
                )
        eval_result["task_id"] = task_id
        results.append(eval_result)
        _save_incremental_session_model(
            run_dir=run_dir,
            session_id=task_id,
            model=model,
            heads=heads,
            task_id_clf=task_router if task_router._class_entries else None,
            task_splits=d["task_splits"],
        )
        router_sessions.append({
            "session": task_id,
            "num_experts": cfg_m["experts_per_task"],
            "train": collect_incremental_routing_stats(model, loaders["train_eval"], task_id, device),
            "test": collect_incremental_routing_stats(model, loaders["test"], task_id, device),
            "taskid_acc": eval_result.get("taskid_acc"),
            "task_router_dual_rate": eval_result.get("task_router_dual_rate"),
            "task_router_debug": eval_result.get("task_router_debug"),
        })

    cached_alpha = get_incremental_2_alpha_cache()
    selected_alpha = float(cached_alpha) if cached_alpha is not None else task_router_alpha_default
    best_alpha = None
    best_alpha_loss = None
    if (
        task_router_alpha_search_enabled
        and task_router_inference == "top2"
        and len(heads) > 1
        and cached_alpha is None
    ):
        best_alpha, best_alpha_loss = _search_incremental_2_task_router_alpha(
            model=model,
            heads=heads,
            task_id_classifier=task_router,
            val_loaders=[val_loaders[-1]],
            task_splits=d["task_splits"][: len(heads)],
            eval_task_ids=[len(heads) - 1],
            device=device,
            taskid_image_size=cfg_t["taskid_image_size"],
            task_router_inference=task_router_inference,
            class_prob_mode=task_router_class_prob_mode,
            default_alpha=task_router_alpha_default,
            search_trials=task_router_alpha_search_trials,
            search_seed=task_router_alpha_search_seed,
        )
        _set_incremental_2_alpha_cache(best_alpha)
        selected_alpha = float(best_alpha)
        print(
            f"[{log_prefix_name}] Optuna-style alpha search finished: best_alpha={selected_alpha:.4f}, "
            f"val_loss={best_alpha_loss:.4f}, trials={task_router_alpha_search_trials}"
        )
        # Re-evaluate all sessions with the searched alpha in the same seed run,
        # then overwrite old metrics/log stats produced with the default alpha.
        print(
            f"[{log_prefix_name}] Re-evaluating all sessions with searched alpha={selected_alpha:.4f} "
            "(overwrite previous eval results)"
        )
        for sid in range(len(heads)):
            with eff.test_timer():
                eval_result = evaluate_all(
                    model=model,
                    heads=heads[: sid + 1],
                    task_id_classifier=task_router,
                    test_loaders=test_loaders[: sid + 1],
                    task_splits=d["task_splits"][: sid + 1],
                    class_names=d["class_names"],
                    device=device,
                    taskid_image_size=cfg_t["taskid_image_size"],
                    oracle_taskid=use_oracle_taskid,
                    task_router_inference=task_router_inference,
                    task_router_alpha=selected_alpha,
                    task_router_class_prob_mode=task_router_class_prob_mode,
                )
            if is_profile_flops_enabled():
                if task_router_inference == "top1":
                    eval_per_image = measure_incremental_top1_forward_per_image(
                        model,
                        heads[: sid + 1],
                        task_router if task_router._class_entries else None,
                        device,
                        int(d["image_size"]),
                        int(cfg_t["taskid_image_size"]),
                    )
                else:
                    eval_per_image = measure_incremental1_forward_per_image(
                        model,
                        heads[: sid + 1],
                        task_router if task_router._class_entries else None,
                        device,
                        int(d["image_size"]),
                        int(cfg_t["taskid_image_size"]),
                        sid + 1,
                        use_oracle_taskid,
                    )
                eval_samples = sum(count_loader_samples(loader) for loader in test_loaders[: sid + 1])
                if eval_per_image:
                    eff.add_eval_flops(
                        estimate_eval_flops_from_steps(eval_per_image, eval_samples)
                    )
            eval_result["task_id"] = sid
            results[sid] = eval_result
            router_sessions[sid]["taskid_acc"] = eval_result.get("taskid_acc")
            router_sessions[sid]["task_router_dual_rate"] = eval_result.get("task_router_dual_rate")
            router_sessions[sid]["task_router_debug"] = eval_result.get("task_router_debug")

    extra_report_sections: List[Dict] = []
    eval_grid_meta: List[Dict] = []
    if (
        bool(cfg_t.get("task_router_eval_grid_enabled", False))
        and len(heads) > 1
        and not use_oracle_taskid
    ):
        alpha_values = [float(v) for v in cfg_t.get("task_router_eval_alpha_values", [])]
        score_modes = [str(v).strip().lower() for v in cfg_t.get("task_router_eval_class_score_modes", [])]
        grid_inference = str(cfg_t.get("task_router_eval_grid_inference", "top2")).strip().lower()
        print(
            f"[{log_prefix_name}] Running eval grid: inference={grid_inference}, "
            f"alphas={alpha_values}, score_modes={score_modes}"
        )
        for alpha in alpha_values:
            for score_mode in score_modes:
                combo_results = []
                print(f"[{log_prefix_name}] eval grid alpha={alpha:.4g} score_mode={score_mode}")
                for sid in range(len(heads)):
                    eval_result = evaluate_all(
                        model=model,
                        heads=heads[: sid + 1],
                        task_id_classifier=task_router,
                        test_loaders=test_loaders[: sid + 1],
                        task_splits=d["task_splits"][: sid + 1],
                        class_names=d["class_names"],
                        device=device,
                        taskid_image_size=cfg_t["taskid_image_size"],
                        oracle_taskid=use_oracle_taskid,
                        task_router_inference=grid_inference,
                        task_router_alpha=float(alpha),
                        task_router_class_prob_mode=score_mode,
                    )
                    if is_profile_flops_enabled():
                        if grid_inference == "top1":
                            eval_per_image = measure_incremental_top1_forward_per_image(
                                model,
                                heads[: sid + 1],
                                task_router if task_router._class_entries else None,
                                device,
                                int(d["image_size"]),
                                int(cfg_t["taskid_image_size"]),
                            )
                        else:
                            eval_per_image = measure_incremental1_forward_per_image(
                                model,
                                heads[: sid + 1],
                                task_router if task_router._class_entries else None,
                                device,
                                int(d["image_size"]),
                                int(cfg_t["taskid_image_size"]),
                                sid + 1,
                                use_oracle_taskid,
                            )
                        eval_samples = sum(count_loader_samples(loader) for loader in test_loaders[: sid + 1])
                        if eval_per_image:
                            eff.add_eval_flops(
                                estimate_eval_flops_from_steps(eval_per_image, eval_samples)
                            )
                    eval_result["task_id"] = sid
                    eval_result["task_router_eval_alpha"] = float(alpha)
                    eval_result["task_router_eval_class_score_mode"] = score_mode
                    eval_result["task_router_eval_inference"] = grid_inference
                    combo_results.append(eval_result)

                section_name = _incremental_router_grid_name(profile_name, alpha, score_mode)
                extra_report_sections.append({
                    "filename": _incremental_router_grid_test_filename(profile_name, alpha, score_mode),
                    "section": {
                        "name": section_name,
                        "items": _build_incremental_session_report_items(combo_results, d["task_splits"]),
                    },
                })
                final_metrics = combo_results[-1] if combo_results else {}
                eval_grid_meta.append({
                    "section": section_name,
                    "filename": _incremental_router_grid_test_filename(profile_name, alpha, score_mode),
                    "inference": grid_inference,
                    "alpha": float(alpha),
                    "class_score_mode": score_mode,
                    "final_micro_acc": final_metrics.get("micro_acc"),
                    "final_macro_acc": final_metrics.get("macro_acc"),
                    "final_macro_f1": final_metrics.get("macro_f1"),
                    "final_taskid_acc": final_metrics.get("taskid_acc"),
                    "final_dual_route_rate": final_metrics.get("task_router_dual_rate"),
                })

    task_score_heatmaps = []
    if (
        bool(cfg_t.get("task_router_score_heatmap_enabled", False))
        and len(heads) > 1
        and not use_oracle_taskid
    ):
        heatmap_modes = [
            str(v).strip().lower()
            for v in cfg_t.get("task_router_score_heatmap_modes", ["is", "recon", "elbo_k"])
        ]
        task_score_heatmaps = _dump_task_router_score_heatmaps(
            run_dir=run_dir,
            task_router=task_router,
            test_loaders=test_loaders,
            num_seen=len(heads),
            device=device,
            modes=heatmap_modes,
            seed=int(cfg_t.get("task_router_score_heatmap_seed", d["seed"])),
        )

    save_router_stats(run_dir, profile_name, router_sessions)
    save_summary(
        run_dir,
        {
            "profile": profile_name,
            "sessions": results,
            "task_router_alpha": selected_alpha,
            "task_router_alpha_search_enabled": task_router_alpha_search_enabled,
            "task_router_alpha_search_trials": task_router_alpha_search_trials,
            "task_router_alpha_search_best": best_alpha,
            "task_router_alpha_search_best_val_loss": best_alpha_loss,
            "task_router_eval_grid": eval_grid_meta,
            "task_router_score_heatmaps": task_score_heatmaps,
        },
    )

    report_items = _build_incremental_session_report_items(results, d["task_splits"])
    num_seen = len(heads)
    if is_profile_flops_enabled():
        if task_router_inference == "top1":
            eff.set_flops_forward_per_image(
                measure_incremental_top1_forward_per_image(
                    model,
                    heads,
                    task_router if task_router._class_entries else None,
                    device,
                    int(d["image_size"]),
                    int(cfg_t["taskid_image_size"]),
                )
            )
        else:
            eff.set_flops_forward_per_image(
                measure_incremental1_forward_per_image(
                    model,
                    heads,
                    task_router if task_router._class_entries else None,
                    device,
                    int(d["image_size"]),
                    int(cfg_t["taskid_image_size"]),
                    num_seen,
                    use_oracle_taskid,
                )
            )
        
    param_modules = [model, *heads, task_router]
    efficiency = eff.to_dict(param_modules)
    save_efficiency(run_dir, {"seed": seed, "method": profile_name, **efficiency})

    return {
        "results": results,
        "report_section": {"name": profile_name, "items": report_items},
        "extra_report_sections": extra_report_sections,
        "efficiency": efficiency,
        "run_dir": run_dir,
    }


def run_incremental_2(root_run_dir, datasets, *, seed: int = 42, multi_seed: bool = False):
    return _run_incremental_vae_router_profile(
        root_run_dir,
        datasets,
        seed=seed,
        multi_seed=multi_seed,
        profile=INCREMENTAL_2,
        profile_name="HiDMoA",
        profile_dir="HiDMoA",
    )


def run_incremental_3(root_run_dir, datasets, *, seed: int = 42, multi_seed: bool = False):
    return _run_incremental_vae_router_profile(
        root_run_dir,
        datasets,
        seed=seed,
        multi_seed=multi_seed,
        profile=INCREMENTAL_3,
        profile_name="incremental_3",
        profile_dir="incre3",
    )


def run_full(
    root_run_dir,
    datasets,
    *,
    seed: int = 42,
    multi_seed: bool = False,
    profile: Optional[dict] = None,
    profile_name: str = "full_1",
    run_subdir: str = "full1",
    checkpoint_name: str = "full1.pt",
    log_prefix: str = "[full1][model]",
):
    profile = profile or FULL_1
    cfg_m = profile["model"]
    cfg_t = profile["train"]
    d = DATA
    device = torch.device(cfg_t["device"])
    run_dir = create_profile_run_dir(root_run_dir, profile_subdir(run_subdir, seed, multi_seed))
    save_config(run_dir, {"data": d, "profile": profile})
    tlog = TrainLogger(run_dir)

    model = FullMoEResNet(
        backbone_name=cfg_m["backbone"],
        pretrained=cfg_m["pretrained"],
        moe_layers=cfg_m["moe_layers"],
        moe_channels=cfg_m["moe_channels"],
        bottleneck_ratios=cfg_m["bottleneck_ratio"],
        num_experts=cfg_m["num_experts"],
        top_k=cfg_m["top_k"],
    ).to(device)

    all_classes = list(range(d["num_classes"]))
    loaders = build_task_loaders(datasets, all_classes, cfg_t["batch_size"], d["num_workers"])
    head = PrototypeHead(cfg_m["feat_dim"], d["num_classes"], cfg_m["scale"]).to(device)
    eff = EfficiencyTracker(device, image_size=int(d["image_size"]))
    batch_size = int(cfg_t["batch_size"])

    if cfg_t.get("imprint_init", True):
        model.eval()
        fi, li = [], []
        with torch.no_grad():
            for imgs, lbls in loaders["train_eval"]:
                fi.append(model(imgs.to(device)).cpu())
                li.append(lbls)
        _imprint_or_fail(head, fi, li, all_classes, "full training")

    with eff.train_timer():
        eff.record_training_modules(model, head)
        best_val, best_ep, epoch_logs = train_full(
            model, head, loaders["train"], loaders["val"],
            lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"],
            epochs=cfg_t["epochs"], patience=cfg_t["early_stopping_patience"],
            min_delta=cfg_t["early_stopping_min_delta"], device=device,
            log_prefix=log_prefix,
        )
        eff.record_training_modules(model, head)
        if is_profile_flops_enabled():
            n_ep = max(len(epoch_logs), 1)
            fwd = measure_full_forward_per_image(model, head, device, int(d["image_size"])) or 0
            if fwd > 0:
                n_train = count_loader_samples(loaders["train"]) * n_ep
                n_val = count_loader_samples(loaders["val"]) * n_ep
                eff.add_train_only_flops(
                    estimate_train_flops_from_steps(fwd, n_train, backward_factor=3.0)
                )
                eff.add_eval_flops(
                    estimate_eval_flops_from_steps(fwd, n_val)
                )

    tlog.log_epoch_table("model", 0, epoch_logs)
    plot_loss_curves(run_dir, epoch_logs, "full_1", "model", session=0)
    with eff.test_timer():
        test_metrics = evaluate_full(model, head, loaders["test"], device, d["class_names"], verbose=True)
    if is_profile_flops_enabled():
        eval_per_image = measure_full_forward_per_image(model, head, device, int(d["image_size"]))
        eval_samples = count_loader_samples(loaders["test"])
        if eval_per_image:
            eff.add_eval_flops(
                estimate_eval_flops_from_steps(eval_per_image, eval_samples)
            )

    router_sessions = [{
        "session": 0,
        "num_experts": cfg_m["num_experts"],
        "train": _collect_full_routing_stats(model, loaders["train_eval"], device),
        "test": _collect_full_routing_stats(model, loaders["test"], device),
    }]
    save_router_stats(run_dir, profile_name, router_sessions)
    save_summary(run_dir, {"profile": profile_name, "test": test_metrics, "best_val_loss": best_val})
    _save_full_model(
        run_dir,
        checkpoint_name,
        {
            "model_state": model.state_dict(),
            "head": head.state_dict(),
            "best_val_loss": best_val,
            "classes": all_classes,
        },
    )

    if is_profile_flops_enabled():
        eff.set_flops_forward_per_image(
            measure_full_forward_per_image(model, head, device, int(d["image_size"]))
        )
    efficiency = eff.to_dict([model, head])
    save_efficiency(run_dir, {"seed": seed, "method": profile_name, **efficiency})

    return {
        "metrics": {**test_metrics, "best_val_loss": best_val},
        "report_section": {
            "name": profile_name,
            "items": [{"title": "full", "metrics": test_metrics, "class_ids": all_classes}],
        },
        "efficiency": efficiency,
        "run_dir": run_dir,
    }

def run_full_fixed(root_run_dir, datasets, *, seed: int = 42, multi_seed: bool = False):
    cfg_m = FULL_2["model"]
    cfg_t = FULL_2["train"]
    d = DATA
    device = torch.device(cfg_t["device"])
    task_splits = d["task_splits"]
    run_dir = create_profile_run_dir(root_run_dir, profile_subdir("full2", seed, multi_seed))
    save_config(run_dir, {"data": d, "profile": FULL_2})
    tlog = TrainLogger(run_dir)

    model = IncrementalMoEResNet(
        backbone_name=cfg_m["backbone"],
        pretrained=cfg_m["pretrained"],
        moe_layers=cfg_m["moe_layers"],
        moe_channels=cfg_m["moe_channels"],
        bottleneck_ratios=cfg_m["bottleneck_ratio"],
    ).to(device)
    for tid in range(len(task_splits)):
        model.add_task(tid, cfg_m["experts_per_task"])

    all_classes = list(range(d["num_classes"]))
    loaders = build_task_loaders(datasets, all_classes, cfg_t["batch_size"], d["num_workers"])
    heads = [PrototypeHead(cfg_m["feat_dim"], len(tc), cfg_m["scale"]).to(device) for tc in task_splits]
    if cfg_t.get("imprint_init", True):
        model.eval()
        for tid, tc in enumerate(task_splits):
            fl, ll = [], []
            with torch.no_grad():
                for imgs, lbls in loaders["train_eval"]:
                    mask = torch.zeros(len(lbls), dtype=torch.bool)
                    for c in tc:
                        mask |= lbls == c
                    if mask.sum() == 0:
                        continue
                    fl.append(model(imgs[mask].to(device), tid).cpu())
                    ll.append(lbls[mask])
            _imprint_or_fail(heads[tid], fl, ll, tc, f"full fixed task {tid}")

    eff = EfficiencyTracker(device, image_size=int(d["image_size"]))
    with eff.train_timer():
        eff.record_training_modules(model, *heads)
        best_val, best_ep, epoch_logs = train_full_fixed(
            model, heads, loaders["train"], loaders["val"], task_splits,
            lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"],
            epochs=cfg_t["epochs"], patience=cfg_t["early_stopping_patience"],
            min_delta=cfg_t["early_stopping_min_delta"], device=device,
            log_prefix="[full2][model]",
        )
        eff.record_training_modules(model, *heads)
        if is_profile_flops_enabled():
            n_ep = max(len(epoch_logs), 1)
            fwd = measure_incremental2_forward_per_image(model, heads, device, int(d["image_size"])) or 0
            if fwd > 0:
                n_train = count_loader_samples(loaders["train"]) * n_ep
                n_val = count_loader_samples(loaders["val"]) * n_ep
                eff.add_train_only_flops(
                    estimate_train_flops_from_steps(fwd, n_train, backward_factor=3.0)
                )
                eff.add_eval_flops(
                    estimate_eval_flops_from_steps(fwd, n_val)
                )

    tlog.log_epoch_table("model", 0, epoch_logs)
    plot_loss_curves(run_dir, epoch_logs, "full_2", "model", session=0)

    eval_mode = cfg_t.get("eval_mode", "max_confidence")
    use_oracle = eval_mode == "oracle_task"
    with eff.test_timer():
        metrics_conf = evaluate_full_fixed(
            model, heads, loaders["test"], task_splits, device,
            class_names=d["class_names"], verbose=True, use_oracle_task=False,
        )
        metrics_oracle = evaluate_full_fixed(
            model, heads, loaders["test"], task_splits, device,
            class_names=d["class_names"], verbose=True, use_oracle_task=True,
        )
    if is_profile_flops_enabled():
        eval_per_image = measure_incremental2_forward_per_image(model, heads, device, int(d["image_size"]))
        eval_samples = count_loader_samples(loaders["test"]) * 2
        if eval_per_image:
            eff.add_eval_flops(
                estimate_eval_flops_from_steps(eval_per_image, eval_samples)
            )
    primary = metrics_oracle if use_oracle else metrics_conf

    router_sessions = []
    for tid, tc in enumerate(task_splits):
        router_sessions.append({
            "session": tid,
            "num_experts": cfg_m["experts_per_task"],
            "train": _collect_full_fixed_routing_stats(model, loaders["train_eval"], tid, tc, device),
            "test": _collect_full_fixed_routing_stats(model, loaders["test"], tid, tc, device),
        })
    save_router_stats(run_dir, "full_2", router_sessions)
    save_summary(run_dir, {
        "profile": "full_2",
        "max_confidence": metrics_conf,
        "oracle_task": metrics_oracle,
        "best_val_loss": best_val,
    })
    _save_full_model(
        run_dir,
        "full2.pt",
        {
            "model_state": model.state_dict(),
            "heads": [h.state_dict() for h in heads],
            "task_splits": task_splits,
            "best_val_loss": best_val,
        },
    )

    if is_profile_flops_enabled():
        eff.set_flops_forward_per_image(
            measure_incremental2_forward_per_image(model, heads, device, int(d["image_size"]))
        )
    efficiency = eff.to_dict([model, *heads])
    save_efficiency(run_dir, {"seed": seed, "method": "full_2", **efficiency})

    return {
        "metrics": {**primary, "metrics_max_confidence": metrics_conf, "metrics_oracle_task": metrics_oracle},
        "report_section": {
            "name": "full_2",
            "items": [{"title": "full", "metrics": primary, "class_ids": all_classes}],
        },
        "efficiency": efficiency,
        "run_dir": run_dir,
    }


def _bootstrap_reproducibility(base_seed: int) -> None:
    if internal_strict_reproducibility_enabled():
        bootstrap_deterministic_env(base_seed)
        print("[repro] strict_reproducibility=all: strict mode for internal + external methods")
        return
    if cil_deterministic_enabled():
        mode = str(EFFICIENCY.get("strict_reproducibility", "all"))
        print(
            f"[repro] strict_reproducibility={mode}: "
            "incre/full/evolve use 0513-like seed; external subprocesses stay strict"
        )
        return
    print("[repro] CIL_DETERMINISTIC=0: all methods use fast non-strict seed")


def _bootstrap_efficiency_profiling() -> None:
    enabled = bool(EFFICIENCY.get("profile_flops", True))
    set_profile_flops_enabled(enabled)
    if not enabled:
        print("[efficiency] profile_flops=False: skip flops_* (params/gpu/time unchanged)")


def main():
    seeds = experiment_seeds()
    multi_seed = len(seeds) > 1
    run_mode = _normalize_run_mode(RUN_MODE)
    root_run_dir = create_root_run_dir("runs")
    base_seed = int(DATA["seed"])
    _bootstrap_reproducibility(base_seed)
    _bootstrap_efficiency_profiling()
    reset_incremental_2_alpha_cache()
    save_config(
        root_run_dir,
        {
            "run_mode": run_mode,
            "experiment_seeds": seeds,
            "repeats": int(DATA.get("repeats", 1)),
            "base_seed": base_seed,
            "efficiency": deepcopy(EFFICIENCY),
        },
    )
    sections_by_name: Dict[str, List[dict]] = {}
    section_order: List[str] = []
    efficiency_by_name: Dict[str, List[dict]] = {}
    efficiency_order: List[str] = []
    datasets = None

    needs_internal = run_mode in (
        "incremental_1",
        "incremental_2",
        "incremental_3",
        "full_1",
        "full_2",
        "all",
    )
    if needs_internal:
        datasets = prepare_data()

    for seed in seeds:
        DATA["seed"] = int(seed)
        set_seed(int(seed))
        print(f"[experiment] seed={seed} ({seeds.index(seed) + 1}/{len(seeds)})")

        if run_mode in ("incremental_1", "all"):
            incr = run_incremental(root_run_dir, datasets, seed=seed, multi_seed=multi_seed)
            _record_section_run(
                sections_by_name, section_order, incr["report_section"], seed, run_dir=incr.get("run_dir")
            )
            _record_efficiency_run(
                efficiency_by_name, efficiency_order, incr.get("efficiency"), "incremental_1", seed
            )

        if run_mode in ("incremental_2", "all"):
            incr2 = run_incremental_2(root_run_dir, datasets, seed=seed, multi_seed=multi_seed)
            _record_section_run(
                sections_by_name, section_order, incr2["report_section"], seed, run_dir=incr2.get("run_dir")
            )
            for extra in incr2.get("extra_report_sections", []):
                _record_section_run(
                    sections_by_name,
                    section_order,
                    extra["section"],
                    seed,
                    run_dir=incr2.get("run_dir"),
                    filename=extra.get("filename", "test.txt"),
                )
            _record_efficiency_run(
                efficiency_by_name, efficiency_order, incr2.get("efficiency"), "HiDMoA", seed
            )

        if run_mode == "incremental_3":
            incr3 = run_incremental_3(root_run_dir, datasets, seed=seed, multi_seed=multi_seed)
            _record_section_run(
                sections_by_name, section_order, incr3["report_section"], seed, run_dir=incr3.get("run_dir")
            )
            for extra in incr3.get("extra_report_sections", []):
                _record_section_run(
                    sections_by_name,
                    section_order,
                    extra["section"],
                    seed,
                    run_dir=incr3.get("run_dir"),
                    filename=extra.get("filename", "test.txt"),
                )
            _record_efficiency_run(
                efficiency_by_name, efficiency_order, incr3.get("efficiency"), "incremental_3", seed
            )

        if run_mode in ("full_1", "all"):
            full1 = run_full(root_run_dir, datasets, seed=seed, multi_seed=multi_seed)
            _record_section_run(
                sections_by_name, section_order, full1["report_section"], seed, run_dir=full1.get("run_dir")
            )
            _record_efficiency_run(
                efficiency_by_name, efficiency_order, full1.get("efficiency"), "full_1", seed
            )

        if run_mode in ("full_2", "all"):
            full2 = run_full_fixed(root_run_dir, datasets, seed=seed, multi_seed=multi_seed)
            _record_section_run(
                sections_by_name, section_order, full2["report_section"], seed, run_dir=full2.get("run_dir")
            )
            _record_efficiency_run(
                efficiency_by_name, efficiency_order, full2.get("efficiency"), "full_2", seed
            )

        if run_mode in EXTERNAL_INCREMENTAL_METHODS:
            ext_result = run_external_incremental_method(
                root_run_dir, run_mode, seed, multi_seed=multi_seed
            )
            _record_section_run(
                sections_by_name,
                section_order,
                ext_result["report_section"],
                seed,
                run_dir=ext_result.get("run_dir"),
            )
            _record_efficiency_run(
                efficiency_by_name, efficiency_order, ext_result.get("efficiency"), run_mode, seed
            )

    if not section_order:
        supported_modes = [
            "incremental_1",
            "incremental_2",
            "incremental_3",
            "full_1",
            "full_2",
            "all",
            *sorted(EXTERNAL_INCREMENTAL_METHODS.keys()),
        ]
        raise ValueError(
            f"RUN_MODE '{RUN_MODE}' (normalized='{run_mode}') did not match any runnable profile. "
            f"Supported values: {supported_modes}"
        )

    report_sections = [
        aggregate_report_section(name, sections_by_name[name]) for name in section_order
    ]
    test_filename = "test_repeat.txt" if multi_seed else "test.txt"
    cost_filename = "cost_repeat.txt" if multi_seed else "cost.txt"
    write_test_report(
        root_run_dir,
        report_sections,
        experiment_seeds=seeds if multi_seed else None,
        filename=test_filename,
        aggregate_header=multi_seed,
    )
    if efficiency_order:
        write_cost_report(
            root_run_dir,
            efficiency_by_name,
            efficiency_order,
            experiment_seeds=seeds if multi_seed else None,
            filename=cost_filename,
            aggregate_header=multi_seed,
        )
    print(f"Results saved -> {root_run_dir}")
    if multi_seed:
        print(f"Per-seed test.txt under */seed_N/; aggregated {test_filename} (mean±std over {seeds})")


if __name__ == "__main__":
    main()
