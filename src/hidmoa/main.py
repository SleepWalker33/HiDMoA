"""
NEU 钢铁缺陷 — 类增量学习 & 全量基线 主入口
"""

import os
import json
import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

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
    COMMON_HEAD,
    DATA,
    EFFICIENCY,
    INCREMENTAL_2,
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
    measure_incremental1_forward_per_image,
    measure_incremental_top1_forward_per_image,
    measure_incremental1_train_step_flops,
    profile_module_forward_flops,
    save_efficiency,
    set_profile_flops_enabled,
)
from .logger import (
    TrainLogger,
    aggregate_report_section,
    create_profile_run_dir,
    create_root_run_dir,
    finalize_report_section,
    plot_loss_curves,
    profile_subdir,
    save_config,
    save_router_stats,
    save_summary,
    write_seed_test_report,
    write_cost_report,
    write_test_report,
)
from .models import (
    CosineHead,
    CosinePrototypeHead,
    IncrementalMoEResNet,
    StandardPrototypeHead,
    LinearHead,
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


def _normalize_run_mode(mode: str) -> str:
    mode = str(mode).strip()
    legacy_internal_aliases = {
        "hidmoa": "incremental_2",
        "HiDMoA": "incremental_2",
    }
    if mode in legacy_internal_aliases:
        alias = legacy_internal_aliases[mode]
        print(f"[warn] RUN_MODE '{mode}' remapped to '{alias}'")
        return alias
    return mode


def _record_section_run(
    sections_by_name: Dict[str, List[dict]],
    section_order: List[str],
    section: dict,
    seed: int,
    run_dir: Optional[str] = None,
    filename: str = "test.txt",
) -> None:
    if not section or not section.get("name"):
        return
    section_name = str(section.get("name"))
    if section_name not in sections_by_name:
        section_order.append(section_name)
        sections_by_name[section_name] = []
    sections_by_name[section_name].append({"seed": int(seed), "section": section})
    write_seed_test_report(run_dir, finalize_report_section(section), int(seed), filename=filename)


def _record_efficiency_run(
    efficiency_by_name: Dict[str, List[dict]],
    section_order: List[str],
    efficiency: Optional[dict],
    method_name: str,
    seed: int,
) -> None:
    if not efficiency:
        return
    eff_record = dict(efficiency)
    eff_record.setdefault("seed", int(seed))
    if method_name not in efficiency_by_name:
        section_order.append(method_name)
        efficiency_by_name[method_name] = []
    efficiency_by_name[method_name].append(eff_record)


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
    save_path = os.path.join(_profile_models_dir(run_dir), f"hidmoa_session{session_id + 1}.pt")
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


def _bootstrap_reproducibility(base_seed: int) -> None:
    if internal_strict_reproducibility_enabled():
        bootstrap_deterministic_env(base_seed)
        print("[repro] strict_reproducibility=all: strict mode for HiDMoA internal pipeline")
        return
    if cil_deterministic_enabled():
        mode = str(EFFICIENCY.get("strict_reproducibility", "all"))
        print(
            f"[repro] strict_reproducibility={mode}: "
            "HiDMoA uses simplified seed mode"
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

    needs_internal = run_mode == "incremental_2"
    if needs_internal:
        datasets = prepare_data()

    for seed in seeds:
        DATA["seed"] = int(seed)
        set_seed(int(seed))
        print(f"[experiment] seed={seed} ({seeds.index(seed) + 1}/{len(seeds)})")

        if run_mode == "incremental_2":
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

    if not section_order:
        supported_modes = ["incremental_2"]
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
