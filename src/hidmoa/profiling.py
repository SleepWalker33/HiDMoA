"""
Efficiency metrics for CIL runs (one seed / one repeat).

Run-level fields:
  params_total, params_ever_trained, params_never_trained
  flops_forward_per_image  — final-session inference only
  flops_train_total        — train-side + eval-side FLOPs over all sessions
  flops_train_only         — train-side FLOPs only
  flops_eval_total         — eval-side FLOPs only
  gpu_peak_mb              — peak allocated GPU memory (train + test)
  time_train_pipeline_sec  — wall time for the full train pipeline (includes val)
  time_test_eval_sec       — wall time for test-set evaluation only
  time_train_total_sec     — wall time for training (includes val)
  time_test_total_sec      — wall time for test-set evaluation only
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Set, Union

import os

import torch
import torch.nn as nn

ModuleRef = Union[nn.Module, Dict, None]


def _parse_profile_flops_env(default: bool = True) -> bool:
    raw = os.getenv("CIL_PROFILE_FLOPS", "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


_PROFILE_FLOPS_ENABLED = _parse_profile_flops_env(True)


def set_profile_flops_enabled(enabled: bool) -> None:
    """Global switch: when False, skip all FLOPs (thop + estimates); params/time/GPU unchanged."""
    global _PROFILE_FLOPS_ENABLED
    _PROFILE_FLOPS_ENABLED = bool(enabled)


def is_profile_flops_enabled() -> bool:
    return _PROFILE_FLOPS_ENABLED


class EfficiencyTracker:
    def __init__(self, device: torch.device, image_size: int = 224):
        self.device = device
        self.image_size = int(image_size)
        self.time_train_total_sec = 0.0
        self.time_test_total_sec = 0.0
        self.flops_train_only = 0
        self.flops_eval_total = 0
        self.flops_forward_per_image: Optional[int] = None
        self._ever_trained_ids: Set[int] = set()
        self._train_depth = 0
        self._test_depth = 0
        self._train_t0: Optional[float] = None
        self._test_t0: Optional[float] = None
        self._gpu_peak_mb = 0.0

    def gpu_reset_peak(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

    def gpu_update_peak(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            self._gpu_peak_mb = max(self._gpu_peak_mb, float(mb))

    def record_training_modules(self, *modules: ModuleRef) -> None:
        for mod in modules:
            if mod is None:
                continue
            if isinstance(mod, dict):
                inner = mod.get("model")
                if isinstance(inner, nn.Module):
                    mod = inner
                else:
                    continue
            if not isinstance(mod, nn.Module):
                continue
            for p in mod.parameters():
                if p.requires_grad:
                    self._ever_trained_ids.add(id(p))

    def add_train_flops(self, flops: int) -> None:
        self.add_train_only_flops(flops)

    def add_train_only_flops(self, flops: int) -> None:
        if not is_profile_flops_enabled():
            return
        if flops and flops > 0:
            self.flops_train_only += int(flops)

    def add_eval_flops(self, flops: int) -> None:
        if not is_profile_flops_enabled():
            return
        if flops and flops > 0:
            self.flops_eval_total += int(flops)

    @contextmanager
    def train_timer(self):
        self._train_depth += 1
        if self._train_depth == 1:
            self.gpu_reset_peak()
            self._train_t0 = time.perf_counter()
        try:
            yield
        finally:
            self._train_depth -= 1
            if self._train_depth == 0 and self._train_t0 is not None:
                self.time_train_total_sec += time.perf_counter() - self._train_t0
                self.gpu_update_peak()
                self._train_t0 = None

    @contextmanager
    def test_timer(self):
        self._test_depth += 1
        if self._test_depth == 1:
            self.gpu_reset_peak()
            self._test_t0 = time.perf_counter()
        try:
            yield
        finally:
            self._test_depth -= 1
            if self._test_depth == 0 and self._test_t0 is not None:
                self.time_test_total_sec += time.perf_counter() - self._test_t0
                self.gpu_update_peak()
                self._test_t0 = None

    def count_params(self, modules: Iterable[ModuleRef]) -> Dict[str, int]:
        total = 0
        ever = 0
        for mod in modules:
            if mod is None:
                continue
            if isinstance(mod, dict):
                inner = mod.get("model")
                if isinstance(inner, nn.Module):
                    mod = inner
                else:
                    continue
            if not isinstance(mod, nn.Module):
                continue
            for p in mod.parameters():
                n = int(p.numel())
                total += n
                if id(p) in self._ever_trained_ids:
                    ever += n
        never = max(total - ever, 0)
        return {
            "params_total": total,
            "params_ever_trained": ever,
            "params_never_trained": never,
        }

    def set_flops_forward_per_image(self, flops: Optional[int]) -> None:
        if not is_profile_flops_enabled():
            return
        if flops is not None and flops > 0:
            self.flops_forward_per_image = int(flops)

    def to_dict(self, modules: Iterable[ModuleRef]) -> dict:
        out = self.count_params(modules)
        out["flops_forward_per_image"] = self.flops_forward_per_image
        flops_train_total = int(self.flops_train_only + self.flops_eval_total)
        out["flops_train_only"] = int(self.flops_train_only) if self.flops_train_only else None
        out["flops_eval_total"] = int(self.flops_eval_total) if self.flops_eval_total else None
        out["flops_train_total"] = flops_train_total if flops_train_total > 0 else None
        out["gpu_peak_mb"] = round(self._gpu_peak_mb, 2) if self._gpu_peak_mb > 0 else None
        out["time_train_pipeline_sec"] = round(self.time_train_total_sec, 3)
        out["time_test_eval_sec"] = round(self.time_test_total_sec, 3)
        out["time_train_total_sec"] = round(self.time_train_total_sec, 3)
        out["time_test_total_sec"] = round(self.time_test_total_sec, 3)
        return out


def _macs_to_flops(macs: int) -> int:
    return int(macs) * 2


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _cleanup_thop_artifacts(module: nn.Module) -> None:
    """Remove thop forward hooks / buffers left after a failed or partial profile()."""
    for m in module.modules():
        hooks = getattr(m, "_forward_hooks", None)
        if hooks is not None and hasattr(hooks, "clear"):
            hooks.clear()
        pre_hooks = getattr(m, "_forward_pre_hooks", None)
        if pre_hooks is not None and hasattr(pre_hooks, "clear"):
            pre_hooks.clear()
        for name in ("total_ops", "total_params"):
            if name in getattr(m, "_buffers", {}):
                m._buffers.pop(name, None)


def profile_module_forward_flops(module: nn.Module, inputs: tuple, device: torch.device) -> Optional[int]:
    """Return FLOPs for one forward pass; None if profiling unavailable."""
    if not is_profile_flops_enabled():
        return None
    original_device = _module_device(module)
    try:
        module.to(device)
        module.eval()
        inputs = tuple(
            x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs
        )
        try:
            from thop import profile

            macs, _ = profile(module, inputs=inputs, verbose=False)
            return _macs_to_flops(int(macs))
        except Exception:
            pass

        try:
            with torch.no_grad():
                module(*inputs)
            return _fallback_flops_estimate(module)
        except Exception:
            return _fallback_flops_estimate(module)
    finally:
        _cleanup_thop_artifacts(module)
        module.to(original_device)


def _fallback_flops_estimate(module: nn.Module) -> int:
    """Rough FLOPs ≈ 2 × param count per forward (fallback when thop missing)."""
    params = sum(int(p.numel()) for p in module.parameters())
    return max(params * 2, 0)


def _feature_generator_probe_inputs(
    model: nn.Module,
    generator_state: dict,
    batch_size: int,
    device: torch.device,
) -> Optional[tuple]:
    class_name = model.__class__.__name__
    if class_name not in {"FeatureVAE", "FeatureConditionalVAE", "FeatureConditionalVQVAE"}:
        return None

    input_dim = int(generator_state.get("input_dim", getattr(model, "input_dim", 0)))
    if input_dim <= 0:
        input_dim = int(getattr(model, "input_dim", 0))
    if input_dim <= 0:
        return None

    bs = max(1, min(batch_size, 4))
    x = torch.randn(bs, input_dim, device=device)
    if class_name == "FeatureVAE":
        return (x,)

    num_classes = int(getattr(model, "num_classes", len(generator_state.get("class_ids", [])) or 1))
    y = torch.zeros(bs, dtype=torch.long, device=device)
    if num_classes > 1:
        y = torch.arange(bs, device=device, dtype=torch.long) % num_classes
    return (x, y)


def estimate_train_flops_from_steps(forward_flops_per_sample: int, num_samples: int, backward_factor: float = 3.0) -> int:
    """Training FLOPs ≈ forward × num_samples × (1 + backward overhead)."""
    if forward_flops_per_sample <= 0 or num_samples <= 0:
        return 0
    return int(forward_flops_per_sample * num_samples * backward_factor)


def estimate_eval_flops_from_steps(forward_flops_per_sample: int, num_samples: int) -> int:
    """Eval FLOPs ≈ forward × num_samples."""
    if forward_flops_per_sample <= 0 or num_samples <= 0:
        return 0
    return int(forward_flops_per_sample * num_samples)


def count_loader_samples(loader) -> int:
    try:
        return len(loader.dataset)
    except Exception:
        n = 0
        for batch in loader:
            if isinstance(batch, (list, tuple)) and batch:
                n += batch[0].size(0)
        return n


def measure_incremental1_train_step_flops(
    model: nn.Module,
    head: nn.Module,
    task_id: int,
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> int:
    class _Probe(nn.Module):
        def __init__(self, backbone, classifier, tid):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier
            self.tid = tid

        def forward(self, x):
            return self.classifier(self.backbone(x, self.tid))

    probe = _Probe(model, head, task_id)
    x = torch.randn(batch_size, 3, image_size, image_size)
    flops = profile_module_forward_flops(probe, (x,), device)
    return flops or 0


def measure_incremental1_forward_per_image(
    model: nn.Module,
    heads: List[nn.Module],
    task_id_clf: Optional[nn.Module],
    device: torch.device,
    image_size: int,
    taskid_image_size: int,
    num_seen_tasks: int,
    oracle_taskid: bool,
) -> Optional[int]:
    """Inference FLOPs for one test image at final session (incremental_1 path)."""
    x = torch.randn(1, 3, image_size, image_size)
    total = 0
    last_tid = max(num_seen_tasks - 1, 0)
    class _BackboneHead(nn.Module):
        def __init__(self, backbone, head, tid):
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.tid = tid

        def forward(self, inp):
            return self.head(self.backbone(inp, self.tid))

    for tid in range(num_seen_tasks):
        f = profile_module_forward_flops(_BackboneHead(model, heads[tid], tid), (x,), device)
        if f:
            total += f

    if num_seen_tasks > 1 and not oracle_taskid and task_id_clf is not None:
        x_tid = torch.randn(1, 3, taskid_image_size, taskid_image_size)
        f_tid = profile_module_forward_flops(task_id_clf, (x_tid,), device)
        if f_tid:
            total += f_tid
    elif total == 0:
        f = profile_module_forward_flops(_BackboneHead(model, heads[last_tid], last_tid), (x,), device)
        total = f or 0
    return total or None


def measure_incremental_top1_forward_per_image(
    model: nn.Module,
    heads: List[nn.Module],
    router: Optional[nn.Module],
    device: torch.device,
    image_size: int,
    route_image_size: int,
) -> Optional[int]:
    """Inference FLOPs for one routed image: router/top1 selector + one selected head."""
    if not heads:
        return None

    class _BackboneHead(nn.Module):
        def __init__(self, backbone, head, tid):
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.tid = tid

        def forward(self, inp):
            return self.head(self.backbone(inp, self.tid))

    num_seen_tasks = len(heads)
    selected_tid = max(num_seen_tasks - 1, 0)
    total = 0

    if router is not None and num_seen_tasks > 1:
        x_route = torch.randn(1, 3, route_image_size, route_image_size)
        saved_profile_importance_batch = None
        if hasattr(router, "profile_importance_batch"):
            saved_profile_importance_batch = getattr(router, "profile_importance_batch")
            router_eval_s = max(1, int(getattr(router, "eval_importance_samples", 32)))
            setattr(router, "profile_importance_batch", min(32, router_eval_s))
        router_original_device = _module_device(router)
        try:
            f_router = profile_module_forward_flops(router, (x_route,), device)
            if f_router:
                total += f_router
            router.to(device)
            router.eval()
            with torch.no_grad():
                router_logits = router(x_route.to(device))
            if isinstance(router_logits, torch.Tensor) and router_logits.ndim >= 2 and router_logits.size(1) > 0:
                pred_tid = int(router_logits.argmax(dim=1)[0].item())
                if 0 <= pred_tid < num_seen_tasks:
                    selected_tid = pred_tid
        except Exception:
            pass
        finally:
            if hasattr(router, "profile_importance_batch"):
                setattr(router, "profile_importance_batch", saved_profile_importance_batch)
            router.to(router_original_device)

    x = torch.randn(1, 3, image_size, image_size)
    f_head = profile_module_forward_flops(
        _BackboneHead(model, heads[selected_tid], selected_tid),
        (x,),
        device,
    )
    if f_head:
        total += f_head
    return total or None


def measure_incremental2_forward_per_image(
    model: nn.Module,
    heads: List[nn.Module],
    device: torch.device,
    image_size: int,
) -> Optional[int]:
    if not heads:
        return None

    total = 0
    x = torch.randn(1, 3, image_size, image_size)
    for task_id, head in enumerate(heads):
        class _Probe(nn.Module):
            def __init__(self, backbone, classifier_head, tid):
                super().__init__()
                self.backbone = backbone
                self.head = classifier_head
                self.tid = tid

            def forward(self, inp):
                return self.head(self.backbone(inp, self.tid))

        f = profile_module_forward_flops(_Probe(model, head, task_id), (x,), device)
        if f:
            total += f

    return total


def measure_full_forward_per_image(model: nn.Module, head: nn.Module, device: torch.device, image_size: int) -> Optional[int]:
    class _Probe(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier

        def forward(self, x):
            return self.classifier(self.backbone(x))

    x = torch.randn(1, 3, image_size, image_size)
    return profile_module_forward_flops(_Probe(model, head), (x,), device)


def accumulate_generator_flops_split(
    generator_state: dict,
    device: torch.device,
    backward_factor: float = 3.0,
) -> tuple[int, int]:
    if not is_profile_flops_enabled():
        return 0, 0
    model = generator_state.get("model")
    if not isinstance(model, nn.Module):
        return 0, 0
    epoch_logs = generator_state.get("epoch_logs") or []
    epochs = len(epoch_logs) if epoch_logs else int(generator_state.get("epochs", 1))
    train_images = generator_state.get("train_images")
    num_train_samples = int(train_images.size(0)) if isinstance(train_images, torch.Tensor) else 0
    if num_train_samples <= 0:
        num_train_samples = int(generator_state.get("num_train_samples", 0))
    num_val_samples = int(generator_state.get("num_val_samples", 0))
    batch_size = int(generator_state.get("batch_size", 32))
    if batch_size <= 0:
        batch_size = 32
    feature_inputs = _feature_generator_probe_inputs(model, generator_state, batch_size, device)
    if feature_inputs is not None:
        try:
            fwd = profile_module_forward_flops(model, feature_inputs, device) or 0
            per_sample = max(fwd // max(feature_inputs[0].size(0), 1), 1)
        except Exception:
            per_sample = _fallback_flops_estimate(model)
    # VQ-VAE / conditional generators: avoid thop (hooks can leak into decode_indices).
    elif model.__class__.__name__ in ("TaskReplayVQVAE", "TaskReplayVAE"):
        per_sample = _fallback_flops_estimate(model)
    else:
        img_size = int(generator_state.get("image_size", 64))
        try:
            x = torch.randn(min(batch_size, 4), 3, img_size, img_size, device=device)
            fwd = profile_module_forward_flops(model, (x,), device) or 0
            per_sample = max(fwd // max(x.size(0), 1), 1)
        except Exception:
            per_sample = _fallback_flops_estimate(model)
    train_flops = estimate_train_flops_from_steps(
        per_sample,
        max(num_train_samples, 0) * max(epochs, 1),
        backward_factor,
    )
    eval_flops = estimate_eval_flops_from_steps(
        per_sample,
        max(num_val_samples, 0) * max(epochs, 1),
    )
    return train_flops, eval_flops


def accumulate_generator_train_flops(generator_state: dict, device: torch.device, backward_factor: float = 3.0) -> int:
    train_flops, _ = accumulate_generator_flops_split(
        generator_state,
        device,
        backward_factor=backward_factor,
    )
    return train_flops


def save_efficiency(run_dir: str, payload: dict) -> str:
    import json
    import os

    path = os.path.join(run_dir, "efficiency.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
