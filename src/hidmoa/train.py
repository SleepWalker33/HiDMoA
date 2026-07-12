"""
训练 & 评估逻辑
  - train_task_experts
  - collect_task_images
  - train_vae / generate_pseudo_images
  - train_task_id_classifier
  - collect_incremental_routing_stats
  - evaluate_all
"""

from copy import deepcopy
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
try:
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:
    ResNet18_Weights = None
    from torchvision.models import resnet18

try:
    import optuna
except ImportError:
    optuna = None

from .data import IMAGENET_MEAN, IMAGENET_STD
from .metrics import compute_metrics
from .models import (
    ClassConditionalPixelCNN,
    FeatureTaskIDClassifier,
    IncrementalMoEResNet,
    LinearHead,
    PrototypeHead,
    FeatureVAE,
    FeatureConditionalVAE,
    FeatureConditionalVQVAE,
    TaskIDClassifier,
    TaskReplayVAE,
    TaskReplayVQVAE,
)


def _resize_images(images: torch.Tensor, image_size: int) -> torch.Tensor:
    if images.shape[-1] == image_size and images.shape[-2] == image_size:
        return images
    return F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)


def _imagenet_stats(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return mean, std


def _denormalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    mean, std = _imagenet_stats(images)
    return (images * std + mean).clamp(0.0, 1.0)


def _normalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    mean, std = _imagenet_stats(images)
    return (images - mean) / std


def _build_resnet18_perceptual(
    pretrained: bool = True,
    max_stage: int = 3,
) -> nn.Module:
    try:
        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
        else:
            backbone = resnet18(pretrained=pretrained)
    except Exception:
        try:
            backbone = resnet18(weights=None)
        except TypeError:
            backbone = resnet18(pretrained=False)
    stages = [
        nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
        nn.Sequential(backbone.maxpool, backbone.layer1),
        backbone.layer2,
        backbone.layer3,
    ]
    model = nn.ModuleList(stages[: max(1, int(max_stage))])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


class PerceptualLoss(nn.Module):
    def __init__(self, pretrained: bool = True, max_stage: int = 3):
        super().__init__()
        self.feature_blocks = _build_resnet18_perceptual(pretrained=pretrained, max_stage=max_stage)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean, std = _imagenet_stats(pred)
        pred = (pred - mean) / std
        target = (target - mean) / std
        loss = pred.new_tensor(0.0)
        for block in self.feature_blocks:
            pred = block(pred)
            target = block(target)
            loss = loss + F.l1_loss(pred, target)
        return loss


@torch.no_grad()
def _collect_cvae_latent_pool(
    vae: TaskReplayVAE,
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> List[torch.Tensor]:
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)
    chunks = {cls_id: [] for cls_id in range(num_classes)}
    for xb, yb in loader:
        mu, _ = vae.encode(xb.to(device), yb.to(device))
        for cls_id in yb.unique().tolist():
            cls_id = int(cls_id)
            mask = yb == cls_id
            chunks[cls_id].append(mu[mask.to(device)].detach().cpu())

    latent_pool = []
    for cls_id in range(num_classes):
        if not chunks[cls_id]:
            raise ValueError(f"cannot build latent pool for local class {cls_id}")
        latent_pool.append(torch.cat(chunks[cls_id], dim=0))
    return latent_pool


@torch.no_grad()
def _collect_vqvae_code_pool(
    vqvae: TaskReplayVQVAE,
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> List[torch.Tensor]:
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)
    chunks = {cls_id: [] for cls_id in range(num_classes)}
    for xb, yb in loader:
        indices = vqvae.encode_indices(xb.to(device), yb.to(device)).detach().cpu()
        for cls_id in yb.unique().tolist():
            cls_id = int(cls_id)
            mask = yb == cls_id
            chunks[cls_id].append(indices[mask])

    code_pool = []
    for cls_id in range(num_classes):
        if not chunks[cls_id]:
            raise ValueError(f"cannot build code pool for local class {cls_id}")
        code_pool.append(torch.cat(chunks[cls_id], dim=0))
    return code_pool


def train_vqvae_pixelcnn_prior(
    train_code_indices: torch.Tensor,
    train_labels: torch.Tensor,
    val_code_indices: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    num_embeddings: int,
    hidden_channels: int,
    num_layers: int,
    kernel_size: int,
    dropout: float,
    epochs: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    log_prefix: str = "",
):
    prior = ClassConditionalPixelCNN(
        num_embeddings=num_embeddings,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)
    train_loader = DataLoader(
        TensorDataset(train_code_indices.long(), train_labels.long()),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_code_indices.long(), val_labels.long()),
        batch_size=batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.Adam(prior.parameters(), lr=lr, weight_decay=weight_decay)
    epoch_logs = []
    best_state = deepcopy(prior.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = early_stopping_patience
    for epoch in range(1, epochs + 1):
        prior.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = prior(xb, yb)
            loss = F.cross_entropy(logits, xb, reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_count += 1

        prior.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = prior(xb, yb)
                loss = F.cross_entropy(logits, xb, reduction="mean")
                val_loss_sum += loss.item()
                val_count += 1

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "val_loss": val_loss_sum / max(val_count, 1),
        })
        cur = epoch_logs[-1]
        if cur["val_loss"] < best_val_loss - early_stopping_min_delta:
            best_val_loss = cur["val_loss"]
            best_epoch = epoch
            best_state = deepcopy(prior.state_dict())
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"{log_prefix} PixelCNN prior epoch {epoch:>3d}/{epochs}: "
                f"train_loss={cur['train_loss']:.4f}  "
                f"val_loss={cur['val_loss']:.4f}"
            )
        if patience_left <= 0:
            print(f"{log_prefix} PixelCNN prior early stopping at epoch {epoch} (best val_loss at epoch {best_epoch})")
            break

    prior.load_state_dict(best_state)
    prior.eval()
    return prior, epoch_logs


def _supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Supervised contrastive loss over normalized task features."""
    if features.ndim != 2 or features.shape[0] <= 1:
        return features.new_zeros(())
    labels = labels.view(-1)
    if labels.numel() != features.shape[0]:
        return features.new_zeros(())

    features = F.normalize(features, dim=1)
    temperature = max(float(temperature), 1e-6)
    logits = (features @ features.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~self_mask)
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not bool(valid.any()):
        return features.new_zeros(())

    logits_mask = ~self_mask
    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (log_prob * positive_mask.float()).sum(dim=1) / positive_counts.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def _focal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float = 2.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce).clamp(1e-6, 1.0)
    return (((1.0 - pt) ** float(gamma)) * ce).mean()


def train_task_experts(
    model: IncrementalMoEResNet,
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_id: int,
    task_classes: List[int],
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    min_delta: float,
    device: torch.device,
    log_prefix: str = "",
    train_backbone: bool = False,
    class_internal_loss_cfg: Optional[Dict[str, Any]] = None,
):
    if len(train_loader.dataset) == 0:
        raise ValueError(f"task {task_id} train set is empty")
    if len(val_loader.dataset) == 0:
        raise ValueError(f"task {task_id} val set is empty")

    g2l = {g: l for l, g in enumerate(task_classes)}
    params = [p for p in model.moe_blocks.parameters() if p.requires_grad]
    if train_backbone:
        for module in model.backbone_modules:
            params.extend(p for p in module.parameters() if p.requires_grad)
    params += list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = None
    epoch_logs = []
    class_internal_loss_cfg = class_internal_loss_cfg or {}
    internal_lambda = float(class_internal_loss_cfg.get("lambda", 0.0) or 0.0)
    internal_enabled = internal_lambda > 0.0
    supcon_weight = float(class_internal_loss_cfg.get("supcon_weight", 1.0))
    supcon_temperature = float(class_internal_loss_cfg.get("temperature", 0.2))
    focal_weight = float(class_internal_loss_cfg.get("focal_weight", 0.5))
    focal_gamma = float(class_internal_loss_cfg.get("focal_gamma", 2.0))

    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        total_loss, total_ce, total_supcon, total_focal, correct, total = 0.0, 0.0, 0.0, 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            local_labels = torch.tensor([g2l[int(l.item())] for l in labels], device=device)

            feats = model(images, task_id)
            logits = head(feats)
            ce_loss = F.cross_entropy(logits, local_labels)
            supcon_loss = logits.new_zeros(())
            focal_loss = logits.new_zeros(())
            if internal_enabled:
                if supcon_weight > 0.0:
                    supcon_loss = _supervised_contrastive_loss(
                        feats,
                        local_labels,
                        temperature=supcon_temperature,
                    )
                if focal_weight > 0.0:
                    focal_loss = _focal_cross_entropy(
                        logits,
                        local_labels,
                        gamma=focal_gamma,
                    )
            loss = ce_loss + internal_lambda * (supcon_weight * supcon_loss + focal_weight * focal_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_ce += float(ce_loss.detach()) * images.size(0)
            total_supcon += float(supcon_loss.detach()) * images.size(0)
            total_focal += float(focal_loss.detach()) * images.size(0)
            correct += (logits.argmax(1) == local_labels).sum().item()
            total += images.size(0)

        scheduler.step()
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_acc, val_loss = _evaluate_task(model, head, val_loader, task_id, g2l, device)
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce_loss": total_ce / max(total, 1),
            "train_supcon_loss": total_supcon / max(total, 1),
            "train_focal_loss": total_focal / max(total, 1),
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        log_parts = [
            f"{log_prefix} Epoch {epoch:>3d}/{epochs}: ",
            f"train_loss={train_loss:.4f}  ",
            f"ce={epoch_logs[-1]['train_ce_loss']:.4f}  ",
        ]
        if internal_lambda > 0.0 or supcon_weight > 0.0:
            log_parts.append(f"supcon={epoch_logs[-1]['train_supcon_loss']:.4f}  ")
        if internal_lambda > 0.0 or focal_weight > 0.0:
            log_parts.append(f"focal={epoch_logs[-1]['train_focal_loss']:.4f}  ")
        log_parts.append(f"val_loss={val_loss:.4f}")
        print("".join(log_parts))

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {"head": {k: v.clone() for k, v in head.state_dict().items()}}
            if train_backbone:
                best_state["model"] = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                best_state["moe"] = {k: v.clone() for k, v in model.moe_blocks.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"{log_prefix} Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        if "model" in best_state:
            model.load_state_dict(best_state["model"], strict=False)
        else:
            model.moe_blocks.load_state_dict(best_state["moe"])
        head.load_state_dict(best_state["head"])

    return best_val_loss, best_epoch, epoch_logs


def _evaluate_task(
    model: IncrementalMoEResNet,
    head: nn.Module,
    loader: DataLoader,
    task_id: int,
    g2l: Dict[int, int],
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    head.eval()
    correct, total = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            local_labels = torch.tensor([g2l[int(l.item())] for l in labels], device=device)
            logits = head(model(images, task_id))
            total_loss += F.cross_entropy(logits, local_labels, reduction="sum").item()
            correct += (logits.argmax(1) == local_labels).sum().item()
            total += images.size(0)
    return correct / max(total, 1), total_loss / max(total, 1)


@torch.no_grad()
def collect_task_images(
    loader: DataLoader,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    images_all, labels_all = [], []
    for images, labels in loader:
        images_all.append(_resize_images(images, image_size).cpu())
        labels_all.append(labels.cpu())
    if not images_all:
        raise ValueError("cannot collect images from an empty loader")
    return torch.cat(images_all), torch.cat(labels_all)


def train_vae(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    class_ids: List[int],
    image_size: int,
    latent_dim: int,
    base_channels: int,
    channel_multipliers: List[int],
    epochs: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    lr: float,
    weight_decay: float,
    beta_kl: float,
    kl_warmup_epochs: int,
    recon_weight: float,
    l1_weight: float,
    perceptual_weight: float,
    perceptual_pretrained: bool,
    perceptual_layers: int,
    latent_pool_noise_std: float,
    batch_size: int,
    device: torch.device,
    log_prefix: str = "",
    model_name: str = "cVAE",
):
    g2l = {g: l for l, g in enumerate(class_ids)}
    train_local_labels = torch.tensor([g2l[int(l.item())] for l in train_labels], dtype=torch.long)
    val_local_labels = torch.tensor([g2l[int(l.item())] for l in val_labels], dtype=torch.long)
    train_images = _denormalize_imagenet(train_images.float())
    val_images = _denormalize_imagenet(val_images.float())
    vae = TaskReplayVAE(
        image_size=image_size,
        num_classes=len(class_ids),
        latent_dim=latent_dim,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
    ).to(device)
    train_loader = DataLoader(TensorDataset(train_images, train_local_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_local_labels), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)
    perceptual_loss_fn = None
    if perceptual_weight > 0.0:
        perceptual_loss_fn = PerceptualLoss(
            pretrained=perceptual_pretrained,
            max_stage=perceptual_layers,
        ).to(device)
        perceptual_loss_fn.eval()

    epoch_logs = []
    best_state = deepcopy(vae.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = early_stopping_patience
    for epoch in range(1, epochs + 1):
        vae.train()
        train_loss_sum = 0.0
        train_l1_sum = 0.0
        train_perc_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        train_count = 0
        beta_t = beta_kl if kl_warmup_epochs <= 0 else beta_kl * min(1.0, epoch / kl_warmup_epochs)
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            recon, mu, logvar = vae(xb, yb)
            loss_l1 = F.l1_loss(recon, xb)
            loss_perc = xb.new_tensor(0.0)
            if perceptual_loss_fn is not None:
                loss_perc = perceptual_loss_fn(recon, xb)
            loss_recon = recon_weight * l1_weight * loss_l1 + perceptual_weight * loss_perc
            loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss_recon + beta_t * loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_l1_sum += loss_l1.item()
            train_perc_sum += loss_perc.item()
            train_recon_sum += loss_recon.item()
            train_kl_sum += loss_kl.item()
            train_count += 1

        vae.eval()
        val_loss_sum = 0.0
        val_l1_sum = 0.0
        val_perc_sum = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon, mu, logvar = vae(xb, yb)
                loss_l1 = F.l1_loss(recon, xb)
                loss_perc = xb.new_tensor(0.0)
                if perceptual_loss_fn is not None:
                    loss_perc = perceptual_loss_fn(recon, xb)
                loss_recon = recon_weight * l1_weight * loss_l1 + perceptual_weight * loss_perc
                loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss_recon + beta_t * loss_kl
                val_loss_sum += loss.item()
                val_l1_sum += loss_l1.item()
                val_perc_sum += loss_perc.item()
                val_recon_sum += loss_recon.item()
                val_kl_sum += loss_kl.item()
                val_count += 1

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "train_l1_loss": train_l1_sum / max(train_count, 1),
            "train_perceptual_loss": train_perc_sum / max(train_count, 1),
            "train_recon_loss": train_recon_sum / max(train_count, 1),
            "train_kl_loss": train_kl_sum / max(train_count, 1),
            "val_loss": val_loss_sum / max(val_count, 1),
            "val_l1_loss": val_l1_sum / max(val_count, 1),
            "val_perceptual_loss": val_perc_sum / max(val_count, 1),
            "val_recon_loss": val_recon_sum / max(val_count, 1),
            "val_kl_loss": val_kl_sum / max(val_count, 1),
            "beta_kl": beta_t,
        })
        cur = epoch_logs[-1]
        if cur["val_loss"] < best_val_loss - early_stopping_min_delta:
            best_val_loss = cur["val_loss"]
            best_epoch = epoch
            best_state = deepcopy(vae.state_dict())
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"{log_prefix} {model_name} epoch {epoch:>3d}/{epochs}: "
                f"train_loss={cur['train_loss']:.4f}  "
                f"train_l1_loss={cur['train_l1_loss']:.4f}  "
                f"train_perceptual_loss={cur['train_perceptual_loss']:.4f}  "
                f"train_recon_loss={cur['train_recon_loss']:.4f}  "
                f"train_kl_loss={cur['train_kl_loss']:.4f}  "
                f"val_loss={cur['val_loss']:.4f}  "
                f"val_l1_loss={cur['val_l1_loss']:.4f}  "
                f"val_perceptual_loss={cur['val_perceptual_loss']:.4f}  "
                f"val_recon_loss={cur['val_recon_loss']:.4f}  "
                f"val_kl_loss={cur['val_kl_loss']:.4f}  "
                f"beta_kl={cur['beta_kl']:.4f}"
            )
        if patience_left <= 0:
            print(f"{log_prefix} {model_name} early stopping at epoch {epoch} (best val_loss at epoch {best_epoch})")
            break

    vae.load_state_dict(best_state)
    vae.eval()
    latent_pool = _collect_cvae_latent_pool(
        vae=vae,
        images=train_images,
        labels=train_local_labels,
        num_classes=len(class_ids),
        batch_size=batch_size,
        device=device,
    )
    return {
        "model": vae,
        "class_ids": list(class_ids),
        "image_size": image_size,
        "latent_pool": latent_pool,
        "latent_pool_noise_std": float(latent_pool_noise_std),
        "recon_weight": float(recon_weight),
        "l1_weight": float(l1_weight),
        "perceptual_weight": float(perceptual_weight),
        "beta_kl": float(beta_kl),
        "epoch_logs": epoch_logs,
    }


def train_feature_vae(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    class_ids: List[int],
    input_dim: int,
    h_dim: int,
    z_dim: int,
    epochs: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    lr: float,
    weight_decay: float,
    beta_kl: float,
    kl_warmup_epochs: int,
    recon_weight: float,
    batch_size: int,
    device: torch.device,
    log_prefix: str = "",
    model_name: str = "fVAE",
) -> Dict:
    if int(train_features.size(0)) <= 0:
        raise ValueError("train_features is empty")
    if int(val_features.size(0)) <= 0:
        raise ValueError("val_features is empty")

    g2l = {g: l for l, g in enumerate(class_ids)}
    train_local_labels = torch.tensor([g2l[int(l.item())] for l in train_labels], dtype=torch.long)
    val_local_labels = torch.tensor([g2l[int(l.item())] for l in val_labels], dtype=torch.long)

    train_features = train_features.float()
    val_features = val_features.float()
    if train_features.shape[1] != int(input_dim):
        raise ValueError(f"feature dim mismatch, expected {int(input_dim)} got {train_features.shape[1]}")

    vae = FeatureVAE(
        input_dim=int(input_dim),
        h_dim=int(h_dim),
        z_dim=int(z_dim),
    ).to(device)

    train_loader = DataLoader(TensorDataset(train_features, train_local_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_features, val_local_labels), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_logs = []
    best_state = deepcopy(vae.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = early_stopping_patience
    for epoch in range(1, epochs + 1):
        vae.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        train_count = 0
        beta_t = beta_kl if kl_warmup_epochs <= 0 else beta_kl * min(1.0, epoch / float(kl_warmup_epochs))
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            recon, mu, logvar, _ = vae(xb)
            loss_recon = F.mse_loss(recon, xb, reduction="mean")
            loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_weight * loss_recon + beta_t * loss_kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_recon_sum += loss_recon.item()
            train_kl_sum += loss_kl.item()
            train_count += 1

        vae.eval()
        val_loss_sum = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon, mu, logvar, _ = vae(xb)
                loss_recon = F.mse_loss(recon, xb, reduction="mean")
                loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_weight * loss_recon + beta_t * loss_kl
                val_loss_sum += loss.item()
                val_recon_sum += loss_recon.item()
                val_kl_sum += loss_kl.item()
                val_count += 1

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "train_recon_loss": train_recon_sum / max(train_count, 1),
            "train_kl_loss": train_kl_sum / max(train_count, 1),
            "val_loss": val_loss_sum / max(val_count, 1),
            "val_recon_loss": val_recon_sum / max(val_count, 1),
            "val_kl_loss": val_kl_sum / max(val_count, 1),
            "beta_kl": beta_t,
        })
        cur = epoch_logs[-1]
        if cur["val_loss"] < best_val_loss - early_stopping_min_delta:
            best_val_loss = cur["val_loss"]
            best_epoch = epoch
            best_state = deepcopy(vae.state_dict())
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"{log_prefix} {model_name} epoch {epoch:>3d}/{epochs}: "
                f"train_loss={cur['train_loss']:.4f}  "
                f"train_recon_loss={cur['train_recon_loss']:.4f}  "
                f"train_kl_loss={cur['train_kl_loss']:.4f}  "
                f"val_loss={cur['val_loss']:.4f}  "
                f"val_recon_loss={cur['val_recon_loss']:.4f}  "
                f"val_kl_loss={cur['val_kl_loss']:.4f}  "
                f"beta_kl={cur['beta_kl']:.4f}"
            )
        if patience_left <= 0:
            print(f"{log_prefix} {model_name} early stopping at epoch {epoch} (best val_loss at epoch {best_epoch})")
            break

    vae.load_state_dict(best_state)
    vae.eval()
    return {
        "model": vae,
        "class_ids": list(class_ids),
        "input_dim": int(input_dim),
        "h_dim": int(h_dim),
        "z_dim": int(z_dim),
        "recon_weight": float(recon_weight),
        "beta_kl": float(beta_kl),
        "type": "fvae",
        "epoch_logs": epoch_logs,
    }


def train_feature_cvae(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    class_ids: List[int],
    input_dim: int,
    h_dim: int,
    z_dim: int,
    epochs: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    lr: float,
    weight_decay: float,
    beta_kl: float,
    kl_warmup_epochs: int,
    recon_weight: float,
    latent_pool_noise_std: float,
    batch_size: int,
    device: torch.device,
    log_prefix: str = "",
    model_name: str = "fCVAE",
) -> Dict:
    if int(train_features.size(0)) <= 0:
        raise ValueError("train_features is empty")
    if int(val_features.size(0)) <= 0:
        raise ValueError("val_features is empty")

    g2l = {g: l for l, g in enumerate(class_ids)}
    train_local_labels = torch.tensor([g2l[int(l.item())] for l in train_labels], dtype=torch.long)
    val_local_labels = torch.tensor([g2l[int(l.item())] for l in val_labels], dtype=torch.long)

    train_features = train_features.float()
    val_features = val_features.float()
    if train_features.shape[1] != int(input_dim):
        raise ValueError(f"feature dim mismatch, expected {int(input_dim)} got {train_features.shape[1]}")

    vae = FeatureConditionalVAE(
        input_dim=int(input_dim),
        num_classes=len(class_ids),
        h_dim=int(h_dim),
        z_dim=int(z_dim),
    ).to(device)

    train_loader = DataLoader(TensorDataset(train_features, train_local_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_features, val_local_labels), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_logs = []
    best_state = deepcopy(vae.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = early_stopping_patience
    for epoch in range(1, epochs + 1):
        vae.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        train_count = 0
        beta_t = beta_kl if kl_warmup_epochs <= 0 else beta_kl * min(1.0, epoch / float(kl_warmup_epochs))
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            recon, mu, logvar, _ = vae(xb, yb)
            loss_recon = F.mse_loss(recon, xb, reduction="mean")
            loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_weight * loss_recon + beta_t * loss_kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_recon_sum += loss_recon.item()
            train_kl_sum += loss_kl.item()
            train_count += 1

        vae.eval()
        val_loss_sum = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon, mu, logvar, _ = vae(xb, yb)
                loss_recon = F.mse_loss(recon, xb, reduction="mean")
                loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_weight * loss_recon + beta_t * loss_kl
                val_loss_sum += loss.item()
                val_recon_sum += loss_recon.item()
                val_kl_sum += loss_kl.item()
                val_count += 1

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "train_recon_loss": train_recon_sum / max(train_count, 1),
            "train_kl_loss": train_kl_sum / max(train_count, 1),
            "val_loss": val_loss_sum / max(val_count, 1),
            "val_recon_loss": val_recon_sum / max(val_count, 1),
            "val_kl_loss": val_kl_sum / max(val_count, 1),
            "beta_kl": beta_t,
        })
        cur = epoch_logs[-1]
        if cur["val_loss"] < best_val_loss - early_stopping_min_delta:
            best_val_loss = cur["val_loss"]
            best_epoch = epoch
            best_state = deepcopy(vae.state_dict())
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"{log_prefix} {model_name} epoch {epoch:>3d}/{epochs}: "
                f"train_loss={cur['train_loss']:.4f}  "
                f"train_recon_loss={cur['train_recon_loss']:.4f}  "
                f"train_kl_loss={cur['train_kl_loss']:.4f}  "
                f"val_loss={cur['val_loss']:.4f}  "
                f"val_recon_loss={cur['val_recon_loss']:.4f}  "
                f"val_kl_loss={cur['val_kl_loss']:.4f}  "
                f"beta_kl={cur['beta_kl']:.4f}"
            )
        if patience_left <= 0:
            print(f"{log_prefix} {model_name} early stopping at epoch {epoch} (best val_loss at epoch {best_epoch})")
            break

    vae.load_state_dict(best_state)
    vae.eval()
    latent_pool = _collect_cvae_latent_pool(
        vae=vae,
        images=train_features,
        labels=train_local_labels,
        num_classes=len(class_ids),
        batch_size=batch_size,
        device=device,
    )
    return {
        "model": vae,
        "class_ids": list(class_ids),
        "input_dim": int(input_dim),
        "h_dim": int(h_dim),
        "z_dim": int(z_dim),
        "recon_weight": float(recon_weight),
        "beta_kl": float(beta_kl),
        "latent_pool": latent_pool,
        "latent_pool_noise_std": float(latent_pool_noise_std),
        "type": "fcvae",
        "epoch_logs": epoch_logs,
    }


def train_feature_vqvae(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    class_ids: List[int],
    input_dim: int,
    h_dim: int,
    embedding_dim: int,
    num_embeddings: int,
    commitment_cost: float,
    codebook_weight: float,
    ema_decay: float,
    epochs: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    lr: float,
    weight_decay: float,
    recon_weight: float,
    batch_size: int,
    device: torch.device,
    log_prefix: str = "",
    model_name: str = "fVQ-VAE",
) -> Dict:
    if int(train_features.size(0)) <= 0:
        raise ValueError("train_features is empty")
    if int(val_features.size(0)) <= 0:
        raise ValueError("val_features is empty")

    g2l = {g: l for l, g in enumerate(class_ids)}
    train_local_labels = torch.tensor([g2l[int(l.item())] for l in train_labels], dtype=torch.long)
    val_local_labels = torch.tensor([g2l[int(l.item())] for l in val_labels], dtype=torch.long)

    train_features = train_features.float()
    val_features = val_features.float()
    if train_features.shape[1] != int(input_dim):
        raise ValueError(f"feature dim mismatch, expected {int(input_dim)} got {train_features.shape[1]}")

    vqvae = FeatureConditionalVQVAE(
        input_dim=int(input_dim),
        num_classes=len(class_ids),
        embedding_dim=int(embedding_dim),
        num_embeddings=int(num_embeddings),
        h_dim=int(h_dim),
        commitment_cost=commitment_cost,
        ema_decay=ema_decay,
    ).to(device)

    train_loader = DataLoader(TensorDataset(train_features, train_local_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_features, val_local_labels), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_logs = []
    best_state = deepcopy(vqvae.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = early_stopping_patience
    for epoch in range(1, epochs + 1):
        vqvae.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_vq_sum = 0.0
        train_perplexity_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            recon, vq_loss, _, perplexity = vqvae(xb, yb)
            loss_recon = F.mse_loss(recon, xb)
            loss = recon_weight * loss_recon + codebook_weight * vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_recon_sum += loss_recon.item()
            train_vq_sum += vq_loss.item()
            train_perplexity_sum += float(perplexity.item())
            train_count += 1

        vqvae.eval()
        val_loss_sum = 0.0
        val_recon_sum = 0.0
        val_vq_sum = 0.0
        val_perplexity_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon, vq_loss, _, perplexity = vqvae(xb, yb)
                loss_recon = F.mse_loss(recon, xb)
                loss = recon_weight * loss_recon + codebook_weight * vq_loss
                val_loss_sum += loss.item()
                val_recon_sum += loss_recon.item()
                val_vq_sum += vq_loss.item()
                val_perplexity_sum += float(perplexity.item())
                val_count += 1

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "train_recon_loss": train_recon_sum / max(train_count, 1),
            "train_vq_loss": train_vq_sum / max(train_count, 1),
            "train_perplexity": train_perplexity_sum / max(train_count, 1),
            "val_loss": val_loss_sum / max(val_count, 1),
            "val_recon_loss": val_recon_sum / max(val_count, 1),
            "val_vq_loss": val_vq_sum / max(val_count, 1),
            "val_perplexity": val_perplexity_sum / max(val_count, 1),
        })
        cur = epoch_logs[-1]
        if cur["val_loss"] < best_val_loss - early_stopping_min_delta:
            best_val_loss = cur["val_loss"]
            best_epoch = epoch
            best_state = deepcopy(vqvae.state_dict())
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"{log_prefix} {model_name} epoch {epoch:>3d}/{epochs}: "
                f"train_loss={cur['train_loss']:.4f}  "
                f"train_recon_loss={cur['train_recon_loss']:.4f}  "
                f"train_vq_loss={cur['train_vq_loss']:.4f}  "
                f"train_perplexity={cur['train_perplexity']:.4f}  "
                f"val_loss={cur['val_loss']:.4f}  "
                f"val_recon_loss={cur['val_recon_loss']:.4f}  "
                f"val_vq_loss={cur['val_vq_loss']:.4f}  "
                f"val_perplexity={cur['val_perplexity']:.4f}"
            )
        if patience_left <= 0:
            print(f"{log_prefix} {model_name} early stopping at epoch {epoch} (best val_loss at epoch {best_epoch})")
            break

    vqvae.load_state_dict(best_state)
    vqvae.eval()
    code_index_pool = _collect_vqvae_code_pool(
        vqvae=vqvae,
        images=train_features,
        labels=train_local_labels,
        num_classes=len(class_ids),
        batch_size=batch_size,
        device=device,
    )
    return {
        "model": vqvae,
        "class_ids": list(class_ids),
        "input_dim": int(input_dim),
        "h_dim": int(h_dim),
        "embedding_dim": int(embedding_dim),
        "num_embeddings": int(num_embeddings),
        "recon_weight": float(recon_weight),
        "codebook_weight": float(codebook_weight),
        "commitment_cost": float(commitment_cost),
        "ema_decay": float(ema_decay),
        "code_index_pool": code_index_pool,
        "type": "fvqvae",
        "epoch_logs": epoch_logs,
    }


@torch.no_grad()
def generate_pseudo_images(
    vae_state: Dict,
    num_per_class: int,
    device: torch.device,
    target_image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vae = vae_state["model"]
    class_ids = vae_state["class_ids"]
    latent_pool = vae_state.get("latent_pool")
    latent_pool_noise_std = float(vae_state.get("latent_pool_noise_std", 0.0))
    images_all, labels_all = [], []
    for local_idx, global_cls in enumerate(class_ids):
        y_local = torch.full((num_per_class,), local_idx, dtype=torch.long, device=device)
        imgs = vae.sample(
            y_local,
            device=device,
            latent_pool=latent_pool,
            latent_noise_std=latent_pool_noise_std,
        ).cpu()
        imgs = _resize_images(imgs, target_image_size)
        imgs = _normalize_imagenet(imgs)
        labels = torch.full((num_per_class,), global_cls, dtype=torch.long)
        images_all.append(imgs)
        labels_all.append(labels)
    return torch.cat(images_all), torch.cat(labels_all)


def train_vqvae(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    class_ids: List[int],
    image_size: int,
    embedding_dim: int,
    num_embeddings: int,
    base_channels: int,
    channel_multipliers: List[int],
    commitment_cost: float,
    codebook_weight: float,
    ema_decay: float,
    epochs: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    lr: float,
    weight_decay: float,
    recon_weight: float,
    perceptual_weight: float,
    perceptual_pretrained: bool,
    perceptual_layers: int,
    use_pixelcnn_prior: bool,
    pixelcnn_hidden_channels: int,
    pixelcnn_num_layers: int,
    pixelcnn_kernel_size: int,
    pixelcnn_dropout: float,
    pixelcnn_epochs: int,
    pixelcnn_early_stopping_patience: int,
    pixelcnn_early_stopping_min_delta: float,
    pixelcnn_lr: float,
    pixelcnn_weight_decay: float,
    pixelcnn_batch_size: int,
    pixelcnn_sampling_temperature: float,
    pixelcnn_sampling_top_k: int,
    batch_size: int,
    device: torch.device,
    log_prefix: str = "",
) -> Dict:
    g2l = {g: l for l, g in enumerate(class_ids)}
    train_local_labels = torch.tensor([g2l[int(l.item())] for l in train_labels], dtype=torch.long)
    val_local_labels = torch.tensor([g2l[int(l.item())] for l in val_labels], dtype=torch.long)
    train_images = _denormalize_imagenet(train_images.float())
    val_images = _denormalize_imagenet(val_images.float())
    vqvae = TaskReplayVQVAE(
        image_size=image_size,
        num_classes=len(class_ids),
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        commitment_cost=commitment_cost,
        ema_decay=ema_decay,
    ).to(device)
    train_loader = DataLoader(TensorDataset(train_images, train_local_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_local_labels), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=lr, weight_decay=weight_decay)
    perceptual_loss_fn = None
    if perceptual_weight > 0.0:
        perceptual_loss_fn = PerceptualLoss(
            pretrained=perceptual_pretrained,
            max_stage=perceptual_layers,
        ).to(device)
        perceptual_loss_fn.eval()

    epoch_logs = []
    best_state = deepcopy(vqvae.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = early_stopping_patience
    for epoch in range(1, epochs + 1):
        vqvae.train()
        train_loss_sum = 0.0
        train_l1_sum = 0.0
        train_perc_sum = 0.0
        train_vq_sum = 0.0
        train_perplexity_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            recon, vq_loss, _, perplexity = vqvae(xb, yb)
            loss_l1 = F.l1_loss(recon, xb)
            loss_perc = xb.new_tensor(0.0)
            if perceptual_loss_fn is not None:
                loss_perc = perceptual_loss_fn(recon, xb)
            loss_recon = recon_weight * loss_l1 + perceptual_weight * loss_perc
            loss = loss_recon + codebook_weight * vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_l1_sum += loss_l1.item()
            train_perc_sum += loss_perc.item()
            train_vq_sum += vq_loss.item()
            train_perplexity_sum += float(perplexity.item())
            train_count += 1

        vqvae.eval()
        val_loss_sum = 0.0
        val_l1_sum = 0.0
        val_perc_sum = 0.0
        val_vq_sum = 0.0
        val_perplexity_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon, vq_loss, _, perplexity = vqvae(xb, yb)
                loss_l1 = F.l1_loss(recon, xb)
                loss_perc = xb.new_tensor(0.0)
                if perceptual_loss_fn is not None:
                    loss_perc = perceptual_loss_fn(recon, xb)
                loss_recon = recon_weight * loss_l1 + perceptual_weight * loss_perc
                loss = loss_recon + codebook_weight * vq_loss
                val_loss_sum += loss.item()
                val_l1_sum += loss_l1.item()
                val_perc_sum += loss_perc.item()
                val_vq_sum += vq_loss.item()
                val_perplexity_sum += float(perplexity.item())
                val_count += 1

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "train_l1_loss": train_l1_sum / max(train_count, 1),
            "train_perceptual_loss": train_perc_sum / max(train_count, 1),
            "train_vq_loss": train_vq_sum / max(train_count, 1),
            "train_perplexity": train_perplexity_sum / max(train_count, 1),
            "val_loss": val_loss_sum / max(val_count, 1),
            "val_l1_loss": val_l1_sum / max(val_count, 1),
            "val_perceptual_loss": val_perc_sum / max(val_count, 1),
            "val_vq_loss": val_vq_sum / max(val_count, 1),
            "val_perplexity": val_perplexity_sum / max(val_count, 1),
        })
        cur = epoch_logs[-1]
        if cur["val_loss"] < best_val_loss - early_stopping_min_delta:
            best_val_loss = cur["val_loss"]
            best_epoch = epoch
            best_state = deepcopy(vqvae.state_dict())
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"{log_prefix} VQ-VAE epoch {epoch:>3d}/{epochs}: "
                f"train_loss={cur['train_loss']:.4f}  "
                f"train_l1_loss={cur['train_l1_loss']:.4f}  "
                f"train_perceptual_loss={cur['train_perceptual_loss']:.4f}  "
                f"train_vq_loss={cur['train_vq_loss']:.4f}  "
                f"train_perplexity={cur['train_perplexity']:.4f}  "
                f"val_loss={cur['val_loss']:.4f}  "
                f"val_l1_loss={cur['val_l1_loss']:.4f}  "
                f"val_perceptual_loss={cur['val_perceptual_loss']:.4f}  "
                f"val_vq_loss={cur['val_vq_loss']:.4f}  "
                f"val_perplexity={cur['val_perplexity']:.4f}"
            )
        if patience_left <= 0:
            print(f"{log_prefix} VQ-VAE early stopping at epoch {epoch} (best val_loss at epoch {best_epoch})")
            break

    vqvae.load_state_dict(best_state)
    vqvae.eval()
    train_code_indices = _collect_vqvae_code_pool(
        vqvae=vqvae,
        images=train_images,
        labels=train_local_labels,
        num_classes=len(class_ids),
        batch_size=batch_size,
        device=device,
    )
    val_code_indices = _collect_vqvae_code_pool(
        vqvae=vqvae,
        images=val_images,
        labels=val_local_labels,
        num_classes=len(class_ids),
        batch_size=batch_size,
        device=device,
    )
    code_index_pool = _collect_vqvae_code_pool(
        vqvae=vqvae,
        images=train_images,
        labels=train_local_labels,
        num_classes=len(class_ids),
        batch_size=batch_size,
        device=device,
    )
    code_height, code_width = code_index_pool[0].shape[-2], code_index_pool[0].shape[-1]
    pixelcnn_prior = None
    pixelcnn_epoch_logs = []
    if use_pixelcnn_prior:
        train_code_all = torch.cat(train_code_indices, dim=0)
        train_code_labels = torch.cat(
            [torch.full((codes.shape[0],), cls_id, dtype=torch.long) for cls_id, codes in enumerate(train_code_indices)],
            dim=0,
        )
        val_code_all = torch.cat(val_code_indices, dim=0)
        val_code_labels = torch.cat(
            [torch.full((codes.shape[0],), cls_id, dtype=torch.long) for cls_id, codes in enumerate(val_code_indices)],
            dim=0,
        )
        pixelcnn_prior, pixelcnn_epoch_logs = train_vqvae_pixelcnn_prior(
            train_code_indices=train_code_all,
            train_labels=train_code_labels,
            val_code_indices=val_code_all,
            val_labels=val_code_labels,
            num_classes=len(class_ids),
            num_embeddings=num_embeddings,
            hidden_channels=pixelcnn_hidden_channels,
            num_layers=pixelcnn_num_layers,
            kernel_size=pixelcnn_kernel_size,
            dropout=pixelcnn_dropout,
            epochs=pixelcnn_epochs,
            early_stopping_patience=pixelcnn_early_stopping_patience,
            early_stopping_min_delta=pixelcnn_early_stopping_min_delta,
            lr=pixelcnn_lr,
            weight_decay=pixelcnn_weight_decay,
            batch_size=pixelcnn_batch_size,
            device=device,
            log_prefix=f"{log_prefix}[prior]",
        )
    return {
        "model": vqvae,
        "class_ids": list(class_ids),
        "image_size": image_size,
        "code_index_pool": code_index_pool,
        "code_height": int(code_height),
        "code_width": int(code_width),
        "epoch_logs": epoch_logs,
        "pixelcnn_prior": pixelcnn_prior,
        "pixelcnn_epoch_logs": pixelcnn_epoch_logs,
        "use_pixelcnn_prior": bool(use_pixelcnn_prior),
        "pixelcnn_sampling_temperature": float(pixelcnn_sampling_temperature),
        "pixelcnn_sampling_top_k": int(pixelcnn_sampling_top_k),
    }


@torch.no_grad()
def generate_vqvae_pseudo_images(
    vqvae_state: Dict,
    num_per_class: int,
    device: torch.device,
    target_image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vqvae = vqvae_state["model"]
    class_ids = vqvae_state["class_ids"]
    code_index_pool = vqvae_state["code_index_pool"]
    pixelcnn_prior = vqvae_state.get("pixelcnn_prior")
    use_pixelcnn_prior = bool(vqvae_state.get("use_pixelcnn_prior", False)) and pixelcnn_prior is not None
    sample_temperature = float(vqvae_state.get("pixelcnn_sampling_temperature", 1.0))
    sample_top_k = int(vqvae_state.get("pixelcnn_sampling_top_k", 0))
    code_height = int(vqvae_state["code_height"])
    code_width = int(vqvae_state["code_width"])
    images_all, labels_all = [], []
    for local_idx, global_cls in enumerate(class_ids):
        y_local = torch.full((num_per_class,), local_idx, dtype=torch.long, device=device)
        if use_pixelcnn_prior:
            sampled_indices = pixelcnn_prior.sample(
                labels=y_local,
                height=code_height,
                width=code_width,
                device=device,
                temperature=sample_temperature,
                top_k=sample_top_k,
            )
        else:
            pool = code_index_pool[local_idx]
            if pool.numel() == 0:
                raise ValueError(f"VQ-VAE code pool for local class {local_idx} is empty")
            choice = torch.randint(pool.shape[0], (num_per_class,))
            sampled_indices = pool[choice].to(device)
        imgs = vqvae.decode_indices(sampled_indices, y_local).cpu()
        imgs = _resize_images(imgs, target_image_size)
        imgs = _normalize_imagenet(imgs)
        labels = torch.full((num_per_class,), global_cls, dtype=torch.long)
        images_all.append(imgs)
        labels_all.append(labels)
    return torch.cat(images_all), torch.cat(labels_all)


def train_task_id_classifier(
    train_sets: List[Tuple],
    val_sets: List[Tuple],
    num_tasks: int,
    pretrained: bool,
    lr: float,
    batch_size: int,
    epochs: int,
    device: torch.device,
    early_stopping_patience: Any,
    early_stopping_min_delta: float,
    weight_decay: float = 1e-4,
    use_cosine_scheduler: bool = True,
    log_prefix: str = "",
    contrastive_cfg: Optional[Dict[str, Any]] = None,
    margin_loss_cfg: Optional[Dict[str, Any]] = None,
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
):
    def _resolve_optional_patience(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            key = value.strip().lower()
            if key in {"none", "off", "disable", "disabled"}:
                return None
        patience = int(value)
        if patience <= 0:
            return None
        return patience

    def _load_taskid_init_state(clf_local: TaskIDClassifier, state_dict: Dict[str, torch.Tensor]) -> None:
        current_state = clf_local.state_dict()
        loaded = {}
        for key, value in state_dict.items():
            if key not in current_state:
                continue
            if key in {"fc_out.weight", "fc_out.bias"}:
                continue
            if current_state[key].shape == value.shape:
                loaded[key] = value
        current_state.update(loaded)
        clf_local.load_state_dict(current_state, strict=False)

        with torch.no_grad():
            if "fc_out.weight" in state_dict:
                old_weight = state_dict["fc_out.weight"]
                new_weight = clf_local.fc_out.weight.data
                copy_rows = min(old_weight.shape[0], new_weight.shape[0])
                copy_cols = min(old_weight.shape[1], new_weight.shape[1])
                new_weight[:copy_rows, :copy_cols].copy_(old_weight[:copy_rows, :copy_cols])
            if clf_local.fc_out.bias is not None and "fc_out.bias" in state_dict:
                old_bias = state_dict["fc_out.bias"]
                new_bias = clf_local.fc_out.bias.data
                copy_rows = min(old_bias.shape[0], new_bias.shape[0])
                new_bias[:copy_rows].copy_(old_bias[:copy_rows])

    def _supervised_contrastive_loss(
        embeds: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        if embeds.ndim != 2 or embeds.shape[0] <= 1:
            return embeds.new_zeros(())
        temp = max(1e-6, float(temperature))
        z = F.normalize(embeds, dim=1)
        logits = (z @ z.t()) / temp
        logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()
        device_local = embeds.device
        batch_size_local = embeds.shape[0]
        self_mask = torch.eye(batch_size_local, device=device_local, dtype=torch.bool)
        positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~self_mask)
        valid_mask = positive_mask.any(dim=1)
        if not bool(valid_mask.any()):
            return embeds.new_zeros(())
        logits_mask = ~self_mask
        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        pos_log_prob = (positive_mask.float() * log_prob).sum(dim=1) / positive_mask.float().sum(dim=1).clamp_min(1.0)
        return -pos_log_prob[valid_mask].mean()

    def _weighted_supervised_contrastive_loss(
        embeds: torch.Tensor,
        positive_weights: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        if embeds.ndim != 2 or embeds.shape[0] <= 1:
            return embeds.new_zeros(())
        if positive_weights.ndim != 2 or positive_weights.shape[0] != embeds.shape[0]:
            raise ValueError("positive_weights must be a square matrix aligned with embeds")
        temp = max(1e-6, float(temperature))
        z = F.normalize(embeds, dim=1)
        logits = (z @ z.t()) / temp
        logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()
        device_local = embeds.device
        batch_size_local = embeds.shape[0]
        self_mask = torch.eye(batch_size_local, device=device_local, dtype=torch.bool)
        pos_weights = positive_weights.to(device=device_local, dtype=embeds.dtype).clone()
        pos_weights.masked_fill_(self_mask, 0.0)
        pos_weight_sum = pos_weights.sum(dim=1)
        valid_mask = pos_weight_sum > 0
        if not bool(valid_mask.any()):
            return embeds.new_zeros(())
        logits_mask = ~self_mask
        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        pos_log_prob = (pos_weights * log_prob).sum(dim=1) / pos_weight_sum.clamp_min(1e-12)
        return -pos_log_prob[valid_mask].mean()

    def _taskid_contrastive_loss(
        embeds: torch.Tensor,
        task_labels: torch.Tensor,
        class_labels: Optional[torch.Tensor],
        cfg: Dict[str, Any],
    ) -> torch.Tensor:
        mode = str(cfg.get("mode", "supervised")).lower()
        temperature = float(cfg.get("temperature", 0.1))
        if mode == "supervised":
            return _supervised_contrastive_loss(embeds, task_labels, temperature=temperature)
        if mode != "hierarchical":
            raise ValueError(
                "taskid_contrastive.mode must be one of ['supervised','hierarchical'], "
                f"got: {mode}"
            )
        same_class_weight = max(0.0, float(cfg.get("same_class_weight", 1.0)))
        same_task_diff_class_weight = max(0.0, float(cfg.get("same_task_diff_class_weight", 0.25)))
        device_local = embeds.device
        batch_size_local = embeds.shape[0]
        self_mask = torch.eye(batch_size_local, device=device_local, dtype=torch.bool)
        same_task = task_labels.unsqueeze(0).eq(task_labels.unsqueeze(1))
        pos_weights = torch.zeros((batch_size_local, batch_size_local), dtype=embeds.dtype, device=device_local)
        weak_mask = same_task & (~self_mask)
        if same_task_diff_class_weight > 0.0:
            pos_weights[weak_mask] = same_task_diff_class_weight
        if class_labels is not None:
            class_labels = class_labels.to(device_local)
            valid_class = class_labels.ge(0)
            same_class = class_labels.unsqueeze(0).eq(class_labels.unsqueeze(1))
            valid_same_class = (
                valid_class.unsqueeze(0)
                & valid_class.unsqueeze(1)
                & same_class
                & (~self_mask)
            )
            if same_class_weight > 0.0:
                pos_weights[valid_same_class] = same_class_weight
        return _weighted_supervised_contrastive_loss(embeds, pos_weights, temperature=temperature)

    def _taskid_margin_logits(
        clf_local: TaskIDClassifier,
        feats: torch.Tensor,
        labels: Optional[torch.Tensor],
        cfg: Dict[str, Any],
        apply_margin: bool,
    ) -> torch.Tensor:
        margin_type = str(cfg.get("type", "arcface")).lower()
        if margin_type not in ("arcface", "cosface"):
            raise ValueError(f"taskid_margin_loss.type must be one of ['arcface','cosface'], got: {margin_type}")
        scale = max(1e-6, float(cfg.get("scale", 30.0)))
        margin = float(cfg.get("margin", 0.35))
        feats_n = F.normalize(feats, dim=1)
        weight_n = F.normalize(clf_local.fc_out.weight, dim=1)
        cosine = feats_n @ weight_n.t()
        if (not apply_margin) or labels is None:
            return scale * cosine
        row_idx = torch.arange(cosine.shape[0], device=cosine.device)
        target_cos = cosine[row_idx, labels]
        if margin_type == "cosface":
            target_margin = target_cos - margin
        else:
            cos_m = math.cos(margin)
            sin_m = math.sin(margin)
            sin_theta = torch.sqrt((1.0 - target_cos.pow(2)).clamp_min(1e-7))
            target_margin = target_cos * cos_m - sin_theta * sin_m
        logits = cosine.clone()
        logits[row_idx, labels] = target_margin
        return scale * logits

    train_x, train_y = [], []
    train_y_class = []
    val_x, val_y = [], []
    val_y_class = []

    for item in train_sets:
        if len(item) == 2:
            images, tid = item
            class_labels = None
        elif len(item) == 3:
            images, tid, class_labels = item
        else:
            raise ValueError("train_sets items must be (images, tid) or (images, tid, class_labels)")
        if images.numel() == 0:
            continue
        train_x.append(images)
        train_y.append(torch.full((images.shape[0],), tid, dtype=torch.long))
        if class_labels is None or class_labels.numel() == 0:
            train_y_class.append(torch.full((images.shape[0],), -1, dtype=torch.long))
        else:
            train_y_class.append(class_labels.to(dtype=torch.long))

    for item in val_sets:
        if len(item) == 2:
            images, tid = item
            class_labels = None
        elif len(item) == 3:
            images, tid, class_labels = item
        else:
            raise ValueError("val_sets items must be (images, tid) or (images, tid, class_labels)")
        if images.numel() == 0:
            continue
        val_x.append(images)
        val_y.append(torch.full((images.shape[0],), tid, dtype=torch.long))
        if class_labels is None or class_labels.numel() == 0:
            val_y_class.append(torch.full((images.shape[0],), -1, dtype=torch.long))
        else:
            val_y_class.append(class_labels.to(dtype=torch.long))

    if not train_x:
        raise ValueError("task-id classifier received no training images")
    if not val_x:
        raise ValueError("task-id classifier received no validation images")

    train_x = torch.cat(train_x)
    train_y = torch.cat(train_y)
    train_y_class = torch.cat(train_y_class) if train_y_class else torch.full((train_y.shape[0],), -1, dtype=torch.long)
    val_x = torch.cat(val_x)
    val_y = torch.cat(val_y)
    val_y_class = torch.cat(val_y_class) if val_y_class else torch.full((val_y.shape[0],), -1, dtype=torch.long)

    contrastive_cfg = contrastive_cfg or {}
    contrastive_enabled = bool(contrastive_cfg.get("enabled", False))
    contrastive_lambda = float(contrastive_cfg.get("lambda", 0.2))
    contrastive_projection_dim = int(contrastive_cfg.get("projection_dim", 0))
    margin_loss_cfg = margin_loss_cfg or {}
    margin_loss_enabled = bool(margin_loss_cfg.get("enabled", False))

    clf = TaskIDClassifier(
        num_tasks=num_tasks,
        pretrained=pretrained,
        projection_dim=contrastive_projection_dim if contrastive_enabled else 0,
    ).to(device)
    if init_state_dict is not None:
        _load_taskid_init_state(clf, init_state_dict)
    train_loader = DataLoader(TensorDataset(train_x, train_y, train_y_class), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y, val_y_class), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if use_cosine_scheduler else None

    resolved_patience = _resolve_optional_patience(early_stopping_patience)
    epoch_logs = []
    best_state = deepcopy(clf.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = resolved_patience
    for epoch in range(1, epochs + 1):
        clf.train()
        train_loss_sum, train_ce_sum, train_contrastive_sum, train_correct, train_total = 0.0, 0.0, 0.0, 0, 0
        for xb, yb, yb_class in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yb_class = yb_class.to(device)
            feats = clf.encode(xb)
            if margin_loss_enabled:
                logits = _taskid_margin_logits(clf, feats, yb, margin_loss_cfg, apply_margin=True)
            else:
                logits = clf.classify_from_feats(feats)
            ce_loss = F.cross_entropy(logits, yb)
            contrastive_loss = torch.tensor(0.0, device=device)
            loss = ce_loss
            if contrastive_enabled:
                z = clf.project_from_feats(feats)
                contrastive_loss = _taskid_contrastive_loss(z, yb, yb_class, contrastive_cfg)
                loss = loss + contrastive_lambda * contrastive_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * yb.size(0)
            train_ce_sum += float(ce_loss.detach()) * yb.size(0)
            train_contrastive_sum += float(contrastive_loss.detach()) * yb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += yb.size(0)

        clf.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb, _yb_class in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                feats = clf.encode(xb)
                if margin_loss_enabled:
                    logits = _taskid_margin_logits(clf, feats, None, margin_loss_cfg, apply_margin=False)
                else:
                    logits = clf.classify_from_feats(feats)
                loss = F.cross_entropy(logits, yb)
                val_loss_sum += loss.item() * yb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += yb.size(0)

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_total, 1),
            "train_ce_loss": train_ce_sum / max(train_total, 1),
            "train_contrastive_loss": train_contrastive_sum / max(train_total, 1),
            "train_acc": train_correct / max(train_total, 1),
            "val_loss": val_loss_sum / max(val_total, 1),
            "val_acc": val_correct / max(val_total, 1),
        })
        cur = epoch_logs[-1]
        if cur["val_loss"] < best_val_loss - early_stopping_min_delta:
            best_val_loss = cur["val_loss"]
            best_epoch = epoch
            best_state = deepcopy(clf.state_dict())
            patience_left = resolved_patience
        elif patience_left is not None:
            patience_left -= 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"{log_prefix} TaskID epoch {epoch:>3d}/{epochs}: "
                f"train_loss={cur['train_loss']:.4f} "
                f"train_ce_loss={cur['train_ce_loss']:.4f} "
                f"train_contrastive_loss={cur['train_contrastive_loss']:.4f} "
                f"val_loss={cur['val_loss']:.4f} "
            )
        if scheduler is not None:
            scheduler.step()
        if patience_left is not None and patience_left <= 0:
            print(f"{log_prefix} TaskID early stopping at epoch {epoch} (best val_loss at epoch {best_epoch})")
            break

    clf.load_state_dict(best_state)
    clf.eval()
    return clf, epoch_logs


def train_feature_task_id_classifier(
    train_sets: List[Tuple],
    val_sets: List[Tuple],
    num_tasks: int,
    input_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    dropout: float,
    lr: float,
    batch_size: int,
    epochs: int,
    device: torch.device,
    early_stopping_patience: Any,
    early_stopping_min_delta: float,
    weight_decay: float = 1e-4,
    use_cosine_scheduler: bool = True,
    log_prefix: str = "",
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
):
    def _resolve_optional_patience(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            key = value.strip().lower()
            if key in {"none", "off", "disable", "disabled"}:
                return None
        patience = int(value)
        if patience <= 0:
            return None
        return patience

    def _load_feature_taskid_init_state(
        clf_local: FeatureTaskIDClassifier,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        current_state = clf_local.state_dict()
        loaded = {}
        for key, value in state_dict.items():
            if key not in current_state:
                continue
            if key in {"fc_out.weight", "fc_out.bias"}:
                continue
            if current_state[key].shape == value.shape:
                loaded[key] = value
        current_state.update(loaded)
        clf_local.load_state_dict(current_state, strict=False)

        with torch.no_grad():
            if "fc_out.weight" in state_dict:
                old_weight = state_dict["fc_out.weight"]
                new_weight = clf_local.fc_out.weight.data
                copy_rows = min(old_weight.shape[0], new_weight.shape[0])
                copy_cols = min(old_weight.shape[1], new_weight.shape[1])
                new_weight[:copy_rows, :copy_cols].copy_(old_weight[:copy_rows, :copy_cols])
            if clf_local.fc_out.bias is not None and "fc_out.bias" in state_dict:
                old_bias = state_dict["fc_out.bias"]
                new_bias = clf_local.fc_out.bias.data
                copy_rows = min(old_bias.shape[0], new_bias.shape[0])
                new_bias[:copy_rows].copy_(old_bias[:copy_rows])

    train_x, train_y = [], []
    val_x, val_y = [], []

    for item in train_sets:
        if len(item) < 2:
            raise ValueError("train_sets items must be (features, tid)")
        features, tid = item[0], item[1]
        if features.numel() == 0:
            continue
        train_x.append(features.float())
        train_y.append(torch.full((features.shape[0],), int(tid), dtype=torch.long))

    for item in val_sets:
        if len(item) < 2:
            raise ValueError("val_sets items must be (features, tid)")
        features, tid = item[0], item[1]
        if features.numel() == 0:
            continue
        val_x.append(features.float())
        val_y.append(torch.full((features.shape[0],), int(tid), dtype=torch.long))

    if not train_x:
        raise ValueError("feature task-id classifier received no training features")
    if not val_x:
        raise ValueError("feature task-id classifier received no validation features")

    train_x = torch.cat(train_x, dim=0)
    train_y = torch.cat(train_y, dim=0)
    val_x = torch.cat(val_x, dim=0)
    val_y = torch.cat(val_y, dim=0)

    clf = FeatureTaskIDClassifier(
        num_tasks=num_tasks,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
    ).to(device)
    if init_state_dict is not None:
        _load_feature_taskid_init_state(clf, init_state_dict)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if use_cosine_scheduler else None

    resolved_patience = _resolve_optional_patience(early_stopping_patience)
    epoch_logs = []
    best_state = deepcopy(clf.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = resolved_patience

    for epoch in range(1, epochs + 1):
        clf.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = clf(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * yb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += yb.size(0)

        clf.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = clf(xb)
                loss = F.cross_entropy(logits, yb)
                val_loss_sum += loss.item() * yb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += yb.size(0)

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_total, 1),
            "train_acc": train_correct / max(train_total, 1),
            "val_loss": val_loss_sum / max(val_total, 1),
            "val_acc": val_correct / max(val_total, 1),
        })
        cur = epoch_logs[-1]
        if cur["val_loss"] < best_val_loss - early_stopping_min_delta:
            best_val_loss = cur["val_loss"]
            best_epoch = epoch
            best_state = deepcopy(clf.state_dict())
            patience_left = resolved_patience
        elif patience_left is not None:
            patience_left -= 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"{log_prefix} FeatureTaskID epoch {epoch:>3d}/{epochs}: "
                f"train_loss={cur['train_loss']:.4f} "
                f"train_acc={cur['train_acc']:.4f} "
                f"val_loss={cur['val_loss']:.4f} "
                f"val_acc={cur['val_acc']:.4f}"
            )
        if scheduler is not None:
            scheduler.step()
        if patience_left is not None and patience_left <= 0:
            print(f"{log_prefix} FeatureTaskID early stopping at epoch {epoch} (best val_loss at epoch {best_epoch})")
            break

    clf.load_state_dict(best_state)
    clf.eval()
    return clf, epoch_logs


@torch.no_grad()
def collect_incremental_routing_stats(
    model: IncrementalMoEResNet,
    loader: DataLoader,
    task_id: int,
    device: torch.device,
) -> Dict[int, Dict[str, List[float]]]:
    per_class: Dict[int, Dict[str, List[torch.Tensor]]]= {}
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        layer_weights = {name: weights.cpu() for name, weights in model.routing_layer_weights(images, task_id).items()}
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


_TASK_ROUTER_INFERENCE_MODES = ("top1", "top2")


def _resolve_task_router_inference(mode: str) -> str:
    key = str(mode).strip().lower()
    if key not in _TASK_ROUTER_INFERENCE_MODES:
        raise ValueError(
            f"unsupported task_router_inference: {mode!r}; choose one of {_TASK_ROUTER_INFERENCE_MODES}"
        )
    return key


_INCREMENTAL_2_ALPHA_CACHE: Optional[float] = None
_INCREMENTAL_2_ROUTE_MISS_PENALTY = 5.0


def reset_incremental_2_alpha_cache() -> None:
    global _INCREMENTAL_2_ALPHA_CACHE
    _INCREMENTAL_2_ALPHA_CACHE = None


def get_incremental_2_alpha_cache() -> Optional[float]:
    return _INCREMENTAL_2_ALPHA_CACHE


def _set_incremental_2_alpha_cache(alpha: float) -> None:
    global _INCREMENTAL_2_ALPHA_CACHE
    _INCREMENTAL_2_ALPHA_CACHE = float(alpha)


def _task_class_scores(logits: torch.Tensor, class_score_mode: str) -> torch.Tensor:
    mode = str(class_score_mode).strip().lower()
    if mode == "raw":
        return logits
    if mode == "cardinality":
        return F.log_softmax(logits, dim=1) + math.log(max(int(logits.shape[1]), 1))
    if mode == "zscore":
        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (logits - mean) / std
    if mode == "softmax":
        return F.log_softmax(logits, dim=1)
    if mode == "sigmoid":
        return F.logsigmoid(logits)
    raise ValueError(
        f"unsupported task_router_class_score_mode: {class_score_mode!r}; "
        "choose raw, cardinality, zscore, softmax, or sigmoid"
    )


@torch.no_grad()
def _evaluate_incremental_2_alpha_loss(
    model: IncrementalMoEResNet,
    heads: List[nn.Module],
    task_id_classifier,
    val_loaders: List[DataLoader],
    task_splits: List[List[int]],
    eval_task_ids: Optional[List[int]],
    device: torch.device,
    taskid_image_size: int,
    router_alpha: float,
    class_prob_mode: str = "raw",
    task_router_inference: str = "top2",
    route_miss_penalty: float = _INCREMENTAL_2_ROUTE_MISS_PENALTY,
) -> float:
    model.eval()
    task_id_classifier.eval()
    num_seen = len(heads)
    if num_seen == 0:
        return 0.0
    if task_router_inference != "top2" or num_seen == 1:
        return 0.0

    total_loss = 0.0
    total = 0
    margin_threshold = float(router_alpha) / max(num_seen, 1)

    if eval_task_ids is None:
        eval_task_ids = list(range(len(val_loaders)))
    elif len(eval_task_ids) != len(val_loaders):
        raise ValueError("eval_task_ids and val_loaders must have the same length")

    for loader_idx, true_tid in enumerate(eval_task_ids):
        if true_tid < 0 or true_tid >= num_seen:
            raise ValueError(f"eval_task_ids contains invalid task id {true_tid} for num_seen={num_seen}")
        for images, labels in val_loaders[loader_idx]:
            images = images.to(device)
            labels = labels.to(device)
            task_inputs = _resize_images(images, taskid_image_size)
            task_logits = task_id_classifier(task_inputs)
            task_logits = task_logits[:, :num_seen]
            task_log_probs = F.log_softmax(task_logits, dim=1)
            task_probs = task_log_probs.exp()
            pred_tids_top1 = task_probs.argmax(1)
            topk = min(2, num_seen)
            top2_probs, top2_tids = task_probs.topk(topk, dim=1)
            p1 = top2_probs[:, 0]
            p2 = top2_probs[:, 1] if topk >= 2 else top2_probs[:, 0]
            ambiguous = (p1 - p2) < margin_threshold

            # confident samples: use the selected task head directly
            for tid in pred_tids_top1[~ambiguous].unique().tolist():
                tid = int(tid)
                mask = (~ambiguous) & (pred_tids_top1 == tid)
                if mask.sum() == 0:
                    continue
                sel_idx = mask.nonzero(as_tuple=False).flatten()
                logits = heads[tid](model(images[mask], tid))
                class_scores = _task_class_scores(logits, class_prob_mode)
                class_log_probs = F.log_softmax(class_scores, dim=1)
                for row, sample_idx in enumerate(sel_idx.tolist()):
                    true_cls = int(labels[sample_idx].item())
                    if true_cls in task_splits[tid]:
                        local_idx = task_splits[tid].index(true_cls)
                        loss_i = -float(class_log_probs[row, local_idx].item())
                    else:
                        loss_i = route_miss_penalty - float(task_log_probs[sample_idx, true_tid].item())
                    total_loss += loss_i
                    total += 1

            # ambiguous samples: compare top-2 candidate tasks with joint scores
            amb_idx = ambiguous.nonzero(as_tuple=False).flatten()
            for sample_idx in amb_idx.tolist():
                tid1 = int(top2_tids[sample_idx, 0].item())
                tid2 = int(top2_tids[sample_idx, 1].item()) if topk >= 2 else tid1
                true_cls = int(labels[sample_idx].item())
                true_pos = None
                cand_scores: List[float] = []
                for tid in (tid1, tid2) if tid2 != tid1 else (tid1,):
                    logits = heads[tid](model(images[sample_idx : sample_idx + 1], tid))
                    class_scores = _task_class_scores(logits, class_prob_mode)[0]
                    joint_task_score = float(task_log_probs[sample_idx, tid].item())
                    for local_i, global_cls in enumerate(task_splits[tid]):
                        cand_scores.append(joint_task_score + float(class_scores[local_i].item()))
                        if global_cls == true_cls and true_pos is None:
                            true_pos = len(cand_scores) - 1
                if true_pos is None:
                    loss_i = route_miss_penalty - float(task_log_probs[sample_idx, true_tid].item())
                else:
                    score_tensor = torch.tensor(cand_scores, device=device, dtype=torch.float32).unsqueeze(0)
                    loss_i = -float(F.log_softmax(score_tensor, dim=1)[0, true_pos].item())
                total_loss += loss_i
                total += 1

    return total_loss / max(total, 1)


def _search_incremental_2_task_router_alpha(
    model: IncrementalMoEResNet,
    heads: List[nn.Module],
    task_id_classifier,
    val_loaders: List[DataLoader],
    task_splits: List[List[int]],
    eval_task_ids: Optional[List[int]],
    device: torch.device,
    taskid_image_size: int,
    task_router_inference: str,
    class_prob_mode: str,
    default_alpha: float,
    search_trials: int,
    search_seed: int,
) -> Tuple[float, float]:
    if task_router_inference != "top2" or len(heads) <= 1:
        return float(default_alpha), 0.0

    n_trials = max(1, int(search_trials))
    lower, upper = 0.0, 1.0

    def _objective(alpha: float) -> float:
        return _evaluate_incremental_2_alpha_loss(
            model=model,
            heads=heads,
            task_id_classifier=task_id_classifier,
            val_loaders=val_loaders,
            task_splits=task_splits,
            eval_task_ids=eval_task_ids,
            device=device,
            taskid_image_size=taskid_image_size,
            router_alpha=float(alpha),
            class_prob_mode=class_prob_mode,
            task_router_inference=task_router_inference,
        )

    if optuna is not None:
        sampler = optuna.samplers.TPESampler(seed=int(search_seed))
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def _trial_objective(trial):
            alpha = trial.suggest_float("task_router_alpha", lower, upper)
            return _objective(alpha)

        study.optimize(_trial_objective, n_trials=n_trials)
        best_alpha = float(study.best_params["task_router_alpha"])
        best_loss = float(study.best_value)
        return best_alpha, best_loss

    print("[incre2] optuna is not installed; using random search fallback for task_router_alpha")
    rng = random.Random(int(search_seed))

    class _Trial:
        def __init__(self, alpha: float):
            self.alpha = float(alpha)
            self.params: Dict[str, float] = {}

        def suggest_float(self, name: str, low: float, high: float) -> float:
            if name != "task_router_alpha":
                raise ValueError(f"unsupported search parameter: {name}")
            self.params[name] = self.alpha
            return self.alpha

    best_alpha = float(default_alpha)
    best_loss = _objective(best_alpha)
    for _ in range(n_trials):
        alpha = rng.uniform(lower, upper)
        trial = _Trial(alpha)
        _ = trial.suggest_float("task_router_alpha", lower, upper)
        loss = _objective(trial.alpha)
        if loss < best_loss:
            best_alpha = float(alpha)
            best_loss = float(loss)
    return best_alpha, best_loss


@torch.no_grad()
def _defect_pred_top1_route(
    model: IncrementalMoEResNet,
    heads: List[nn.Module],
    images: torch.Tensor,
    labels: torch.Tensor,
    pred_tids: torch.Tensor,
    task_splits: List[List[int]],
    num_seen: int,
) -> Tuple[List[int], List[int]]:
    preds: List[int] = []
    gts: List[int] = []
    for tid in range(num_seen):
        mask = pred_tids == tid
        if mask.sum() == 0:
            continue
        logits = heads[tid](model(images[mask], tid))
        pred_local = logits.argmax(1)
        preds.extend(task_splits[tid][int(l.item())] for l in pred_local)
        gts.extend(labels[mask].cpu().tolist())
    return preds, gts


@torch.no_grad()
def _defect_pred_top2_adaptive(
    model: IncrementalMoEResNet,
    heads: List[nn.Module],
    task_id_classifier,
    images: torch.Tensor,
    labels: torch.Tensor,
    task_splits: List[List[int]],
    num_seen: int,
    taskid_image_size: int,
    router_alpha: float,
    class_prob_mode: str = "raw",
) -> Tuple[List[int], List[int], torch.Tensor, Dict[str, Any]]:
    """Returns preds, gts, pred_tids_top1, and batch top2 routing stats."""
    task_inputs = _resize_images(images, taskid_image_size)
    task_logits = task_id_classifier(task_inputs)
    task_logits = task_logits[:, :num_seen]
    task_log_probs = F.log_softmax(task_logits, dim=1)
    task_probs = task_log_probs.exp()
    pred_tids_top1 = task_probs.argmax(1)
    class_prob_mode = class_prob_mode.lower().strip()

    margin_threshold = float(router_alpha) / max(num_seen, 1)
    topk = min(2, num_seen)
    top2_probs, top2_tids = task_probs.topk(topk, dim=1)
    p1 = top2_probs[:, 0]
    p2 = top2_probs[:, 1] if topk >= 2 else top2_probs[:, 0]
    ambiguous = (p1 - p2) < margin_threshold
    dual_route_count = int(ambiguous.sum().item())
    batch_size = int(images.size(0))
    route_stats = {
        "batch_size": batch_size,
        "dual_route_count": dual_route_count,
        "single_route_count": batch_size - dual_route_count,
        "task_prob_sum": task_probs.sum(dim=0).detach().cpu(),
        "top1_prob_sum": float(p1.sum().item()),
        "top2_prob_sum": float(p2.sum().item()),
        "margin_sum": float((p1 - p2).sum().item()),
    }

    preds: List[int] = []
    gts: List[int] = []
    confident = ~ambiguous
    if confident.any():
        batch_preds, batch_gts = _defect_pred_top1_route(
            model, heads, images[confident], labels[confident], pred_tids_top1[confident], task_splits, num_seen
        )
        preds.extend(batch_preds)
        gts.extend(batch_gts)

    amb_idx = ambiguous.nonzero(as_tuple=False).flatten()
    for idx in amb_idx:
        img = images[idx : idx + 1]
        tid1 = int(top2_tids[idx, 0].item())
        tid2 = int(top2_tids[idx, 1].item()) if topk >= 2 else tid1
        best_score = float("-inf")
        best_global = int(task_splits[tid1][0])
        for tid in (tid1, tid2) if tid2 != tid1 else (tid1,):
            logits = heads[tid](model(img, tid))
            class_log_probs = _task_class_scores(logits, class_prob_mode)[0]
            joint_task_score = float(task_log_probs[idx, tid].item())
            for local_i, global_cls in enumerate(task_splits[tid]):
                score = joint_task_score + float(class_log_probs[local_i].item())
                if score > best_score:
                    best_score = score
                    best_global = int(global_cls)
        preds.append(best_global)
        gts.append(int(labels[idx].item()))

    return preds, gts, pred_tids_top1, route_stats


@torch.no_grad()
def evaluate_all(
    model: IncrementalMoEResNet,
    heads: List[nn.Module],
    task_id_classifier,
    test_loaders: List[DataLoader],
    task_splits: List[List[int]],
    class_names: List[str],
    device: torch.device,
    taskid_image_size: int,
    oracle_taskid: bool = False,
    task_router_inference: str = "top1",
    task_router_alpha: float = 0.4,
    task_router_class_prob_mode: str = "raw",
) -> Dict:
    model.eval()
    num_seen = len(heads)
    task_router_inference = _resolve_task_router_inference(task_router_inference)
    router_alpha = float(task_router_alpha)
    if not (0.0 <= router_alpha <= 1.0):
        raise ValueError(f"task_router_alpha must be in [0, 1], got {router_alpha}")
    class_prob_mode = task_router_class_prob_mode.lower().strip()
    if class_prob_mode not in {"raw", "cardinality", "zscore", "softmax", "sigmoid"}:
        raise ValueError(
            "task_router_class_score_mode must be one of "
            f"{{'raw', 'cardinality', 'zscore', 'softmax', 'sigmoid'}}, got {class_prob_mode!r}"
        )

    all_preds: List[int] = []
    all_labels: List[int] = []
    taskid_correct = 0
    all_total = 0
    dual_route_total = 0
    top2_debug_enabled = task_router_inference == "top2" and (not oracle_taskid) and num_seen > 1
    top2_task_prob_sums_by_true_tid = [
        torch.zeros(num_seen, dtype=torch.float64) for _ in range(num_seen)
    ] if top2_debug_enabled else []
    top2_count_by_true_tid = [0 for _ in range(num_seen)] if top2_debug_enabled else []
    top2_dual_by_true_tid = [0 for _ in range(num_seen)] if top2_debug_enabled else []
    top2_top1_prob_sum_total = 0.0
    top2_top2_prob_sum_total = 0.0
    top2_margin_sum_total = 0.0

    for true_tid in range(num_seen):
        for images, labels in test_loaders[true_tid]:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            if num_seen == 1 or oracle_taskid:
                pred_tids = torch.full((batch_size,), true_tid, dtype=torch.long, device=device)
                batch_preds, batch_gts = _defect_pred_top1_route(
                    model, heads, images, labels, pred_tids, task_splits, num_seen
                )
                all_preds.extend(batch_preds)
                all_labels.extend(batch_gts)
                if not oracle_taskid:
                    taskid_correct += (pred_tids == true_tid).sum().item()
                else:
                    taskid_correct += batch_size
            elif task_router_inference == "top1":
                task_inputs = _resize_images(images, taskid_image_size)
                task_logits = task_id_classifier(task_inputs)[:, :num_seen]
                pred_tids = task_logits.argmax(1)
                taskid_correct += (pred_tids == true_tid).sum().item()
                batch_preds, batch_gts = _defect_pred_top1_route(
                    model, heads, images, labels, pred_tids, task_splits, num_seen
                )
                all_preds.extend(batch_preds)
                all_labels.extend(batch_gts)
            else:
                batch_preds, batch_gts, pred_tids_top1, route_stats = _defect_pred_top2_adaptive(
                    model,
                    heads,
                    task_id_classifier,
                    images,
                    labels,
                    task_splits,
                    num_seen,
                    taskid_image_size,
                    router_alpha,
                    class_prob_mode,
                )
                taskid_correct += (pred_tids_top1 == true_tid).sum().item()
                all_preds.extend(batch_preds)
                all_labels.extend(batch_gts)
                dual_n = int(route_stats.get("dual_route_count", 0))
                dual_route_total += dual_n
                if top2_debug_enabled:
                    top2_count_by_true_tid[true_tid] += int(route_stats.get("batch_size", batch_size))
                    top2_dual_by_true_tid[true_tid] += dual_n
                    task_prob_sum = route_stats.get("task_prob_sum")
                    if isinstance(task_prob_sum, torch.Tensor):
                        top2_task_prob_sums_by_true_tid[true_tid] += task_prob_sum.to(dtype=torch.float64)
                    top2_top1_prob_sum_total += float(route_stats.get("top1_prob_sum", 0.0))
                    top2_top2_prob_sum_total += float(route_stats.get("top2_prob_sum", 0.0))
                    top2_margin_sum_total += float(route_stats.get("margin_sum", 0.0))

            all_total += batch_size

    taskid_acc = taskid_correct / max(all_total, 1)
    if num_seen > 1:
        mode = "oracle" if oracle_taskid else "predicted"
        print(f"\n  Task-ID accuracy ({mode}) : {taskid_acc:.4f}")
        if task_router_inference == "top2" and not oracle_taskid:
            dual_rate = dual_route_total / max(all_total, 1)
            print(
                f"  Adaptive top-2 routing rate: {dual_rate:.4f} "
                f"(alpha={router_alpha}, delta(T)={router_alpha / max(num_seen, 1):.4f}, class_prob_mode={class_prob_mode})"
            )
    result = compute_metrics(all_preds, all_labels, class_names, verbose=True)
    result["taskid_acc"] = taskid_acc
    if task_router_inference == "top2" and not oracle_taskid and num_seen > 1:
        result["task_router_dual_rate"] = dual_route_total / max(all_total, 1)
        per_true_task_stats = []
        for tid in range(num_seen):
            n_tid = int(top2_count_by_true_tid[tid])
            if n_tid > 0:
                avg_probs_tensor = top2_task_prob_sums_by_true_tid[tid] / float(n_tid)
            else:
                avg_probs_tensor = torch.zeros(num_seen, dtype=torch.float64)
            avg_probs = [float(v) for v in avg_probs_tensor.tolist()]
            top1_task_by_mean_prob = int(avg_probs_tensor.argmax().item()) if num_seen > 0 else -1
            dual_tid = int(top2_dual_by_true_tid[tid])
            per_true_task_stats.append({
                "true_task": tid,
                "num_samples": n_tid,
                "single_route_count": max(n_tid - dual_tid, 0),
                "dual_route_count": dual_tid,
                "dual_route_rate": float(dual_tid / max(n_tid, 1)),
                "top1_task_by_mean_prob": top1_task_by_mean_prob,
                "avg_task_probs": avg_probs,
            })
        result["task_router_debug"] = {
            "mode": "top2",
            "class_prob_mode": class_prob_mode,
            "alpha": float(router_alpha),
            "num_seen_tasks": int(num_seen),
            "delta_threshold": float(router_alpha / max(num_seen, 1)),
            "total_samples": int(all_total),
            "single_route_total": int(max(all_total - dual_route_total, 0)),
            "dual_route_total": int(dual_route_total),
            "dual_route_rate": float(dual_route_total / max(all_total, 1)),
            "avg_top1_prob": float(top2_top1_prob_sum_total / max(all_total, 1)),
            "avg_top2_prob": float(top2_top2_prob_sum_total / max(all_total, 1)),
            "avg_top1_top2_margin": float(top2_margin_sum_total / max(all_total, 1)),
            "per_true_task": per_true_task_stats,
        }
    return result
