"""
模型定义
  - ExpertAdapter        : Conv1×1 bottleneck 专家
  - MoEBlock             : 单层 MoE（管理多个任务的专家 + gate）— 增量用
  - IncrementalMoEResNet : 可配置 backbone + 逐任务 MoE
  - FullMoEBlock         : 单层 MoE（固定专家池 + 线性 gate + top-k）— 全量用
  - FullMoEResNet        : 可配置 backbone + 全量 MoE
  - StandardPrototypeHead: 欧氏距离原型分类头
  - CosinePrototypeHead  : 余弦相似度原型分类头
  - CosineHead           : 纯余弦分类头（无 prototype imprint）
  - LinearHead           : 线性分类头
  - FrozenFeatureExtractor: 冻结特征提取器（genclassifer 风格）用于路由前端
  - FeatureVAE           : 同构 MLP VAE（特征空间）
  - FeatureConditionalVAE: 条件特征空间 VAE
  - FeatureConditionalVQVAE: 条件特征空间 VQ-VAE
  - TaskReplayVAE        : 条件图像 VAE 生成器
  - TaskReplayVQVAE      : 条件图像 VQ-VAE 生成器
  - TaskIDClassifier     : Task-ID 图像分类器（ResNet18 + linear head，可选投影头）
  - FeatureTaskIDClassifier: Task-ID 特征分类器（MLP）
"""
import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchvision.models import (
        ConvNeXt_Small_Weights,
        EfficientNet_B3_Weights,
        ResNet18_Weights,
        ResNet50_Weights,
        ResNet101_Weights,
        VGG19_Weights,
        convnext_small,
        efficientnet_b3,
        resnet18,
        resnet50,
        resnet101,
        vgg19,
    )
except ImportError:
    ConvNeXt_Small_Weights = None
    EfficientNet_B3_Weights = None
    ResNet18_Weights = None
    ResNet50_Weights = None
    ResNet101_Weights = None
    VGG19_Weights = None
    from torchvision.models import resnet18, resnet101
    try:
        from torchvision.models import resnet50
    except ImportError:
        resnet50 = None
    try:
        from torchvision.models import convnext_small
    except ImportError:
        convnext_small = None
    try:
        from torchvision.models import efficientnet_b3
    except ImportError:
        efficientnet_b3 = None
    try:
        from torchvision.models import vgg19
    except ImportError:
        vgg19 = None


def _build_resnet_stages(
    backbone,
    moe_layers: Optional[List[str]] = None,
) -> Tuple[nn.ModuleDict, List[str], int]:
    moe_layers = [str(name) for name in (moe_layers or [])]
    use_block_layers = any("." in name or "_" in name for name in moe_layers)
    stages = nn.ModuleDict({
        "stem": nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),
    })
    if use_block_layers:
        order = ["stem"]
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(backbone, layer_name)
            for idx, block in enumerate(layer):
                key = f"{layer_name}_{idx}"
                stages[key] = block
                order.append(key)
        return stages, order, int(backbone.fc.in_features)
    stages.update({
        "layer1": backbone.layer1,
        "layer2": backbone.layer2,
        "layer3": backbone.layer3,
        "layer4": backbone.layer4,
    })
    return stages, ["stem", "layer1", "layer2", "layer3", "layer4"], int(backbone.fc.in_features)


def _build_resnet50_backbone(pretrained: bool):
    if resnet50 is None:
        raise ImportError("resnet50 is not available in the installed torchvision")
    try:
        if ResNet50_Weights is not None:
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            return resnet50(weights=weights)
        return resnet50(pretrained=pretrained)
    except Exception:
        try:
            return resnet50(weights=None)
        except TypeError:
            return resnet50(pretrained=False)


def _build_resnet101_backbone(pretrained: bool):
    try:
        if ResNet101_Weights is not None:
            weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            return resnet101(weights=weights)
        return resnet101(pretrained=pretrained)
    except Exception:
        try:
            return resnet101(weights=None)
        except TypeError:
            return resnet101(pretrained=False)


def _build_resnet18_backbone(pretrained: bool):
    try:
        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet18(weights=weights)
        return resnet18(pretrained=pretrained)
    except Exception:
        try:
            return resnet18(weights=None)
        except TypeError:
            return resnet18(pretrained=False)


def _build_efficientnet_b3_backbone(pretrained: bool):
    if efficientnet_b3 is None:
        raise ImportError("efficientnet_b3 is not available in the installed torchvision")
    try:
        if EfficientNet_B3_Weights is not None:
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            return efficientnet_b3(weights=weights)
        return efficientnet_b3(pretrained=pretrained)
    except Exception:
        try:
            return efficientnet_b3(weights=None)
        except TypeError:
            return efficientnet_b3(pretrained=False)


def _build_convnext_small_backbone(pretrained: bool):
    if convnext_small is None:
        raise ImportError("convnext_small is not available in the installed torchvision")
    try:
        if ConvNeXt_Small_Weights is not None:
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            return convnext_small(weights=weights)
        return convnext_small(pretrained=pretrained)
    except Exception:
        try:
            return convnext_small(weights=None)
        except TypeError:
            return convnext_small(pretrained=False)


def _build_vgg19_backbone(pretrained: bool):
    if vgg19 is None:
        raise ImportError("vgg19 is not available in the installed torchvision")
    try:
        if VGG19_Weights is not None:
            weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            return vgg19(weights=weights)
        return vgg19(pretrained=pretrained)
    except Exception:
        try:
            return vgg19(weights=None)
        except TypeError:
            return vgg19(pretrained=False)


def _tpl_deit_pretrained_path(filename: str) -> str:
    explicit = os.getenv("CIL_MORE_PRETRAINED_PATH", "").strip()
    if explicit:
        return explicit

    preferred_dir = os.getenv("CIL_TPL_PRETRAINED_DIR", "").strip()
    if preferred_dir:
        candidate = os.path.join(preferred_dir, filename)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Missing pretrained DeiT checkpoint. Set CIL_MORE_PRETRAINED_PATH "
        f"or place {filename} under CIL_TPL_PRETRAINED_DIR."
    )


def _build_tpl_deit_small_patch16_224_in661_backbone(pretrained: bool):
    tpl_repo_dir = os.getenv("CIL_TPL_REPO_DIR", "").strip()
    if not tpl_repo_dir:
        raise ModuleNotFoundError(
            "DeiT backbone code is not bundled in this repository. "
            "Set CIL_TPL_REPO_DIR to the directory containing networks/vit_hat.py."
        )
    if not os.path.isdir(tpl_repo_dir):
        raise FileNotFoundError(f"CIL_TPL_REPO_DIR is not a directory: {tpl_repo_dir}")
    if tpl_repo_dir not in sys.path:
        sys.path.insert(0, tpl_repo_dir)

    try:
        from networks.vit_hat import deit_small_patch16_224
    except ModuleNotFoundError as exc:
        if exc.name == "timm":
            raise ModuleNotFoundError(
                "DeiT backbone requires `timm` in the active environment. "
                "Use the mmlab environment or install timm there."
            ) from exc
        raise

    model = deit_small_patch16_224(
        pretrained=False,
        num_classes=1000,
        latent=64,
        args=None,
        hat=False,
    )

    if pretrained:
        checkpoint_path = _tpl_deit_pretrained_path("deit_small_patch16_224_in661.pth")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "missing DeiT pretrained checkpoint for MORE/TPL: "
                f"{checkpoint_path}"
            )
        try:
            from torch.serialization import add_safe_globals

            add_safe_globals([argparse.Namespace])
        except Exception:
            pass
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        target = model.state_dict()
        transfer = {
            k: v
            for k, v in checkpoint.items()
            if k in target and "head" not in k and tuple(target[k].shape) == tuple(v.shape)
        }
        target.update(transfer)
        model.load_state_dict(target)

    return model


class _DeiTPatchEmbedStage(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.dist_token = backbone.dist_token
        self.pos_embed = backbone.pos_embed
        self.pos_drop = backbone.pos_drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            dist_token = self.dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        return self.pos_drop(x + self.pos_embed)


class _DeiTOutputStage(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.norm = backbone.norm
        self.pre_logits = backbone.pre_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.pre_logits(x[:, 0])


def _build_deit_stages(
    backbone: nn.Module,
    moe_layers: Optional[List[str]] = None,
) -> Tuple[nn.ModuleDict, List[str], int]:
    blocks = list(backbone.blocks.children())
    if len(blocks) != 12:
        raise ValueError(f"expected DeiT-small with 12 transformer blocks, got {len(blocks)}")
    moe_layers = [str(name) for name in (moe_layers or [])]
    use_block_layers = any(name.startswith("block") for name in moe_layers)
    if use_block_layers:
        stages = nn.ModuleDict({"patch_embed": _DeiTPatchEmbedStage(backbone)})
        for idx, block in enumerate(blocks, start=1):
            stages[f"block{idx}"] = block
        stages["norm"] = _DeiTOutputStage(backbone)
        return stages, ["patch_embed", *[f"block{i}" for i in range(1, 13)], "norm"], int(backbone.embed_dim)
    stages = nn.ModuleDict({
        "patch_embed": _DeiTPatchEmbedStage(backbone),
        "stage1": nn.Sequential(*blocks[:4]),
        "stage2": nn.Sequential(*blocks[4:8]),
        "stage3": nn.Sequential(*blocks[8:10]),
        "stage4": nn.Sequential(*blocks[10:12]),
        "norm": _DeiTOutputStage(backbone),
    })
    return stages, ["patch_embed", "stage1", "stage2", "stage3", "stage4", "norm"], int(backbone.embed_dim)


def _build_backbone_stages(
    backbone_name: str,
    pretrained: bool,
    moe_layers: Optional[List[str]] = None,
) -> Tuple[nn.ModuleDict, List[str], int]:
    backbone_name = str(backbone_name).lower()

    if backbone_name == "resnet18":
        return _build_resnet_stages(_build_resnet18_backbone(pretrained), moe_layers=moe_layers)

    if backbone_name == "resnet50":
        return _build_resnet_stages(_build_resnet50_backbone(pretrained), moe_layers=moe_layers)

    if backbone_name == "resnet101":
        return _build_resnet_stages(_build_resnet101_backbone(pretrained), moe_layers=moe_layers)

    if backbone_name == "efficientnet_b3":
        backbone = _build_efficientnet_b3_backbone(pretrained)
        features = backbone.features
        stages = nn.ModuleDict({
            "stem": features[0],
            "stage1": features[1],
            "stage2": features[2],
            "stage3": features[3],
            "stage4": features[4],
            "stage5": features[5],
            "stage6": features[6],
            "stage7": features[7],
            "head": features[8],
        })
        return stages, ["stem", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7", "head"], int(backbone.classifier[-1].in_features)

    if backbone_name == "convnext_small":
        backbone = _build_convnext_small_backbone(pretrained)
        features = backbone.features
        stages = nn.ModuleDict({
            "stem": features[0],
            "stage1": features[1],
            "down1": features[2],
            "stage2": features[3],
            "down2": features[4],
            "stage3": features[5],
            "down3": features[6],
            "stage4": features[7],
        })
        return stages, ["stem", "stage1", "down1", "stage2", "down2", "stage3", "down3", "stage4"], int(backbone.classifier[-1].in_features)

    if backbone_name == "vgg19":
        backbone = _build_vgg19_backbone(pretrained)
        features = backbone.features
        stages = nn.ModuleDict({
            "stage1": nn.Sequential(*features[:5]),
            "stage2": nn.Sequential(*features[5:10]),
            "stage3": nn.Sequential(*features[10:19]),
            "stage4": nn.Sequential(*features[19:28]),
            "stage5": nn.Sequential(*features[28:37]),
        })
        return stages, ["stage1", "stage2", "stage3", "stage4", "stage5"], 512

    if backbone_name == "deit_small_patch16_224_in661":
        return _build_deit_stages(_build_tpl_deit_small_patch16_224_in661_backbone(pretrained), moe_layers=moe_layers)

    raise ValueError(f"unsupported backbone: {backbone_name}")


class FrozenFeatureExtractor(nn.Module):
    """ImageNet 预训练骨干（冻结）+ 展平特征输出。"""

    def __init__(self, backbone: str = "resnet18", image_size: int = 224, pretrained: bool = True):
        super().__init__()
        backbone = str(backbone).lower()
        self._feature_mode = "cnn"
        if backbone == "resnet18":
            net = _build_resnet18_backbone(pretrained=pretrained)
            self.body = nn.Sequential(*list(net.children())[:-1])
        elif backbone == "resnet50":
            net = _build_resnet50_backbone(pretrained=pretrained)
            self.body = nn.Sequential(*list(net.children())[:-1])
        elif backbone == "deit_small_patch16_224_in661":
            self.body = _build_tpl_deit_small_patch16_224_in661_backbone(pretrained=pretrained)
            self._feature_mode = "vit"
        else:
            raise ValueError(
                "FrozenFeatureExtractor supports only "
                "'resnet18', 'resnet50', or 'deit_small_patch16_224_in661'"
            )

        for p in self.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            probe = torch.zeros(1, 3, int(image_size), int(image_size))
            self.feat_dim = int(self.forward(probe).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._feature_mode == "vit":
            return self.body.forward_features(x)
        return self.body(x).flatten(1)


def _log_normal_diag(z: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * (
        log_var + ((z - mean).pow(2) / log_var.exp().clamp(min=1e-8)) + math.log(2 * math.pi)
    ).sum(dim=-1)


def _log_standard_normal(x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    return -0.5 * ((x - mean).pow(2) + math.log(2 * math.pi)).sum(dim=-1)


class FeatureVAE(nn.Module):
    """Feature-space VAE，架构与 genclassifier 的 FeatureVAE 对齐。"""

    def __init__(self, input_dim: int, h_dim: int = 512, z_dim: int = 128):
        super().__init__()
        self.input_dim = int(input_dim)
        self.h_dim = int(h_dim)
        self.z_dim = int(z_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(self.h_dim, self.z_dim)
        self.fc_logvar = nn.Linear(self.h_dim, self.z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = logvar.mul(0.5).exp()
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    @torch.no_grad()
    def estimate_loglikelihood(self, x: torch.Tensor, S: int = 100, is_batch: int = 32) -> torch.Tensor:
        """Importance sampling 下的 log p(x|y) 似然近似。"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1).contiguous()
        if batch_size == 0:
            return x.new_zeros((0,), dtype=x.dtype)

        mu, logvar = self.encode(x)
        is_batch = max(1, int(is_batch))
        repeats = max(1, int(math.ceil(float(S) / float(is_batch))))
        all_ll = []

        for rep in range(repeats):
            cur = is_batch if rep < repeats - 1 else (S - (repeats - 1) * is_batch)
            if cur <= 0:
                continue
            mu_rep = mu.repeat_interleave(cur, dim=0)
            logvar_rep = logvar.repeat_interleave(cur, dim=0)
            z = self.reparameterize(mu_rep, logvar_rep)
            recon = self.decode(z)

            log_p_z = _log_standard_normal(z, torch.zeros_like(z))
            log_q = _log_normal_diag(z, mu_rep, logvar_rep)
            x_rep = x.repeat_interleave(cur, dim=0)
            log_p_xz = _log_standard_normal(x_rep, recon)
            all_ll.append((log_p_xz + log_p_z - log_q).view(batch_size, cur))

        ll = torch.cat(all_ll, dim=1)[:, :S]
        return ll.logsumexp(dim=1) - math.log(float(S))

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        count = int(labels.shape[0])
        z = torch.randn((count, self.z_dim), device=device)
        return self.decode(z)


class FeatureConditionalVAE(nn.Module):
    """Feature-space conditional VAE，输入为特征向量，条件为局部类别 id。"""

    def __init__(self, input_dim: int, num_classes: int, h_dim: int = 512, z_dim: int = 128):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.h_dim = int(h_dim)
        self.z_dim = int(z_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + self.num_classes, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(self.h_dim, self.z_dim)
        self.fc_logvar = nn.Linear(self.h_dim, self.z_dim)
        self.label_embed = nn.Embedding(self.num_classes, self.z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim * 2, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.input_dim),
        )

    def _label_one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long()
        return F.one_hot(labels, num_classes=self.num_classes).float()

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x, self._label_one_hot(labels).to(device=x.device, dtype=x.dtype)], dim=1)
        h = self.encoder(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = logvar.mul(0.5).exp()
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z, self.label_embed(labels.long())], dim=1)
        return self.decoder(z)

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar, z

    @torch.no_grad()
    def estimate_loglikelihood(self, x: torch.Tensor, labels: torch.Tensor, S: int = 100, is_batch: int = 32) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1).contiguous()
        labels = labels.reshape(batch_size).long()
        if batch_size == 0:
            return x.new_zeros((0,), dtype=x.dtype)

        mu, logvar = self.encode(x, labels)
        is_batch = max(1, int(is_batch))
        repeats = max(1, int(math.ceil(float(S) / float(is_batch))))
        all_ll = []

        for rep in range(repeats):
            cur = is_batch if rep < repeats - 1 else (S - (repeats - 1) * is_batch)
            if cur <= 0:
                continue
            mu_rep = mu.repeat_interleave(cur, dim=0)
            logvar_rep = logvar.repeat_interleave(cur, dim=0)
            labels_rep = labels.repeat_interleave(cur)
            z = self.reparameterize(mu_rep, logvar_rep)
            recon = self.decode(z, labels_rep)

            log_p_z = _log_standard_normal(z, torch.zeros_like(z))
            log_q = _log_normal_diag(z, mu_rep, logvar_rep)
            x_rep = x.repeat_interleave(cur, dim=0)
            log_p_xz = _log_standard_normal(x_rep, recon)
            all_ll.append((log_p_xz + log_p_z - log_q).view(batch_size, cur))

        ll = torch.cat(all_ll, dim=1)[:, :S]
        return ll.logsumexp(dim=1) - math.log(float(S))

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        device: torch.device,
        latent_pool: Optional[List[torch.Tensor]] = None,
        latent_noise_std: float = 0.0,
    ) -> torch.Tensor:
        labels = labels.to(device).long()
        if latent_pool is not None:
            z = torch.empty((labels.shape[0], self.z_dim), device=device)
            for cls_id in labels.unique().tolist():
                cls_id = int(cls_id)
                pool = latent_pool[cls_id]
                if pool.numel() == 0:
                    raise ValueError(f"latent pool for class {cls_id} is empty")
                mask = labels == cls_id
                choice = torch.randint(pool.shape[0], (int(mask.sum().item()),), device=device)
                sampled = pool.to(device)[choice]
                if latent_noise_std > 0.0:
                    sampled = sampled + torch.randn_like(sampled) * float(latent_noise_std)
                z[mask] = sampled
        else:
            z = torch.randn(labels.shape[0], self.z_dim, device=device)
        return self.decode(z, labels)


# ═══════════════════════════════════════════════
#  MoE 专家 & 路由
# ═══════════════════════════════════════════════
class ExpertAdapter(nn.Module):
    """
    Conv1×1 Bottleneck Adapter (并联到 backbone 层之后):
        x → Conv1x1(C→r) → BN → ReLU → Conv1x1(r→C) → BN → out
    输出与输入同维，可直接残差相加。
    """

    def __init__(self, channels: int, bottleneck: int, feature_ndim: int = 4):
        super().__init__()
        self.feature_ndim = int(feature_ndim)
        if self.feature_ndim == 4:
            self.down = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False)
            self.norm1 = nn.BatchNorm2d(bottleneck)
            self.up = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False)
            self.norm2 = nn.BatchNorm2d(channels)
        elif self.feature_ndim == 3:
            self.down = nn.Linear(channels, bottleneck, bias=False)
            self.norm1 = nn.LayerNorm(bottleneck)
            self.up = nn.Linear(bottleneck, channels, bias=False)
            self.norm2 = nn.LayerNorm(channels)
        else:
            raise ValueError(f"unsupported ExpertAdapter feature_ndim={feature_ndim}")
        # 初始化 up 权重接近零，保证刚加入时残差输出 ≈ 0
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.up(F.relu(self.norm1(self.down(x)))))


ExpertCountSpec = Union[int, List[int], Tuple[int, ...], Dict[str, int]]


def normalize_moe_expert_counts(num_experts: ExpertCountSpec, moe_layers: List[str]) -> Dict[str, int]:
    """Normalize expert-count config to a per-layer dict keyed by moe layer name."""
    if isinstance(num_experts, int):
        counts = [int(num_experts)] * len(moe_layers)
    elif isinstance(num_experts, (list, tuple)):
        if len(num_experts) != len(moe_layers):
            raise ValueError(
                f"num_experts length {len(num_experts)} does not match moe_layers length {len(moe_layers)}"
            )
        counts = [int(v) for v in num_experts]
    elif isinstance(num_experts, dict):
        missing = [name for name in moe_layers if name not in num_experts]
        if missing:
            raise KeyError(f"num_experts is missing layer keys: {missing}")
        counts = [int(num_experts[name]) for name in moe_layers]
    else:
        raise TypeError(
            "num_experts must be an int, a list/tuple aligned with moe_layers, or a dict keyed by moe layer name"
        )

    normalized = {}
    for name, count in zip(moe_layers, counts):
        if count <= 0:
            raise ValueError(f"num_experts for {name} must be positive, got {count}")
        normalized[name] = count
    return normalized


class MoEBlock(nn.Module):
    """
    一层的 MoE 模块，管理所有任务的专家。
    每个任务拥有独立的 num_experts 个专家 + 1 个 gate。
    可选：为较新任务额外开放 1 个来自历史任务的自由旧专家路由机会。
    """

    def __init__(
        self,
        channels: int,
        bottleneck: int,
        feature_ndim: int = 4,
        allow_old_expert_reuse: bool = False,
        old_expert_top_k: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.bottleneck = bottleneck
        self.feature_ndim = int(feature_ndim)
        self.allow_old_expert_reuse = bool(allow_old_expert_reuse)
        self.old_expert_top_k = max(1, int(old_expert_top_k))
        self.task_experts = nn.ModuleDict()  # str(task_id) → ModuleList[ExpertAdapter]
        self.task_gates = nn.ModuleDict()    # str(task_id) → Linear(C, num_experts)
        self.task_old_gates = nn.ModuleDict()  # str(task_id) → Linear(C, visible_old_experts)
        self.task_old_refs: Dict[str, List[Tuple[str, int]]] = {}

    def add_task(self, task_id: int, num_experts: int = 2):
        key = str(task_id)
        device = None
        for module_list in self.task_experts.values():
            if len(module_list) > 0:
                device = next(module_list.parameters()).device
                break
        experts = nn.ModuleList([
            ExpertAdapter(self.channels, self.bottleneck, self.feature_ndim)
            for _ in range(num_experts)
        ])
        gate = nn.Linear(self.channels, num_experts)
        if device is not None:
            experts = experts.to(device)
            gate = gate.to(device)
        self.task_experts[key] = experts
        self.task_gates[key] = gate
        if self.allow_old_expert_reuse and int(task_id) > 0:
            visible_old_refs: List[Tuple[str, int]] = []
            for old_key in sorted(self.task_experts.keys(), key=int):
                if int(old_key) >= int(task_id):
                    continue
                for expert_idx in range(len(self.task_experts[old_key])):
                    visible_old_refs.append((old_key, expert_idx))
            if visible_old_refs:
                old_gate = nn.Linear(self.channels, len(visible_old_refs))
                if device is not None:
                    old_gate = old_gate.to(device)
                self.task_old_gates[key] = old_gate
                self.task_old_refs[key] = visible_old_refs

    def freeze_task(self, task_id: int):
        key = str(task_id)
        for p in self.task_experts[key].parameters():
            p.requires_grad_(False)
        for p in self.task_gates[key].parameters():
            p.requires_grad_(False)
        if key in self.task_old_gates:
            for p in self.task_old_gates[key].parameters():
                p.requires_grad_(False)

    def _gate_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return F.adaptive_avg_pool2d(x, 1).flatten(1)
        if x.dim() == 3:
            return x[:, 0]
        raise ValueError(f"unsupported MoEBlock input shape: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor, task_id: int, return_gate: bool = False):
        key = str(task_id)
        experts = self.task_experts[key]
        gate = self.task_gates[key]

        # Gate: CNN uses GAP; ViT/DeiT token features use the class token.
        g = self._gate_inputs(x)                              # [B, C]
        w = F.softmax(gate(g), dim=1)                     # [B, num_experts]

        # 加权融合所有专家输出
        out = torch.zeros_like(x)
        for i, expert in enumerate(experts):
            if x.dim() == 4:
                weight = w[:, i].view(-1, 1, 1, 1)
            else:
                weight = w[:, i].view(-1, 1, 1)
            out = out + weight * expert(x)

        old_sparse_w = None
        if self.allow_old_expert_reuse and key in self.task_old_gates and self.task_old_refs.get(key):
            old_gate = self.task_old_gates[key]
            old_scores = old_gate(g)
            select_k = min(self.old_expert_top_k, int(old_scores.shape[1]))
            topk_vals, topk_idx = old_scores.topk(select_k, dim=1)
            topk_weights = F.softmax(topk_vals, dim=1)
            old_sparse_w = torch.zeros_like(old_scores)
            old_sparse_w.scatter_(1, topk_idx, topk_weights)
            for old_idx, (old_task_key, old_expert_idx) in enumerate(self.task_old_refs[key]):
                expert_weight = old_sparse_w[:, old_idx]
                if expert_weight.sum() <= 0:
                    continue
                old_expert = self.task_experts[old_task_key][old_expert_idx]
                if x.dim() == 4:
                    weight = expert_weight.view(-1, 1, 1, 1)
                else:
                    weight = expert_weight.view(-1, 1, 1)
                out = out + weight * old_expert(x)

        out = x + out  # 残差连接: backbone 特征 + MoE 修正
        if return_gate:
            return out, w if old_sparse_w is None else torch.cat([w, old_sparse_w], dim=1)
        return out


# ═══════════════════════════════════════════════
#  Backbone + MoE
# ═══════════════════════════════════════════════
class IncrementalMoEResNet(nn.Module):
    """
    冻结 backbone，在指定 stage 后插入逐任务 MoE 模块。
    提供两种前向：
      - forward(x, task_id)          : backbone + MoE → 特征 (用于分类)
      - forward_backbone_only(x)     : 纯 backbone → 特征 (用于 task-id / VAE)
    """

    def __init__(
        self,
        backbone_name: str = "resnet101",
        pretrained: bool = True,
        moe_layers: Optional[List[str]] = None,
        moe_channels: Optional[Dict[str, int]] = None,
        bottleneck_ratios: Optional[Dict[str, int]] = None,
        allow_old_expert_reuse: bool = False,
        old_expert_top_k: int = 1,
    ):
        super().__init__()
        self.backbone_name = str(backbone_name).lower()
        self.stages, self.stage_order, self.feat_dim = _build_backbone_stages(
            self.backbone_name,
            pretrained,
            moe_layers=moe_layers,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.backbone_modules = [self.stages[name] for name in self.stage_order]
        self.backbone_trainable = False
        self.moe_feature_ndim = 3 if self.backbone_name == "deit_small_patch16_224_in661" else 4

        for param in self.parameters():
            param.requires_grad_(False)

        self.moe_layer_names = moe_layers or ["layer2", "layer3", "layer4"]
        self.moe_blocks = nn.ModuleDict()
        for name in self.moe_layer_names:
            if name not in self.stages:
                raise ValueError(f"moe layer {name} is not available in backbone {self.backbone_name}")
            ch = moe_channels[name]
            r = ch // bottleneck_ratios[name]
            self.moe_blocks[name] = MoEBlock(
                ch,
                r,
                self.moe_feature_ndim,
                allow_old_expert_reuse=allow_old_expert_reuse,
                old_expert_top_k=old_expert_top_k,
            )
        self._freeze_backbone_bn_stats()

    def _finalize_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = self.avgpool(x)
            return torch.flatten(x, 1)
        if x.dim() == 3:
            return x[:, 0]
        if x.dim() == 2:
            return x
        raise ValueError(f"unsupported backbone output shape: {tuple(x.shape)}")

    def _freeze_backbone_bn_stats(self) -> None:
        for module in self.backbone_modules:
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.backbone_trainable:
            self._freeze_backbone_bn_stats()
        return self

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone_trainable = bool(trainable)
        for module in self.backbone_modules:
            for param in module.parameters():
                param.requires_grad_(self.backbone_trainable)
        if not self.backbone_trainable:
            self._freeze_backbone_bn_stats()

    # ---------- 任务管理 ----------
    def add_task(self, task_id: int, num_experts: ExpertCountSpec = 2):
        device = next(self.backbone_modules[0].parameters()).device
        per_layer_counts = normalize_moe_expert_counts(num_experts, self.moe_layer_names)
        for name in self.moe_layer_names:
            self.moe_blocks[name].add_task(task_id, per_layer_counts[name])
            self.moe_blocks[name].to(device)

    def freeze_task(self, task_id: int):
        for name in self.moe_layer_names:
            self.moe_blocks[name].freeze_task(task_id)

    # ---------- 前向 ----------
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        for name in self.stage_order:
            x = self.stages[name](x)
            if name in self.moe_blocks:
                x = self.moe_blocks[name](x, task_id)
        return self._finalize_features(x)

    def forward_with_gates(self, x: torch.Tensor, task_id: int):
        gate_dict = {}
        for name in self.stage_order:
            x = self.stages[name](x)
            if name in self.moe_blocks:
                x, gate_dict[name] = self.moe_blocks[name](x, task_id, return_gate=True)
        return self._finalize_features(x), gate_dict

    @torch.no_grad()
    def routing_layer_weights(self, x: torch.Tensor, task_id: int) -> Dict[str, torch.Tensor]:
        _, gate_dict = self.forward_with_gates(x, task_id)
        if not gate_dict:
            raise ValueError("no moe blocks available for routing statistics")
        return gate_dict

    @torch.no_grad()
    def routing_weights(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        gate_dict = self.routing_layer_weights(x, task_id)
        widths = {weights.shape[1] for weights in gate_dict.values()}
        if len(widths) == 1:
            gates = torch.stack(list(gate_dict.values()), dim=0)
            return gates.mean(dim=0)
        return torch.cat([gate_dict[name] for name in self.moe_layer_names], dim=1)

    @torch.no_grad()
    def forward_backbone_only(self, x: torch.Tensor) -> torch.Tensor:
        """纯 backbone (无 MoE) → GAP → [B, feat_dim]，用于 task-id 分类器和 VAE。"""
        for name in self.stage_order:
            x = self.stages[name](x)
        return self._finalize_features(x)


class FullMoEBlock(nn.Module):
    """
    全量 MoE: 固定 num_experts 个专家 + 线性 gate + top-k 稀疏路由。
    所有专家一次性创建、一起训练，不区分任务。
    """

    def __init__(
        self,
        channels: int,
        bottleneck: int,
        num_experts: int,
        top_k: int = 2,
        feature_ndim: int = 4,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.feature_ndim = int(feature_ndim)
        self.experts = nn.ModuleList(
            [ExpertAdapter(channels, bottleneck, self.feature_ndim) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(channels, num_experts)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        # Gate: CNN uses GAP; ViT/DeiT token features use the class token.
        if x.dim() == 4:
            g = F.adaptive_avg_pool2d(x, 1).flatten(1)     # [B, C]
        elif x.dim() == 3:
            g = x[:, 0]                                    # [B, C]
        else:
            raise ValueError(f"unsupported FullMoEBlock input shape: {tuple(x.shape)}")
        scores = self.gate(g)                            # [B, num_experts]

        # Top-k 稀疏路由
        topk_vals, topk_idx = scores.topk(self.top_k, dim=1)  # [B, k]
        topk_weights = F.softmax(topk_vals, dim=1)             # [B, k]

        # 构建稀疏权重矩阵
        w = torch.zeros_like(scores)                           # [B, num_experts]
        w.scatter_(1, topk_idx, topk_weights)

        # 加权融合被选中的专家
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            wi = w[:, i]
            if wi.sum() > 0:
                if x.dim() == 4:
                    weight = wi.view(-1, 1, 1, 1)
                else:
                    weight = wi.view(-1, 1, 1)
                out = out + weight * expert(x)

        out = x + out
        if return_gate:
            return out, w
        return out


class FullMoEResNet(nn.Module):
    """
    全量基线: 冻结 backbone + 每层一个 FullMoEBlock。
    所有 6 个专家一次性训练，不需要 task-id / VAE。
    """

    def __init__(
        self,
        backbone_name: str = "resnet101",
        pretrained: bool = True,
        moe_layers: Optional[List[str]] = None,
        moe_channels: Optional[Dict[str, int]] = None,
        bottleneck_ratios: Optional[Dict[str, int]] = None,
        num_experts: int = 6,
        top_k: int = 2,
    ):
        super().__init__()
        self.backbone_name = str(backbone_name).lower()
        self.stages, self.stage_order, self.feat_dim = _build_backbone_stages(
            self.backbone_name,
            pretrained,
            moe_layers=moe_layers,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.backbone_modules = [self.stages[name] for name in self.stage_order]
        self.moe_feature_ndim = 3 if self.backbone_name == "deit_small_patch16_224_in661" else 4

        for param in self.parameters():
            param.requires_grad_(False)

        self.moe_layer_names = moe_layers or ["layer2", "layer3", "layer4"]
        self.moe_blocks = nn.ModuleDict()
        for name in self.moe_layer_names:
            if name not in self.stages:
                raise ValueError(f"moe layer {name} is not available in backbone {self.backbone_name}")
            ch = moe_channels[name]
            r = ch // bottleneck_ratios[name]
            self.moe_blocks[name] = FullMoEBlock(ch, r, num_experts, top_k, self.moe_feature_ndim)
        self._freeze_backbone_bn_stats()

    def _finalize_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = self.avgpool(x)
            return torch.flatten(x, 1)
        if x.dim() == 3:
            return x[:, 0]
        if x.dim() == 2:
            return x
        raise ValueError(f"unsupported backbone output shape: {tuple(x.shape)}")

    def _freeze_backbone_bn_stats(self) -> None:
        for module in self.backbone_modules:
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_backbone_bn_stats()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """backbone + MoE → GAP → [B, feat_dim]"""
        for name in self.stage_order:
            x = self.stages[name](x)
            if name in self.moe_blocks:
                x = self.moe_blocks[name](x)
        return self._finalize_features(x)

    def forward_with_gates(self, x: torch.Tensor):
        gate_dict = {}
        for name in self.stage_order:
            x = self.stages[name](x)
            if name in self.moe_blocks:
                x, gate_dict[name] = self.moe_blocks[name](x, return_gate=True)
        return self._finalize_features(x), gate_dict

    @torch.no_grad()
    def routing_layer_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, gate_dict = self.forward_with_gates(x)
        if not gate_dict:
            raise ValueError("no moe blocks available for routing statistics")
        return gate_dict

    @torch.no_grad()
    def routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        gate_dict = self.routing_layer_weights(x)
        widths = {weights.shape[1] for weights in gate_dict.values()}
        if len(widths) == 1:
            gates = torch.stack(list(gate_dict.values()), dim=0)
            return gates.mean(dim=0)
        return torch.cat([gate_dict[name] for name in self.moe_layer_names], dim=1)


# ═══════════════════════════════════════════════
#  分类头: Prototype
# ═══════════════════════════════════════════════
class StandardPrototypeHead(nn.Module):
    """
    标准欧氏距离原型分类头。
    logits = -||feat - prototypes||^2
    """

    def __init__(self, feat_dim: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        del scale
        self.prototypes = nn.Parameter(torch.zeros(num_classes, feat_dim))
        self.register_buffer("train_mask", torch.ones(num_classes, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sq = x.pow(2).sum(dim=1, keepdim=True)
        p_sq = self.prototypes.pow(2).sum(dim=1).unsqueeze(0)
        return -(x_sq - 2.0 * (x @ self.prototypes.t()) + p_sq)

    def expand(self, num_new: int):
        if num_new <= 0:
            return
        old_p = self.prototypes.data
        new_p = torch.zeros(old_p.shape[0] + num_new, old_p.shape[1], device=old_p.device, dtype=old_p.dtype)
        new_p[: old_p.shape[0]].copy_(old_p)
        self.prototypes = nn.Parameter(new_p)

        old_mask = self.train_mask
        new_mask = torch.zeros(new_p.shape[0], dtype=torch.bool, device=old_mask.device)
        new_mask[: old_mask.shape[0]] = old_mask
        new_mask[old_mask.shape[0] :] = True
        self.register_buffer("train_mask", new_mask)

    def set_trainable_classes(self, class_ids: List[int]):
        mask = torch.zeros_like(self.train_mask)
        for cls_id in class_ids:
            idx = int(cls_id)
            if 0 <= idx < mask.shape[0]:
                mask[idx] = True
        self.register_buffer("train_mask", mask)

    @torch.no_grad()
    def imprint(self, features: torch.Tensor, labels: torch.Tensor, class_ids: List[int]):
        """用训练集特征均值初始化原型 (weight imprinting)。"""
        for local_idx, global_cls in enumerate(class_ids):
            mask = labels == global_cls
            if mask.sum() > 0:
                proto_idx = int(global_cls) if 0 <= int(global_cls) < self.prototypes.shape[0] else int(local_idx)
                self.prototypes[proto_idx] = features[mask].mean(0)

    def apply_grad_mask(self):
        if self.prototypes.grad is None:
            return
        if self.train_mask.shape[0] != self.prototypes.shape[0]:
            return
        mask = self.train_mask.to(device=self.prototypes.grad.device)
        self.prototypes.grad[~mask] = 0


class CosinePrototypeHead(nn.Module):
    """
    基于余弦相似度的原型分类头。
    logits = scale * cosine(feat, prototypes)
    """

    def __init__(self, feat_dim: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        self.scale = float(scale)
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.kaiming_uniform_(self.prototypes)
        self.register_buffer("train_mask", torch.ones(num_classes, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = F.normalize(x, dim=1)
        p_n = F.normalize(self.prototypes, dim=1)
        return self.scale * (x_n @ p_n.t())

    def expand(self, num_new: int):
        if num_new <= 0:
            return
        old_p = self.prototypes.data
        new_p = torch.randn(old_p.shape[0] + num_new, old_p.shape[1], device=old_p.device, dtype=old_p.dtype)
        new_p[: old_p.shape[0]].copy_(old_p)
        nn.init.kaiming_uniform_(new_p[old_p.shape[0] :])
        self.prototypes = nn.Parameter(new_p)

        old_mask = self.train_mask
        new_mask = torch.zeros(new_p.shape[0], dtype=torch.bool, device=old_mask.device)
        new_mask[: old_mask.shape[0]] = old_mask
        new_mask[old_mask.shape[0] :] = True
        self.register_buffer("train_mask", new_mask)

    def set_trainable_classes(self, class_ids: List[int]):
        mask = torch.zeros_like(self.train_mask)
        for cls_id in class_ids:
            idx = int(cls_id)
            if 0 <= idx < mask.shape[0]:
                mask[idx] = True
        self.register_buffer("train_mask", mask)

    @torch.no_grad()
    def imprint(self, features: torch.Tensor, labels: torch.Tensor, class_ids: List[int]):
        """用训练集特征均值初始化原型 (weight imprinting)."""
        for local_idx, global_cls in enumerate(class_ids):
            mask = labels == global_cls
            if mask.sum() > 0:
                proto_idx = int(global_cls) if 0 <= int(global_cls) < self.prototypes.shape[0] else int(local_idx)
                self.prototypes[proto_idx] = F.normalize(features[mask].mean(0), dim=0)

    def apply_grad_mask(self):
        if self.prototypes.grad is None:
            return
        if self.train_mask.shape[0] != self.prototypes.shape[0]:
            return
        mask = self.train_mask.to(device=self.prototypes.grad.device)
        self.prototypes.grad[~mask] = 0


# Backward compatibility: legacy full-training code paths still import PrototypeHead.
PrototypeHead = CosinePrototypeHead


class LinearHead(nn.Module):
    """Standard linear classifier head for per-task classification."""

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class CosineHead(nn.Module):
    """Cosine classifier without prototype-specific imprinting behavior."""

    def __init__(self, feat_dim: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = F.normalize(x, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        return self.scale * (x_n @ w_n.t())


# ═══════════════════════════════════════════════
#  VAE 生成器 (特征空间)
# ═══════════════════════════════════════════════
class TaskReplayVAE(nn.Module):
    """
    条件图像 VAE。
    输入输出都是图像张量，条件是任务内类别 id。
    """

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        latent_dim: int,
        base_channels: int = 32,
        channel_multipliers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers or [1, 2, 4, 8]
        if not self.channel_multipliers:
            raise ValueError("channel_multipliers must not be empty")
        num_stages = len(self.channel_multipliers)
        self.spatial_size = image_size // (2 ** num_stages)
        if self.spatial_size <= 0:
            raise ValueError("image_size is too small for the configured VAE depth")
        if image_size % (2 ** num_stages) != 0:
            raise ValueError("image_size must be divisible by 2 ** len(channel_multipliers)")

        enc_in = 3 + num_classes
        encoder_layers = []
        prev_channels = enc_in
        self.encoder_channels = []
        for idx, mult in enumerate(self.channel_multipliers):
            out_channels = base_channels * mult
            encoder_layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=4, stride=2, padding=1))
            if idx > 0:
                encoder_layers.append(nn.BatchNorm2d(out_channels))
            encoder_layers.append(nn.ReLU(inplace=True))
            prev_channels = out_channels
            self.encoder_channels.append(out_channels)
        self.encoder = nn.Sequential(*encoder_layers)

        enc_dim = self.encoder_channels[-1] * self.spatial_size * self.spatial_size
        self.fc_mu = nn.Linear(enc_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_dim, latent_dim)
        self.label_embed = nn.Embedding(num_classes, latent_dim)
        self.fc_decode = nn.Linear(latent_dim * 2, enc_dim)
        decoder_layers = []
        decoder_channels = list(reversed(self.encoder_channels))
        current_channels = decoder_channels[0]
        for out_channels in decoder_channels[1:]:
            decoder_layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.BatchNorm2d(out_channels))
            decoder_layers.append(nn.ReLU(inplace=True))
            current_channels = out_channels
        decoder_layers.append(nn.ConvTranspose2d(current_channels, 3, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def _label_map(self, labels: torch.Tensor, height: int, width: int) -> torch.Tensor:
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        return one_hot[:, :, None, None].expand(-1, -1, height, width)

    def encode(self, x: torch.Tensor, labels: torch.Tensor):
        h = torch.cat([x, self._label_map(labels, x.shape[2], x.shape[3])], dim=1)
        h = self.encoder(h)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor):
        z = torch.cat([z, self.label_embed(labels)], dim=1)
        h = self.fc_decode(z)
        h = h.view(-1, self.encoder_channels[-1], self.spatial_size, self.spatial_size)
        return self.decoder(h)

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        device: torch.device,
        latent_pool: Optional[List[torch.Tensor]] = None,
        latent_noise_std: float = 0.0,
    ) -> torch.Tensor:
        if latent_pool is not None:
            labels_device = labels.to(device)
            z = torch.empty((labels.shape[0], self.latent_dim), device=device)
            for cls_id in labels_device.unique().tolist():
                cls_id = int(cls_id)
                pool = latent_pool[cls_id]
                if pool.numel() == 0:
                    raise ValueError(f"latent pool for class {cls_id} is empty")
                mask = labels_device == cls_id
                choice = torch.randint(pool.shape[0], (int(mask.sum().item()),), device=device)
                sampled = pool.to(device)[choice]
                if latent_noise_std > 0.0:
                    sampled = sampled + torch.randn_like(sampled) * float(latent_noise_std)
                z[mask] = sampled
        else:
            z = torch.randn(labels.shape[0], self.latent_dim, device=device)
        return self.decode(z, labels.to(device))


class VectorQuantizer(nn.Module):
    """
    VQ codebook with EMA updates and dead-code reset.

    When ``ema_decay > 0`` (default 0.99) the codebook is updated via
    exponential moving averages of encoder outputs instead of gradient
    descent, which is much more stable on small datasets and prevents
    codebook collapse.  Dead codes (unused for ``dead_code_threshold``
    consecutive forward passes) are reset to randomly sampled encoder
    outputs.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 2,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.ema_decay = float(ema_decay)
        self.use_ema = self.ema_decay > 0.0
        self.dead_code_threshold = int(dead_code_threshold)

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

        if self.use_ema:
            self.embedding.weight.requires_grad_(False)
            self.register_buffer("_ema_cluster_size", torch.zeros(self.num_embeddings))
            self.register_buffer("_ema_embed_sum", self.embedding.weight.data.clone())
            self.register_buffer("_dead_count", torch.zeros(self.num_embeddings, dtype=torch.long))

    def _flatten(self, z: torch.Tensor) -> torch.Tensor:
        return z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

    def _nearest_indices(self, z: torch.Tensor) -> torch.Tensor:
        flat_z = self._flatten(z)
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = torch.argmin(distances, dim=1)
        return indices.view(z.shape[0], z.shape[2], z.shape[3])

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        quantized = self.embedding(indices.reshape(-1))
        quantized = quantized.view(indices.shape[0], indices.shape[1], indices.shape[2], self.embedding_dim)
        return quantized.permute(0, 3, 1, 2).contiguous()

    def _ema_update(self, flat_z: torch.Tensor, indices_flat: torch.Tensor):
        encodings = F.one_hot(indices_flat, num_classes=self.num_embeddings).float()
        batch_cluster_size = encodings.sum(dim=0)
        batch_embed_sum = encodings.t() @ flat_z

        self._ema_cluster_size.mul_(self.ema_decay).add_(
            batch_cluster_size, alpha=1.0 - self.ema_decay
        )
        self._ema_embed_sum.mul_(self.ema_decay).add_(
            batch_embed_sum, alpha=1.0 - self.ema_decay
        )

        n = self._ema_cluster_size.sum()
        smoothed = (self._ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
        self.embedding.weight.data.copy_(self._ema_embed_sum / smoothed.unsqueeze(1))

        # dead code reset
        used_mask = batch_cluster_size > 0
        self._dead_count[used_mask] = 0
        self._dead_count[~used_mask] += 1
        dead_mask = self._dead_count >= self.dead_code_threshold
        num_dead = dead_mask.sum().item()
        if num_dead > 0 and flat_z.shape[0] > 0:
            num_avail = int(flat_z.shape[0])
            if num_dead <= num_avail:
                pick = torch.randperm(num_avail, device=flat_z.device)[:num_dead]
            else:
                # dead code can exceed available latents in a small tail batch.
                # fallback to sampling with replacement to keep shapes aligned.
                pick = torch.randint(0, num_avail, (num_dead,), device=flat_z.device)
            reset_vecs = flat_z[pick].detach()
            self.embedding.weight.data[dead_mask] = reset_vecs
            self._ema_embed_sum[dead_mask] = reset_vecs
            self._ema_cluster_size[dead_mask] = 1.0
            self._dead_count[dead_mask] = 0

    def forward(self, z: torch.Tensor):
        indices = self._nearest_indices(z)
        quantized = self.lookup(indices)

        if self.use_ema and self.training:
            flat_z = self._flatten(z)
            self._ema_update(flat_z, indices.reshape(-1))
            # EMA mode: only commitment loss (encoder → codebook)
            vq_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)
        else:
            codebook_loss = F.mse_loss(quantized, z.detach())
            commitment_loss = F.mse_loss(quantized.detach(), z)
            vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized_st = z + (quantized - z).detach()
        encodings = F.one_hot(indices.reshape(-1), num_classes=self.num_embeddings).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs.clamp_min(1e-10))))
        return quantized_st, vq_loss, indices, perplexity


class FeatureConditionalVQVAE(nn.Module):
    """Feature-space conditional VQ-VAE，输入为特征向量，条件为局部类别 id。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embedding_dim: int,
        num_embeddings: int,
        h_dim: int = 512,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.num_embeddings = int(num_embeddings)
        self.h_dim = int(h_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + self.num_classes, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.embedding_dim),
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim + self.num_classes, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.input_dim),
        )

    def _label_one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long()
        return F.one_hot(labels, num_classes=self.num_classes).float()

    def encode(self, x: torch.Tensor, labels: torch.Tensor):
        h = torch.cat([x, self._label_one_hot(labels).to(device=x.device, dtype=x.dtype)], dim=1)
        z_e = self.encoder(h)
        z_e_4d = z_e.unsqueeze(-1).unsqueeze(-1)
        quantized_4d, vq_loss, indices, perplexity = self.quantizer(z_e_4d)
        quantized = quantized_4d.squeeze(-1).squeeze(-1)
        return z_e, quantized, vq_loss, indices, perplexity

    def decode(self, quantized: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self._label_one_hot(labels).to(device=quantized.device, dtype=quantized.dtype)
        return self.decoder(torch.cat([quantized, cond], dim=1))

    def decode_indices(self, indices: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.lookup(indices.to(self.quantizer.embedding.weight.device))
        quantized = quantized.squeeze(-1).squeeze(-1)
        return self.decode(quantized, labels.to(quantized.device))

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        _, quantized, vq_loss, indices, perplexity = self.encode(x, labels)
        recon = self.decode(quantized, labels)
        return recon, vq_loss, indices, perplexity

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _, _, _, indices, _ = self.encode(x, labels)
        return indices

    @torch.no_grad()
    def estimate_loglikelihood(self, x: torch.Tensor, labels: torch.Tensor, S: int = 1, is_batch: int = 32) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        x = x.reshape(x.size(0), -1).contiguous()
        labels = labels.reshape(labels.size(0)).long()
        if x.size(0) == 0:
            return x.new_zeros((0,), dtype=x.dtype)
        _, quantized, vq_loss, _, _ = self.encode(x, labels)
        recon = self.decode(quantized, labels)
        recon_loss = F.mse_loss(recon, x, reduction="none").mean(dim=1)
        return -(recon_loss + float(vq_loss.detach().item()))

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        device: torch.device,
        code_index_pool: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        labels = labels.to(device).long()
        if code_index_pool is not None:
            indices = torch.empty((labels.shape[0], 1, 1), dtype=torch.long, device=device)
            for cls_id in labels.unique().tolist():
                cls_id = int(cls_id)
                pool = code_index_pool[cls_id]
                if pool.numel() == 0:
                    raise ValueError(f"code pool for class {cls_id} is empty")
                mask = labels == cls_id
                choice = torch.randint(pool.shape[0], (int(mask.sum().item()),), device=device)
                sampled = pool.to(device)[choice]
                indices[mask] = sampled
        else:
            indices = torch.randint(self.num_embeddings, (labels.shape[0], 1, 1), device=device)
        quantized = self.quantizer.lookup(indices).squeeze(-1).squeeze(-1)
        return self.decode(quantized, labels)


class TaskReplayVQVAE(nn.Module):
    """条件图像 VQ-VAE，使用离散 codebook 保存局部纹理。"""

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        embedding_dim: int,
        num_embeddings: int,
        base_channels: int = 32,
        channel_multipliers: Optional[List[int]] = None,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.num_embeddings = int(num_embeddings)
        self.base_channels = int(base_channels)
        self.channel_multipliers = channel_multipliers or [1, 2, 4, 8]
        if not self.channel_multipliers:
            raise ValueError("channel_multipliers must not be empty")
        num_stages = len(self.channel_multipliers)
        self.spatial_size = self.image_size // (2 ** num_stages)
        if self.spatial_size <= 0:
            raise ValueError("image_size is too small for the configured VQ-VAE depth")
        if self.image_size % (2 ** num_stages) != 0:
            raise ValueError("image_size must be divisible by 2 ** len(channel_multipliers)")

        enc_in = 3 + self.num_classes
        encoder_layers = []
        prev_channels = enc_in
        self.encoder_channels = []
        for idx, mult in enumerate(self.channel_multipliers):
            out_channels = self.base_channels * mult
            encoder_layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=4, stride=2, padding=1))
            if idx > 0:
                encoder_layers.append(nn.BatchNorm2d(out_channels))
            encoder_layers.append(nn.ReLU(inplace=True))
            prev_channels = out_channels
            self.encoder_channels.append(out_channels)
        encoder_layers.append(nn.Conv2d(prev_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1))
        self.encoder = nn.Sequential(*encoder_layers)

        self.quantizer = VectorQuantizer(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
        )

        decoder_layers = []
        prev_channels = self.embedding_dim + self.num_classes
        decoder_channels = list(reversed(self.encoder_channels))
        for out_channels in decoder_channels:
            decoder_layers.append(nn.ConvTranspose2d(prev_channels, out_channels, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.BatchNorm2d(out_channels))
            decoder_layers.append(nn.ReLU(inplace=True))
            prev_channels = out_channels
        decoder_layers.append(nn.Conv2d(prev_channels, 3, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def _label_map(self, labels: torch.Tensor, height: int, width: int) -> torch.Tensor:
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        return one_hot[:, :, None, None].expand(-1, -1, height, width)

    def encode(self, x: torch.Tensor, labels: torch.Tensor):
        h = torch.cat([x, self._label_map(labels, x.shape[2], x.shape[3])], dim=1)
        z_e = self.encoder(h)
        quantized, vq_loss, indices, perplexity = self.quantizer(z_e)
        return z_e, quantized, vq_loss, indices, perplexity

    def decode(self, quantized: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self._label_map(labels, quantized.shape[2], quantized.shape[3])
        return self.decoder(torch.cat([quantized, cond], dim=1))

    def decode_indices(self, indices: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.lookup(indices.to(self.quantizer.embedding.weight.device))
        return self.decode(quantized, labels.to(quantized.device))

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        _, quantized, vq_loss, indices, perplexity = self.encode(x, labels)
        recon = self.decode(quantized, labels)
        return recon, vq_loss, indices, perplexity

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _, _, _, indices, _ = self.encode(x, labels)
        return indices


class MaskedConv2d(nn.Conv2d):
    """PixelCNN masked convolution."""

    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mask_type not in {"A", "B"}:
            raise ValueError(f"unsupported mask_type: {mask_type}")
        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError("MaskedConv2d requires odd kernel sizes")
        self.mask_type = mask_type
        mask = torch.ones_like(self.weight)
        center_h = self.kernel_size[0] // 2
        center_w = self.kernel_size[1] // 2
        mask[:, :, center_h + 1 :, :] = 0
        mask[:, :, center_h, center_w + 1 :] = 0
        if mask_type == "A":
            mask[:, :, center_h, center_w] = 0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.mask
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PixelCNNResidualBlock(nn.Module):
    def __init__(self, channels: int, num_classes: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = MaskedConv2d(
            "B",
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.class_proj = nn.Linear(num_classes, channels)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, class_one_hot: torch.Tensor) -> torch.Tensor:
        class_bias = self.class_proj(class_one_hot).unsqueeze(-1).unsqueeze(-1)
        h = self.conv(x)
        h = h + class_bias
        h = self.norm(h)
        h = F.relu(h, inplace=True)
        h = self.dropout(h)
        return x + h


class ClassConditionalPixelCNN(nn.Module):
    """Class-conditional PixelCNN prior over VQ-VAE code indices."""

    def __init__(
        self,
        num_embeddings: int,
        num_classes: int,
        hidden_channels: int = 128,
        num_layers: int = 6,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.token_embed = nn.Embedding(self.num_embeddings, self.hidden_channels)
        self.input_conv = MaskedConv2d(
            "A",
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.input_class_proj = nn.Linear(self.num_classes, self.hidden_channels)
        self.blocks = nn.ModuleList(
            [
                PixelCNNResidualBlock(
                    channels=self.hidden_channels,
                    num_classes=self.num_classes,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.num_embeddings, kernel_size=1),
        )

    def forward(self, indices: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        class_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        x = self.token_embed(indices.long()).permute(0, 3, 1, 2).contiguous()
        x = self.input_conv(x) + self.input_class_proj(class_one_hot).unsqueeze(-1).unsqueeze(-1)
        x = F.relu(x, inplace=True)
        for block in self.blocks:
            x = block(x, class_one_hot)
        return self.out(x)

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        labels = labels.to(device)
        samples = torch.zeros((labels.shape[0], height, width), dtype=torch.long, device=device)
        temp = max(float(temperature), 1e-6)
        top_k = max(0, int(top_k))
        for row in range(height):
            for col in range(width):
                logits = self(samples, labels)[:, :, row, col] / temp
                if top_k > 0 and top_k < logits.shape[1]:
                    values, indices = torch.topk(logits, k=top_k, dim=1)
                    probs = F.softmax(values, dim=1)
                    chosen = torch.multinomial(probs, num_samples=1).squeeze(1)
                    next_token = indices.gather(1, chosen.unsqueeze(1)).squeeze(1)
                else:
                    probs = F.softmax(logits, dim=1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                samples[:, row, col] = next_token
        return samples


# ═══════════════════════════════════════════════
#  Task-ID 分类器
# ═══════════════════════════════════════════════
class TaskIDClassifier(nn.Module):
    """
    ResNet18 task-id 分类器。
    输入图像，输出 task_id logits；可选用于对比学习的投影头。
    """

    def __init__(
        self,
        num_tasks: int,
        pretrained: bool = True,
        projection_dim: int = 0,
    ):
        super().__init__()
        backbone = _build_resnet18_backbone(pretrained)
        feat_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_dim = feat_dim
        self.fc_out = nn.Linear(feat_dim, num_tasks)
        self.projection = None
        proj_dim = int(projection_dim)
        if proj_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(feat_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim),
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def classify_from_feats(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc_out(feats)

    def project_from_feats(self, feats: torch.Tensor) -> torch.Tensor:
        if self.projection is None:
            raise RuntimeError("projection head is disabled (projection_dim <= 0)")
        return self.projection(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_from_feats(self.encode(x))


class FeatureTaskIDClassifier(nn.Module):
    """输入冻结 backbone 特征，输出 task-id logits 的 MLP 分类器。"""

    def __init__(
        self,
        num_tasks: int,
        input_dim: int,
        hidden_dim: int = 512,
        hidden_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = max(0, int(hidden_layers))
        self.dropout = max(0.0, float(dropout))

        mlp_layers = []
        in_dim = self.input_dim
        for _ in range(self.hidden_layers):
            mlp_layers.append(nn.Linear(in_dim, self.hidden_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0.0:
                mlp_layers.append(nn.Dropout(p=self.dropout))
            in_dim = self.hidden_dim
        self.encoder = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.feature_dim = int(in_dim)
        self.fc_out = nn.Linear(self.feature_dim, self.num_tasks)

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() != 2:
            feats = feats.reshape(feats.shape[0], -1)
        return self.encoder(feats.float())

    def classify_from_feats(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.encode(feats))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classify_from_feats(feats)
