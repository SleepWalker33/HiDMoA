"""
生成器后端统一封装:
  - vae
  - cVAE
  - vqvae
  - diffusion_lora
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import IMAGENET_MEAN, IMAGENET_STD
from .train import generate_pseudo_images, generate_vqvae_pseudo_images, train_vae, train_vqvae


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


def _artifact_dir(run_dir: str, backend: str, session_id: int) -> str:
    path = os.path.join(run_dir, "generator_artifacts", backend, f"session{session_id + 1}")
    os.makedirs(path, exist_ok=True)
    return path


def _default_prompt(class_name: str) -> str:
    return f"close-up steel surface with {class_name.replace('_', ' ').lower()} defect, industrial texture"


def _build_prompts(class_ids: List[int], class_names: List[str], prompt_templates: Dict) -> Dict[int, str]:
    prompts = {}
    prompt_templates = prompt_templates or {}
    for cid in class_ids:
        key = str(cid)
        prompts[cid] = prompt_templates.get(key, _default_prompt(class_names[cid]))
    return prompts


def _diffusion_lora_cfg(cfg_t: Dict) -> Dict:
    if "diffusion_lora" in cfg_t:
        cfg = dict(cfg_t["diffusion_lora"])
        cfg["image_size"] = int(cfg["image_size"])
        cfg["train_batch_size"] = int(cfg["train_batch_size"])
        cfg["gradient_accumulation_steps"] = int(cfg["gradient_accumulation_steps"])
        cfg["lr"] = float(cfg["lr"])
        cfg["max_train_steps"] = int(cfg["max_train_steps"])
        cfg["rank"] = int(cfg["rank"])
        cfg["weight_decay"] = float(cfg["weight_decay"])
        cfg["num_inference_steps"] = int(cfg["num_inference_steps"])
        cfg["guidance_scale"] = float(cfg["guidance_scale"])
        cfg["generated_per_class"] = int(cfg["generated_per_class"])
        cfg["sample_batch_size"] = int(cfg["sample_batch_size"])
        cfg["prompt_templates"] = dict(cfg["prompt_templates"])
        return cfg
    return {
        "base_model_path": cfg_t["diffusion_lora_base_model_path"],
        "base_model_hf_id": cfg_t["diffusion_lora_base_model_hf_id"],
        "local_files_only": bool(cfg_t["diffusion_lora_local_files_only"]),
        "image_size": int(cfg_t["diffusion_lora_image_size"]),
        "train_batch_size": int(cfg_t["diffusion_lora_train_batch_size"]),
        "gradient_accumulation_steps": int(cfg_t["diffusion_lora_gradient_accumulation_steps"]),
        "lr": float(cfg_t["diffusion_lora_lr"]),
        "max_train_steps": int(cfg_t["diffusion_lora_max_train_steps"]),
        "rank": int(cfg_t["diffusion_lora_rank"]),
        "weight_decay": float(cfg_t["diffusion_lora_weight_decay"]),
        "mixed_precision": cfg_t["diffusion_lora_mixed_precision"],
        "num_inference_steps": int(cfg_t["diffusion_lora_num_inference_steps"]),
        "guidance_scale": float(cfg_t["diffusion_lora_guidance_scale"]),
        "generated_per_class": int(cfg_t["diffusion_lora_generated_per_class"]),
        "sample_batch_size": int(cfg_t["diffusion_lora_sample_batch_size"]),
        "prompt_templates": dict(cfg_t["diffusion_lora_prompt_templates"]),
    }


def _vae_cfg(cfg_t: Dict) -> Dict:
    if "vae" in cfg_t:
        cfg = dict(cfg_t["vae"])
        cfg["image_size"] = int(cfg["image_size"])
        cfg["latent_dim"] = int(cfg["latent_dim"])
        cfg["base_channels"] = int(cfg["base_channels"])
        cfg["channel_multipliers"] = list(cfg["channel_multipliers"])
        cfg["epochs"] = int(cfg["epochs"])
        cfg["early_stopping_patience"] = int(cfg["early_stopping_patience"])
        cfg["early_stopping_min_delta"] = float(cfg["early_stopping_min_delta"])
        cfg["lr"] = float(cfg["lr"])
        cfg["weight_decay"] = float(cfg["weight_decay"])
        cfg["beta_kl"] = float(cfg["beta_kl"])
        cfg["kl_warmup_epochs"] = int(cfg["kl_warmup_epochs"])
        cfg["recon_weight"] = float(cfg["recon_weight"])
        cfg["l1_weight"] = float(cfg["l1_weight"])
        cfg["perceptual_weight"] = float(cfg["perceptual_weight"])
        cfg["perceptual_pretrained"] = bool(cfg["perceptual_pretrained"])
        cfg["perceptual_layers"] = int(cfg["perceptual_layers"])
        cfg["latent_pool_noise_std"] = float(cfg["latent_pool_noise_std"])
        cfg["generated_per_class"] = int(cfg["generated_per_class"])
        cfg["batch_size"] = int(cfg["batch_size"])
        return cfg
    return {
        "image_size": int(cfg_t["cvae_image_size"]),
        "latent_dim": int(cfg_t["vae_latent_dim"]),
        "base_channels": int(cfg_t["vae_base_channels"]),
        "channel_multipliers": list(cfg_t["vae_channel_multipliers"]),
        "epochs": int(cfg_t["vae_epochs"]),
        "early_stopping_patience": int(cfg_t["vae_early_stopping_patience"]),
        "early_stopping_min_delta": float(cfg_t["vae_early_stopping_min_delta"]),
        "lr": float(cfg_t["vae_lr"]),
        "weight_decay": float(cfg_t["vae_weight_decay"]),
        "beta_kl": float(cfg_t["vae_beta_kl"]),
        "kl_warmup_epochs": int(cfg_t["vae_kl_warmup_epochs"]),
        "recon_weight": float(cfg_t["vae_recon_weight"]),
        "l1_weight": float(cfg_t.get("vae_l1_weight", 1.0)),
        "perceptual_weight": float(cfg_t.get("vae_perceptual_weight", 0.0)),
        "perceptual_pretrained": bool(cfg_t.get("vae_perceptual_pretrained", True)),
        "perceptual_layers": int(cfg_t.get("vae_perceptual_layers", 3)),
        "latent_pool_noise_std": float(cfg_t.get("vae_latent_pool_noise_std", 0.0)),
        "generated_per_class": int(cfg_t["vae_generated_per_class"]),
        "batch_size": int(cfg_t["vae_batch_size"]),
    }


def _vqvae_cfg(cfg_t: Dict) -> Dict:
    if "vqvae" in cfg_t:
        cfg = dict(cfg_t["vqvae"])
        cfg["image_size"] = int(cfg["image_size"])
        cfg["embedding_dim"] = int(cfg["embedding_dim"])
        cfg["num_embeddings"] = int(cfg["num_embeddings"])
        cfg["base_channels"] = int(cfg["base_channels"])
        cfg["channel_multipliers"] = list(cfg["channel_multipliers"])
        cfg["commitment_cost"] = float(cfg["commitment_cost"])
        cfg["codebook_weight"] = float(cfg["codebook_weight"])
        cfg["ema_decay"] = float(cfg["ema_decay"])
        cfg["epochs"] = int(cfg["epochs"])
        cfg["early_stopping_patience"] = int(cfg["early_stopping_patience"])
        cfg["early_stopping_min_delta"] = float(cfg["early_stopping_min_delta"])
        cfg["lr"] = float(cfg["lr"])
        cfg["weight_decay"] = float(cfg["weight_decay"])
        cfg["recon_weight"] = float(cfg["recon_weight"])
        cfg["perceptual_weight"] = float(cfg["perceptual_weight"])
        cfg["perceptual_pretrained"] = bool(cfg["perceptual_pretrained"])
        cfg["perceptual_layers"] = int(cfg["perceptual_layers"])
        cfg["generated_per_class"] = int(cfg["generated_per_class"])
        cfg["batch_size"] = int(cfg["batch_size"])
        cfg["use_pixelcnn_prior"] = bool(cfg["use_pixelcnn_prior"])
        cfg["pixelcnn_hidden_channels"] = int(cfg["pixelcnn_hidden_channels"])
        cfg["pixelcnn_num_layers"] = int(cfg["pixelcnn_num_layers"])
        cfg["pixelcnn_kernel_size"] = int(cfg["pixelcnn_kernel_size"])
        cfg["pixelcnn_dropout"] = float(cfg["pixelcnn_dropout"])
        cfg["pixelcnn_epochs"] = int(cfg["pixelcnn_epochs"])
        cfg["pixelcnn_early_stopping_patience"] = int(cfg["pixelcnn_early_stopping_patience"])
        cfg["pixelcnn_early_stopping_min_delta"] = float(cfg["pixelcnn_early_stopping_min_delta"])
        cfg["pixelcnn_lr"] = float(cfg["pixelcnn_lr"])
        cfg["pixelcnn_weight_decay"] = float(cfg["pixelcnn_weight_decay"])
        cfg["pixelcnn_batch_size"] = int(cfg["pixelcnn_batch_size"])
        cfg["pixelcnn_sampling_temperature"] = float(cfg["pixelcnn_sampling_temperature"])
        cfg["pixelcnn_sampling_top_k"] = int(cfg["pixelcnn_sampling_top_k"])
        return cfg
    return {
        "image_size": int(cfg_t["vqvae_image_size"]),
        "embedding_dim": int(cfg_t["vqvae_embedding_dim"]),
        "num_embeddings": int(cfg_t["vqvae_num_embeddings"]),
        "base_channels": int(cfg_t["vqvae_base_channels"]),
        "channel_multipliers": list(cfg_t["vqvae_channel_multipliers"]),
        "commitment_cost": float(cfg_t["vqvae_commitment_cost"]),
        "codebook_weight": float(cfg_t["vqvae_codebook_weight"]),
        "ema_decay": float(cfg_t["vqvae_ema_decay"]),
        "epochs": int(cfg_t["vqvae_epochs"]),
        "early_stopping_patience": int(cfg_t["vqvae_early_stopping_patience"]),
        "early_stopping_min_delta": float(cfg_t["vqvae_early_stopping_min_delta"]),
        "lr": float(cfg_t["vqvae_lr"]),
        "weight_decay": float(cfg_t["vqvae_weight_decay"]),
        "recon_weight": float(cfg_t["vqvae_recon_weight"]),
        "perceptual_weight": float(cfg_t["vqvae_perceptual_weight"]),
        "perceptual_pretrained": bool(cfg_t["vqvae_perceptual_pretrained"]),
        "perceptual_layers": int(cfg_t["vqvae_perceptual_layers"]),
        "generated_per_class": int(cfg_t["vqvae_generated_per_class"]),
        "batch_size": int(cfg_t["vqvae_batch_size"]),
        "use_pixelcnn_prior": bool(cfg_t["vqvae_use_pixelcnn_prior"]),
        "pixelcnn_hidden_channels": int(cfg_t["vqvae_pixelcnn_hidden_channels"]),
        "pixelcnn_num_layers": int(cfg_t["vqvae_pixelcnn_num_layers"]),
        "pixelcnn_kernel_size": int(cfg_t["vqvae_pixelcnn_kernel_size"]),
        "pixelcnn_dropout": float(cfg_t["vqvae_pixelcnn_dropout"]),
        "pixelcnn_epochs": int(cfg_t["vqvae_pixelcnn_epochs"]),
        "pixelcnn_early_stopping_patience": int(cfg_t["vqvae_pixelcnn_early_stopping_patience"]),
        "pixelcnn_early_stopping_min_delta": float(cfg_t["vqvae_pixelcnn_early_stopping_min_delta"]),
        "pixelcnn_lr": float(cfg_t["vqvae_pixelcnn_lr"]),
        "pixelcnn_weight_decay": float(cfg_t["vqvae_pixelcnn_weight_decay"]),
        "pixelcnn_batch_size": int(cfg_t["vqvae_pixelcnn_batch_size"]),
        "pixelcnn_sampling_temperature": float(cfg_t["vqvae_pixelcnn_sampling_temperature"]),
        "pixelcnn_sampling_top_k": int(cfg_t["vqvae_pixelcnn_sampling_top_k"]),
    }


def _save_prompt_dataset(dataset_dir: str, images: torch.Tensor, labels: torch.Tensor, prompts: Dict[int, str]):
    os.makedirs(dataset_dir, exist_ok=True)
    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
    images = _denormalize_imagenet(images.float()).cpu()
    with open(metadata_path, "w", encoding="utf-8") as f:
        for idx, (img, label) in enumerate(zip(images, labels)):
            file_name = f"sample_{idx:06d}.png"
            path = os.path.join(dataset_dir, file_name)
            arr = (img.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(path)
            f.write(json.dumps({"file_name": file_name, "text": prompts[int(label.item())]}, ensure_ascii=False) + "\n")


def _save_generated_preview_images(
    images: torch.Tensor,
    labels: torch.Tensor,
    out_dir: str,
    class_names: Optional[List[str]] = None,
    max_per_class: int = 20,
    seed: Optional[int] = None,
):
    if images.numel() == 0 or labels.numel() == 0 or max_per_class <= 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    images = _denormalize_imagenet(images.float().cpu())
    labels = labels.long().cpu()
    rng = np.random.default_rng(seed)

    for cls_id in sorted(labels.unique().tolist()):
        cls_id = int(cls_id)
        cls_indices = torch.nonzero(labels == cls_id, as_tuple=True)[0].cpu().numpy()
        if cls_indices.size == 0:
            continue
        if cls_indices.size > max_per_class:
            chosen = rng.choice(cls_indices, size=max_per_class, replace=False)
        else:
            chosen = cls_indices
        cls_name = class_names[cls_id] if class_names is not None and 0 <= cls_id < len(class_names) else str(cls_id)
        cls_dir = os.path.join(out_dir, f"{cls_id}_{cls_name}")
        os.makedirs(cls_dir, exist_ok=True)
        for save_idx, img_idx in enumerate(sorted(int(i) for i in chosen)):
            img = images[img_idx]
            arr = (img.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(cls_dir, f"sample_{save_idx:03d}.png"))


def train_diffusion_lora_generator(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    class_ids: List[int],
    class_names: List[str],
    cfg: Dict,
    task_id: int,
    run_dir: str,
    device: torch.device,
    log_prefix: str = "",
) -> Dict:
    try:
        from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
        from diffusers.training_utils import compute_snr
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as e:
        raise ImportError(
            "diffusion_lora backend requires `diffusers`, `transformers`, and `peft` to be installed"
        ) from e

    artifact_dir = _artifact_dir(run_dir, "diffusion_lora", task_id)
    lora_dir = os.path.join(artifact_dir, "lora")
    prompts = _build_prompts(class_ids, class_names, cfg.get("prompt_templates", {}))
    os.makedirs(lora_dir, exist_ok=True)
    base_model_path = cfg.get("base_model_path", "")
    if not base_model_path or not os.path.isdir(base_model_path):
        raise FileNotFoundError(
            "diffusion_lora requires a local pretrained model directory in "
            "`train.diffusion_lora_base_model_path`"
        )
    local_files_only = bool(cfg.get("local_files_only", True))

    resolution = cfg["image_size"]
    train_images = _denormalize_imagenet(train_images.float())
    val_images = _denormalize_imagenet(val_images.float())
    train_images = F.interpolate(train_images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    val_images = F.interpolate(val_images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    train_images = train_images * 2.0 - 1.0
    val_images = val_images * 2.0 - 1.0

    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer", local_files_only=local_files_only)
    text_encoder = CLIPTextModel.from_pretrained(
        base_model_path, subfolder="text_encoder", local_files_only=local_files_only
    )
    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", local_files_only=local_files_only)
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", local_files_only=local_files_only)
    noise_scheduler = DDPMScheduler.from_pretrained(
        base_model_path, subfolder="scheduler", local_files_only=local_files_only
    )

    if device.type == "cuda" and cfg.get("mixed_precision", "fp16") == "fp16":
        weight_dtype = torch.float16
    elif device.type == "cuda" and cfg.get("mixed_precision", "fp16") == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    lora_config = LoraConfig(
        r=cfg["rank"],
        lora_alpha=cfg["rank"],
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    def _tokenize(labels: torch.Tensor) -> torch.Tensor:
        texts = [prompts[int(label.item())] for label in labels]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    train_ids = _tokenize(train_labels)
    val_ids = _tokenize(val_labels)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_images, train_ids),
        batch_size=cfg["train_batch_size"],
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_images, val_ids),
        batch_size=cfg["train_batch_size"],
        shuffle=False,
    )

    params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-2))
    grad_accum = max(1, int(cfg.get("gradient_accumulation_steps", 1)))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_epochs = max(1, math.ceil(cfg["max_train_steps"] / steps_per_epoch))
    global_step = 0
    epoch_logs = []

    def _forward_loss(pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        input_ids = input_ids.to(device)
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
            dtype=torch.long,
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = text_encoder(input_ids)[0]
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"unsupported prediction type: {noise_scheduler.config.prediction_type}")
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    for epoch in range(1, total_epochs + 1):
        unet.train()
        optimizer.zero_grad()
        train_loss_sum = 0.0
        train_batches = 0
        for step, (pixel_values, input_ids) in enumerate(train_loader, start=1):
            loss = _forward_loss(pixel_values, input_ids) / grad_accum
            loss.backward()
            if step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            train_loss_sum += loss.item() * grad_accum
            train_batches += 1
            if global_step >= cfg["max_train_steps"]:
                break
        if train_batches % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        unet.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for pixel_values, input_ids in val_loader:
                loss = _forward_loss(pixel_values, input_ids)
                val_loss_sum += loss.item()
                val_batches += 1

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_batches, 1),
            "val_loss": val_loss_sum / max(val_batches, 1),
        })
        cur = epoch_logs[-1]
        print(
            f"{log_prefix} diffusion_lora epoch {epoch:>3d}/{total_epochs}: "
            f"train_loss={cur['train_loss']:.4f}  val_loss={cur['val_loss']:.4f}"
        )
        if global_step >= cfg["max_train_steps"]:
            break

    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(save_directory=lora_dir, unet_lora_layers=unet_lora_state_dict)

    return {
        "type": "diffusion_lora",
        "class_ids": list(class_ids),
        "class_prompts": {str(k): v for k, v in prompts.items()},
        "base_model_path": base_model_path,
        "lora_dir": lora_dir,
        "num_inference_steps": cfg["num_inference_steps"],
        "guidance_scale": cfg["guidance_scale"],
        "sample_batch_size": max(1, int(cfg.get("sample_batch_size", 1))),
        "mixed_precision": cfg.get("mixed_precision", "fp16"),
        "epoch_logs": epoch_logs,
        "local_files_only": local_files_only,
    }


@torch.no_grad()
def sample_diffusion_lora_images(
    state: Dict,
    num_per_class: int,
    target_image_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        from diffusers import DiffusionPipeline
    except ImportError as e:
        raise ImportError("diffusion_lora sampling requires `diffusers` to be installed") from e

    mixed_precision = state.get("mixed_precision", "fp16")
    if device.type == "cuda" and mixed_precision == "fp16":
        pipe_dtype = torch.float16
    elif device.type == "cuda" and mixed_precision == "bf16":
        pipe_dtype = torch.bfloat16
    else:
        pipe_dtype = torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        state["base_model_path"],
        local_files_only=state.get("local_files_only", True),
        torch_dtype=pipe_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.load_lora_weights(state["lora_dir"])
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    images_all, labels_all = [], []
    prompts = {int(k): v for k, v in state["class_prompts"].items()}
    sample_batch_size = max(1, int(state.get("sample_batch_size", 1)))
    for global_cls in state["class_ids"]:
        batch_parts = []
        remaining = num_per_class
        while remaining > 0:
            cur_bs = min(sample_batch_size, remaining)
            batch_prompts = [prompts[int(global_cls)]] * cur_bs
            outputs = pipe(
                prompt=batch_prompts,
                num_inference_steps=state["num_inference_steps"],
                guidance_scale=state["guidance_scale"],
            ).images
            cur_batch = []
            for img in outputs:
                arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                ten = torch.from_numpy(arr).permute(2, 0, 1)
                cur_batch.append(ten)
            batch_parts.append(torch.stack(cur_batch, dim=0))
            remaining -= cur_bs
            if device.type == "cuda":
                torch.cuda.empty_cache()
        batch = torch.cat(batch_parts, dim=0)
        batch = F.interpolate(batch, size=(target_image_size, target_image_size), mode="bilinear", align_corners=False)
        batch = _normalize_imagenet(batch)
        labels = torch.full((num_per_class,), global_cls, dtype=torch.long)
        images_all.append(batch.cpu())
        labels_all.append(labels)
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(images_all), torch.cat(labels_all)


def train_generator(
    generator_type: str,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    class_ids: List[int],
    class_names: List[str],
    task_id: int,
    cfg_t: Dict,
    run_dir: str,
    device: torch.device,
    log_prefix: str = "",
) -> Dict:
    if generator_type in {"vae", "cvae"}:
        vae_cfg = _vae_cfg(cfg_t)
        model_name = "VAE" if generator_type == "vae" else "cVAE"
        state = train_vae(
            train_images=train_images,
            train_labels=train_labels,
            val_images=val_images,
            val_labels=val_labels,
            class_ids=class_ids,
            image_size=vae_cfg["image_size"],
            latent_dim=vae_cfg["latent_dim"],
            base_channels=vae_cfg["base_channels"],
            channel_multipliers=vae_cfg["channel_multipliers"],
            epochs=vae_cfg["epochs"],
            early_stopping_patience=vae_cfg["early_stopping_patience"],
            early_stopping_min_delta=vae_cfg["early_stopping_min_delta"],
            lr=vae_cfg["lr"],
            weight_decay=vae_cfg["weight_decay"],
            beta_kl=vae_cfg["beta_kl"],
            kl_warmup_epochs=vae_cfg["kl_warmup_epochs"],
            recon_weight=vae_cfg["recon_weight"],
            l1_weight=vae_cfg["l1_weight"],
            perceptual_weight=vae_cfg["perceptual_weight"],
            perceptual_pretrained=vae_cfg["perceptual_pretrained"],
            perceptual_layers=vae_cfg["perceptual_layers"],
            latent_pool_noise_std=vae_cfg["latent_pool_noise_std"],
            batch_size=vae_cfg["batch_size"],
            device=device,
            log_prefix=log_prefix,
            model_name=model_name,
        )
        state["type"] = "vae" if generator_type == "vae" else "cvae"
        return state

    if generator_type == "vqvae":
        vqvae_cfg = _vqvae_cfg(cfg_t)
        state = train_vqvae(
            train_images=train_images,
            train_labels=train_labels,
            val_images=val_images,
            val_labels=val_labels,
            class_ids=class_ids,
            image_size=vqvae_cfg["image_size"],
            embedding_dim=vqvae_cfg["embedding_dim"],
            num_embeddings=vqvae_cfg["num_embeddings"],
            base_channels=vqvae_cfg["base_channels"],
            channel_multipliers=vqvae_cfg["channel_multipliers"],
            commitment_cost=vqvae_cfg["commitment_cost"],
            codebook_weight=vqvae_cfg["codebook_weight"],
            ema_decay=vqvae_cfg["ema_decay"],
            epochs=vqvae_cfg["epochs"],
            early_stopping_patience=vqvae_cfg["early_stopping_patience"],
            early_stopping_min_delta=vqvae_cfg["early_stopping_min_delta"],
            lr=vqvae_cfg["lr"],
            weight_decay=vqvae_cfg["weight_decay"],
            recon_weight=vqvae_cfg["recon_weight"],
            perceptual_weight=vqvae_cfg["perceptual_weight"],
            perceptual_pretrained=vqvae_cfg["perceptual_pretrained"],
            perceptual_layers=vqvae_cfg["perceptual_layers"],
            use_pixelcnn_prior=vqvae_cfg["use_pixelcnn_prior"],
            pixelcnn_hidden_channels=vqvae_cfg["pixelcnn_hidden_channels"],
            pixelcnn_num_layers=vqvae_cfg["pixelcnn_num_layers"],
            pixelcnn_kernel_size=vqvae_cfg["pixelcnn_kernel_size"],
            pixelcnn_dropout=vqvae_cfg["pixelcnn_dropout"],
            pixelcnn_epochs=vqvae_cfg["pixelcnn_epochs"],
            pixelcnn_early_stopping_patience=vqvae_cfg["pixelcnn_early_stopping_patience"],
            pixelcnn_early_stopping_min_delta=vqvae_cfg["pixelcnn_early_stopping_min_delta"],
            pixelcnn_lr=vqvae_cfg["pixelcnn_lr"],
            pixelcnn_weight_decay=vqvae_cfg["pixelcnn_weight_decay"],
            pixelcnn_batch_size=vqvae_cfg["pixelcnn_batch_size"],
            pixelcnn_sampling_temperature=vqvae_cfg["pixelcnn_sampling_temperature"],
            pixelcnn_sampling_top_k=vqvae_cfg["pixelcnn_sampling_top_k"],
            batch_size=vqvae_cfg["batch_size"],
            device=device,
            log_prefix=log_prefix,
        )
        state["type"] = "vqvae"
        return state

    if generator_type == "diffusion_lora":
        return train_diffusion_lora_generator(
            train_images=train_images,
            train_labels=train_labels,
            val_images=val_images,
            val_labels=val_labels,
            class_ids=class_ids,
            class_names=class_names,
            cfg=_diffusion_lora_cfg(cfg_t),
            task_id=task_id,
            run_dir=run_dir,
            device=device,
            log_prefix=log_prefix,
        )

    raise ValueError(f"unsupported generator_type: {generator_type}")


def sample_generator_images(
    generator_state: Dict,
    num_per_class: int,
    target_image_size: int,
    device: torch.device,
    save_preview_dir: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    max_save_per_class: int = 20,
    save_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gen_type = generator_state.get("type", "cvae")
    if gen_type in {"vae", "cvae"}:
        images, labels = generate_pseudo_images(
            generator_state,
            num_per_class=num_per_class,
            device=device,
            target_image_size=target_image_size,
        )
    elif gen_type == "vqvae":
        images, labels = generate_vqvae_pseudo_images(
            generator_state,
            num_per_class=num_per_class,
            device=device,
            target_image_size=target_image_size,
        )
    elif gen_type == "diffusion_lora":
        images, labels = sample_diffusion_lora_images(
            generator_state,
            num_per_class=num_per_class,
            target_image_size=target_image_size,
            device=device,
        )
    else:
        raise ValueError(f"unsupported generator state type: {gen_type}")

    if save_preview_dir:
        _save_generated_preview_images(
            images=images,
            labels=labels,
            out_dir=save_preview_dir,
            class_names=class_names,
            max_per_class=max_save_per_class,
            seed=save_seed,
        )
    return images, labels
