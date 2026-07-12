"""
全量基线训练 & 评估
  FULL_1: 6 专家 + 线性 gate + top-2，单 PrototypeHead(6 类)
  FULL_2: 6 专家按任务固定分配，每任务独立 PrototypeHead(2 类)，全量数据同时训练
"""

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import compute_metrics
from .models import FullMoEResNet, IncrementalMoEResNet, PrototypeHead


# ═══════════════════════════════════════════════
#  训练
# ═══════════════════════════════════════════════
def train_full(
    model: FullMoEResNet,
    head: PrototypeHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    min_delta: float,
    device: torch.device,
    log_prefix: str = "",
) -> float:
    """训练全量 MoE 模型，返回最佳验证集准确率。"""
    if len(train_loader.dataset) == 0:
        raise ValueError("full training set is empty")
    if len(val_loader.dataset) == 0:
        raise ValueError("full validation set is empty")

    params = [p for p in model.moe_blocks.parameters() if p.requires_grad]
    params += list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = None
    epoch_logs = []

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        head.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            features = model(images)
            logits = head(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        # ---- validate ----
        val_metrics = evaluate_full(model, head, val_loader, device)
        val_acc = val_metrics["overall_acc"]
        # val loss
        model.eval(); head.eval()
        vl = 0.0; vn = 0
        with torch.no_grad():
            for vi, vl_ in val_loader:
                vi, vl_ = vi.to(device), vl_.to(device)
                vl += F.cross_entropy(head(model(vi)), vl_, reduction="sum").item()
                vn += vl_.size(0)
        val_loss = vl / vn if vn else 0.0

        print(
            f"{log_prefix} Epoch {epoch:>3d}/{epochs}: "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}"
        )
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {
                "moe": {k: v.clone() for k, v in model.moe_blocks.state_dict().items()},
                "head": {k: v.clone() for k, v in head.state_dict().items()},
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"{log_prefix} Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.moe_blocks.load_state_dict(best_state["moe"])
        head.load_state_dict(best_state["head"])

    return best_val_loss, best_epoch, epoch_logs


# ═══════════════════════════════════════════════
#  评估
# ═══════════════════════════════════════════════
@torch.no_grad()
def evaluate_full(
    model: FullMoEResNet,
    head: PrototypeHead,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str] = None,
    verbose: bool = False,
) -> Dict:
    """评估全量模型，返回指标 dict。verbose=True 时打印逐类结果。"""
    model.eval()
    head.eval()

    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = head(model(images))
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_preds, all_labels, class_names, verbose=verbose)
    return metrics


# ═══════════════════════════════════════════════
#  FULL_2: 固定路由 — 训练
# ═══════════════════════════════════════════════
def _build_class_to_task(task_splits: List[List[int]]) -> Dict[int, int]:
    """全局类 → task_id 映射"""
    c2t = {}
    for tid, classes in enumerate(task_splits):
        for c in classes:
            c2t[c] = tid
    return c2t


def train_full_fixed(
    model: IncrementalMoEResNet,
    heads: List[PrototypeHead],
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_splits: List[List[int]],
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    min_delta: float,
    device: torch.device,
    log_prefix: str = "",
) -> float:
    """
    全量数据同时训练，但专家按任务固定路由。
    每个 batch 按类拆分到对应任务的专家，合并 loss 反传。
    """
    num_tasks = len(task_splits)
    if len(train_loader.dataset) == 0:
        raise ValueError("full fixed-route training set is empty")
    if len(val_loader.dataset) == 0:
        raise ValueError("full fixed-route validation set is empty")

    # 全部专家 + 全部 head 都可训练
    params = [p for p in model.moe_blocks.parameters() if p.requires_grad]
    for h in heads:
        params += list(h.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = None
    epoch_logs = []

    for epoch in range(1, epochs + 1):
        model.train()
        for h in heads:
            h.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 按任务拆分 batch
            loss = torch.tensor(0.0, device=device)
            batch_has_samples = False
            for tid in range(num_tasks):
                task_classes = task_splits[tid]
                g2l = {g: l for l, g in enumerate(task_classes)}

                mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
                for c in task_classes:
                    mask |= (labels == c)
                if mask.sum() == 0:
                    continue

                batch_has_samples = True
                feats = model(images[mask], tid)
                logits = heads[tid](feats)
                local_labels = torch.tensor(
                    [g2l[l.item()] for l in labels[mask]], device=device
                )
                loss = loss + F.cross_entropy(logits, local_labels)

                correct += (logits.argmax(1) == local_labels).sum().item()
                total += mask.sum().item()

            if batch_has_samples:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_acc, val_loss = _evaluate_full_fixed_loss(
            model, heads, val_loader, task_splits, device
        )
        print(
            f"{log_prefix} Epoch {epoch:>3d}/{epochs}: "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}"
        )
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {
                "moe": {k: v.clone() for k, v in model.moe_blocks.state_dict().items()},
                "heads": [{k: v.clone() for k, v in h.state_dict().items()} for h in heads],
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"{log_prefix} Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.moe_blocks.load_state_dict(best_state["moe"])
        for h, sd in zip(heads, best_state["heads"]):
            h.load_state_dict(sd)

    return best_val_loss, best_epoch, epoch_logs


@torch.no_grad()
def _evaluate_full_fixed_loss(
    model: IncrementalMoEResNet,
    heads: List[PrototypeHead],
    loader: DataLoader,
    task_splits: List[List[int]],
    device: torch.device,
) -> tuple:
    model.eval()
    for h in heads:
        h.eval()

    num_tasks = len(task_splits)
    correct, total = 0, 0
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        batch_loss = torch.tensor(0.0, device=device)
        batch_count = 0
        for tid in range(num_tasks):
            task_classes = task_splits[tid]
            g2l = {g: l for l, g in enumerate(task_classes)}
            mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
            for c in task_classes:
                mask |= (labels == c)
            if mask.sum() == 0:
                continue

            feats = model(images[mask], tid)
            logits = heads[tid](feats)
            local_labels = torch.tensor([g2l[l.item()] for l in labels[mask]], device=device)
            loss = F.cross_entropy(logits, local_labels, reduction="sum")
            batch_loss = batch_loss + loss
            correct += (logits.argmax(1) == local_labels).sum().item()
            total += mask.sum().item()
            batch_count += mask.sum().item()

        if batch_count > 0:
            total_loss += batch_loss.item()

    return correct / max(total, 1), total_loss / max(total, 1)


# ═══════════════════════════════════════════════
#  FULL_2: 固定路由 — 评估
# ═══════════════════════════════════════════════
@torch.no_grad()
def evaluate_full_fixed(
    model: IncrementalMoEResNet,
    heads: List[PrototypeHead],
    loader: DataLoader,
    task_splits: List[List[int]],
    device: torch.device,
    class_names: List[str] = None,
    verbose: bool = False,
    use_oracle_task: bool = False,
) -> Dict:
    """
    FULL_2 推理。

    use_oracle_task=False (默认):
        对每个样本，逐任务前向 → 取 max logit 最高的任务预测。
    use_oracle_task=True:
        用真实标签确定 task_id → 直接路由到对应专家（性能上界）。

    返回指标 dict。
    """
    model.eval()
    for h in heads:
        h.eval()

    num_tasks = len(task_splits)
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        B = images.size(0)

        if use_oracle_task:
            # ── Oracle: 用真实标签推出 task_id，直接路由 ──
            batch_preds = [0] * B
            for tid in range(num_tasks):
                task_classes = task_splits[tid]
                mask = torch.zeros(B, dtype=torch.bool, device=device)
                for c in task_classes:
                    mask |= (labels == c)
                if mask.sum() == 0:
                    continue

                feats = model(images[mask], tid)
                logits = heads[tid](feats)
                local_pred = logits.argmax(1)
                global_pred = [task_classes[l.item()] for l in local_pred]

                idx_list = mask.nonzero(as_tuple=True)[0].tolist()
                for j, gp in zip(idx_list, global_pred):
                    batch_preds[j] = gp

            all_preds.extend(batch_preds)
            all_labels.extend(labels.cpu().tolist())

        else:
            # ── Max-confidence: 映射到全局类空间后统一比较 ──
            score_dim = max(int(c) for task in task_splits for c in task) + 1
            global_scores = torch.full((B, score_dim), float("-inf"), device=device)

            for tid in range(num_tasks):
                feats = model(images, tid)
                logits = heads[tid](feats)
                for local_i, global_cls in enumerate(task_splits[tid]):
                    global_scores[:, int(global_cls)] = logits[:, local_i]

            final_pred = global_scores.argmax(dim=1)
            all_preds.extend(final_pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    if verbose:
        mode_str = "oracle_task" if use_oracle_task else "max_confidence"
        print(f"\n  [{mode_str}]")

    metrics = compute_metrics(all_preds, all_labels, class_names, verbose=verbose)
    return metrics
