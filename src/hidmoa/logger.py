"""训练结果保存结构（仅 HiDMoA）。"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_root_run_dir(base_dir: str = "runs") -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def profile_subdir(profile: str, seed: int, multi_seed: bool) -> str:
    if multi_seed:
        return f"{profile}/seed_{int(seed)}"
    return profile


_REPORT_SECTION_PROFILE = {
    "incremental_2": "HiDMoA",
    "HiDMoA": "HiDMoA",
}


def create_profile_run_dir(root_run_dir: str, profile_dir: str) -> str:
    run_dir = os.path.join(root_run_dir, profile_dir)
    os.makedirs(run_dir, exist_ok=True)
    for sub in ("model", "taskid", "generator"):
        os.makedirs(os.path.join(run_dir, "plot", sub), exist_ok=True)
    return run_dir


def save_config(run_dir: str, config_dict: dict):
    path = os.path.join(run_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)


def save_summary(run_dir: str, summary: dict):
    path = os.path.join(run_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


class TrainLogger:
    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "train.txt")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("# training log\n")

    def log_epoch_table(self, component: str, session: int, epoch_logs: List[dict]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"\n## {component}\n")
            f.write(f"# session{session + 1}\n")
            for row in epoch_logs:
                fields = [f"epoch={row.get('epoch', 0)}"]
                for key, value in row.items():
                    if key == "epoch":
                        continue
                    if isinstance(value, float):
                        fields.append(f"{key}={value:.6f}")
                    else:
                        fields.append(f"{key}={value}")
                f.write("\t".join(fields) + "\n")


def plot_loss_curves(
    run_dir: str,
    epoch_logs: List[dict],
    profile: str,
    plot_group: str,
    session: Optional[int] = None,
):
    if not epoch_logs:
        return

    epochs = [r["epoch"] for r in epoch_logs]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in epoch_logs[0].keys():
        if key == "epoch":
            continue
        if "loss" not in key and key != "beta_kl":
            continue
        series = [r.get(key, 0.0) for r in epoch_logs]
        linestyle = "-"
        if key.startswith("val_"):
            linestyle = "--"
        elif key == "beta_kl":
            linestyle = "-."
        ax.plot(epochs, series, marker="o", markersize=3, linestyle=linestyle, label=key)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Metric")
    title = f"{profile}"
    if session is not None:
        title += f" session{session + 1}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    suffix = f"_session{session + 1}" if session is not None else ""
    path = os.path.join(run_dir, "plot", plot_group, f"loss_{profile}{suffix}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _per_class_entry(per_class: dict, cid: int) -> dict:
    """Lookup per-class metrics; JSON round-trip uses string keys ('0' not 0)."""
    if not isinstance(per_class, dict):
        return {}
    entry = per_class.get(cid, per_class.get(str(cid), {}))
    return entry if isinstance(entry, dict) else {}


def _metric_row(metrics: dict, class_ids: List[int], key: str) -> str:
    vals = []
    per_class = metrics.get("per_class", {})
    for cid in class_ids:
        val = _per_class_entry(per_class, cid).get(key, "NA")
        if isinstance(val, (int, float)):
            vals.append(f"{float(val):.4f}")
        else:
            vals.append(str(val))
    return "\t".join(vals)


def _format_scalar(v):
    if isinstance(v, (int, float)):
        return f"{float(v):.4f}"
    return str(v)


def _is_numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


_SESSION_TITLE_RE = re.compile(r"^session(\d+)$")
_SESSION_SUFFIX_RE = re.compile(r"^session(\d+)_(linear|knn)$")
_SESSION_AVG_SUFFIX_RE = re.compile(r"^session_avg_(linear|knn)$")
_SESSION_SUFFIX_ORDER = {"": 0, "linear": 1, "knn": 2}


def _parse_session_num(title: str) -> Optional[int]:
    """Session index for plain #sessionN titles (backward compatible)."""
    parts = _parse_session_title(str(title))
    if parts is None or parts[0] != "session":
        return None
    return int(parts[1])


def _parse_session_title(title: str) -> Optional[Tuple[str, int, str]]:
    """
    Parse report item titles.

    Returns (kind, session_num, suffix):
      - ("session", N, "")       -> #sessionN
      - ("session", N, "linear") -> #sessionN_linear
      - ("session", N, "knn")    -> #sessionN_knn
      - ("avg", 0, "")           -> #session_avg
      - ("avg", 0, "linear")     -> #session_avg_linear
      - ("avg", 0, "knn")        -> #session_avg_knn
    """
    s = str(title)
    if s == "session_avg":
        return ("avg", 0, "")
    m = _SESSION_AVG_SUFFIX_RE.match(s)
    if m:
        return ("avg", 0, m.group(1))
    m = _SESSION_SUFFIX_RE.match(s)
    if m:
        return ("session", int(m.group(1)), m.group(2))
    m = _SESSION_TITLE_RE.match(s)
    if m:
        return ("session", int(m.group(1)), "")
    return None


def _session_avg_title(suffix: str) -> str:
    return "session_avg" if suffix == "" else f"session_avg_{suffix}"


def _average_numeric_metrics(metrics_list: List[dict]) -> dict:
    """Arithmetic mean over multiple session metric dicts (single seed)."""
    if not metrics_list:
        return {}

    all_cids = set()
    scalar_keys = set()
    for metrics in metrics_list:
        per_class = metrics.get("per_class", {})
        for cid in per_class:
            try:
                all_cids.add(int(cid))
            except (TypeError, ValueError):
                pass
        for key in metrics:
            if key != "per_class":
                scalar_keys.add(key)

    out: dict = {"per_class": {}}
    for cid in sorted(all_cids):
        out["per_class"][cid] = {}
        for mkey in ("acc", "f1", "precision", "recall"):
            vals = []
            for metrics in metrics_list:
                per_class = metrics.get("per_class", {})
                entry = per_class.get(cid, per_class.get(str(cid), {}))
                v = entry.get(mkey, "NA")
                if _is_numeric(v):
                    vals.append(float(v))
            out["per_class"][cid][mkey] = float(np.mean(vals)) if vals else "NA"

    for skey in sorted(scalar_keys):
        vals = []
        for metrics in metrics_list:
            v = metrics.get(skey, "NA")
            if _is_numeric(v):
                vals.append(float(v))
        if vals:
            out[skey] = float(np.mean(vals))
    return out


def _compute_session_avg_item(session_items: List[dict]) -> Optional[dict]:
    if not session_items:
        return None
    class_ids = sorted(
        {
            int(c)
            for item in session_items
            for c in item.get("class_ids", [])
        }
    )
    avg_metrics = _average_numeric_metrics([item.get("metrics", {}) for item in session_items])
    return {"title": "session_avg", "metrics": avg_metrics, "class_ids": class_ids}


def append_session_avg_to_items(items: List[dict]) -> List[dict]:
    """
    Insert session_avg block(s) after the last session row(s).

    - Plain methods: one #session_avg (mean over #session1..N).
    - Suffixed tracks: #session_avg_<suffix> (mean over each track).
    """
    existing_avg = {
        str(it.get("title", ""))
        for it in items
        if str(it.get("title", "")).startswith("session_avg")
    }

    groups: Dict[str, List[Tuple[int, int, dict]]] = defaultdict(list)
    for idx, item in enumerate(items):
        parsed = _parse_session_title(str(item.get("title", "")))
        if parsed is None or parsed[0] != "session":
            continue
        num, suffix = parsed[1], parsed[2]
        groups[suffix].append((num, idx, item))

    if not groups:
        return items

    out = list(items)
    inserts: List[Tuple[int, dict]] = []
    for suffix, indexed_sessions in groups.items():
        avg_title = _session_avg_title(suffix)
        if avg_title in existing_avg:
            continue
        indexed_sessions.sort(key=lambda x: x[0])
        avg_item = _compute_session_avg_item([x[2] for x in indexed_sessions])
        if avg_item is None:
            continue
        avg_item["title"] = avg_title
        last_session_idx = max(x[1] for x in indexed_sessions)
        inserts.append((last_session_idx + 1, avg_item))

    for pos, avg_item in sorted(inserts, key=lambda x: x[0], reverse=True):
        out.insert(pos, avg_item)
    return out


def _sort_report_items(items: List[dict]) -> List[dict]:
    sessions: List[Tuple[int, int, dict]] = []
    avgs: List[Tuple[int, dict]] = []
    others: List[dict] = []
    for item in items:
        title = str(item.get("title", ""))
        parsed = _parse_session_title(title)
        if parsed is None:
            others.append(item)
            continue
        kind, num, suffix = parsed
        if kind == "avg":
            avgs.append((_SESSION_SUFFIX_ORDER.get(suffix, 9), item))
        else:
            sessions.append((num, _SESSION_SUFFIX_ORDER.get(suffix, 9), item))
    sessions.sort(key=lambda x: (x[0], x[1]))
    avgs.sort(key=lambda x: x[0])
    return [x[2] for x in sessions] + others + [x[1] for x in avgs]


def finalize_report_section(section: dict) -> dict:
    items = append_session_avg_to_items(list(section.get("items", [])))
    return {**section, "items": _sort_report_items(items)}


def _format_mean_std(values: List) -> str:
    nums = [float(v) for v in values if _is_numeric(v)]
    if not nums:
        if values and all(v == "NA" for v in values):
            return "NA"
        return str(values[0]) if len(values) == 1 else "NA"
    if len(nums) == 1:
        return f"{nums[0]:.4f}"
    mean = float(np.mean(nums))
    std = float(np.std(nums, ddof=1)) if len(nums) > 1 else 0.0
    return f"{mean:.4f}±{std:.4f}"


def _aggregate_metric_row(
    metrics_list: List[dict],
    class_ids: List[int],
    key: str,
) -> str:
    cells = []
    for cid in class_ids:
        vals = []
        for metrics in metrics_list:
            per_class = metrics.get("per_class", {})
            entry = per_class.get(cid, per_class.get(str(cid), {}))
            vals.append(entry.get(key, "NA"))
        cells.append(_format_mean_std(vals))
    return "\t".join(cells)


def _aggregate_scalar(metrics_list: List[dict], key: str) -> str:
    vals = [m.get(key, "NA") for m in metrics_list if key in m]
    if not vals:
        return "NA"
    return _format_mean_std(vals)


def aggregate_report_section(section_name: str, seed_runs: List[dict]) -> dict:
    """
    Merge multiple single-seed report sections into one with mean±sd cells.

    session_avg: per seed, mean(session1..N); across seeds, mean±std on those
    session_avg scalars (not the mean of per-session stds).
    """
    if not seed_runs:
        return {"name": section_name, "items": []}

    title_to_metrics: Dict[str, List[dict]] = defaultdict(list)
    title_to_class_ids: Dict[str, List[int]] = {}
    title_order: List[str] = []

    for run in seed_runs:
        section = run.get("section") or {}
        items = append_session_avg_to_items(list(section.get("items", [])))
        for item in items:
            title = str(item.get("title", ""))
            if title not in title_to_metrics:
                title_order.append(title)
            title_to_metrics[title].append(item.get("metrics", {}))
            cids = item.get("class_ids", [])
            title_to_class_ids[title] = sorted(set(title_to_class_ids.get(title, [])) | {int(c) for c in cids})

    avg_titles = sorted(
        [t for t in title_order if str(t).startswith("session_avg")],
        key=lambda t: _SESSION_SUFFIX_ORDER.get(
            _parse_session_title(str(t))[2] if _parse_session_title(str(t)) else "", 9
        ),
    )
    if avg_titles:
        title_order = [t for t in title_order if not str(t).startswith("session_avg")] + avg_titles

    aggregated_items = []
    for title in title_order:
        metrics_list = title_to_metrics[title]
        class_ids = title_to_class_ids[title]
        if not metrics_list:
            continue

        agg = {"per_class": {}}
        for cid in class_ids:
            agg["per_class"][cid] = {}
            for mkey in ("acc", "f1", "precision", "recall"):
                vals = []
                for metrics in metrics_list:
                    per_class = metrics.get("per_class", {})
                    entry = per_class.get(cid, per_class.get(str(cid), {}))
                    vals.append(entry.get(mkey, "NA"))
                agg["per_class"][cid][mkey] = _format_mean_std(vals)

        for skey in (
            "macro_acc",
            "micro_acc",
            "macro_f1",
            "micro_f1",
            "old_micro_acc",
            "new_micro_acc",
            "total_micro_acc",
            "old_acc",
            "new_acc",
            "total_acc",
            "taskid_acc",
        ):
            vals = [m.get(skey) for m in metrics_list if skey in m]
            if vals:
                agg[skey] = _format_mean_std(vals)

        aggregated_items.append({
            "title": title,
            "metrics": agg,
            "class_ids": class_ids,
        })

    return {"name": section_name, "items": _sort_report_items(aggregated_items)}


def write_test_report(
    root_run_dir: str,
    report_sections: List[dict],
    experiment_seeds: Optional[List[int]] = None,
    *,
    filename: str = "test.txt",
    aggregate_header: bool = False,
):
    def _write_top1_block(f, metrics, class_ids):
        f.write("#top1\n")
        f.write("metric\t" + "\t".join(str(c) for c in class_ids) + "\n")
        f.write("acc\t" + _metric_row(metrics, class_ids, "acc") + "\n")
        f.write("f1\t" + _metric_row(metrics, class_ids, "f1") + "\n")
        f.write("precision\t" + _metric_row(metrics, class_ids, "precision") + "\n")
        f.write("recall\t" + _metric_row(metrics, class_ids, "recall") + "\n")
        if "macro_acc" in metrics:
            f.write(f"macro_acc\t{_format_scalar(metrics['macro_acc'])}\n")
        if "micro_acc" in metrics:
            f.write(f"micro_acc\t{_format_scalar(metrics['micro_acc'])}\n")
        if "macro_f1" in metrics:
            f.write(f"macro_f1\t{_format_scalar(metrics['macro_f1'])}\n")
        if "micro_f1" in metrics:
            f.write(f"micro_f1\t{_format_scalar(metrics['micro_f1'])}\n")
        if "old_micro_acc" in metrics:
            f.write(f"old_micro_acc(sample_weighted_on_old_samples)\t{_format_scalar(metrics['old_micro_acc'])}\n")
        elif "old_acc" in metrics:
            f.write(f"old_acc(micro_on_old_samples)\t{_format_scalar(metrics['old_acc'])}\n")
        if "new_micro_acc" in metrics:
            f.write(f"new_micro_acc(sample_weighted_on_new_samples)\t{_format_scalar(metrics['new_micro_acc'])}\n")
        elif "new_acc" in metrics:
            f.write(f"new_acc(micro_on_new_samples)\t{_format_scalar(metrics['new_acc'])}\n")
        if "total_micro_acc" in metrics:
            f.write(f"total_micro_acc(sample_weighted_on_all_seen_samples)\t{_format_scalar(metrics['total_micro_acc'])}\n")
        elif "total_acc" in metrics:
            f.write(f"total_acc(micro_all_samples)\t{_format_scalar(metrics['total_acc'])}\n")
        if "taskid_acc" in metrics:
            f.write(f"taskid_acc\t{_format_scalar(metrics['taskid_acc'])}\n")

    path = os.path.join(root_run_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        if experiment_seeds:
            f.write(f"# experiment_seeds={experiment_seeds}\n\n")
        for section in report_sections:
            section = finalize_report_section(section)
            header = f"##{section['name']}"
            if aggregate_header and experiment_seeds and len(experiment_seeds) > 1:
                header += f" (n={len(experiment_seeds)}, mean±std)"
            f.write(header + "\n")
            for item in section["items"]:
                title = item["title"]
                metrics = item["metrics"]
                class_ids = item["class_ids"]
                f.write(f"#{title}\n")
                _write_top1_block(f, metrics, class_ids)
                f.write("\n")


EFFICIENCY_METRIC_KEYS = [
    "params_total",
    "params_ever_trained",
    "params_never_trained",
    "flops_forward_per_image",
    "flops_train_only",
    "flops_eval_total",
    "flops_train_total",
    "gpu_peak_mb",
    "time_train_pipeline_sec",
    "time_test_eval_sec",
    "time_train_total_sec",
    "time_test_total_sec",
]


def _format_efficiency_scalar(key: str, value) -> str:
    if value is None:
        return "NA"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if key.startswith("params_"):
        if v >= 1e9:
            return f"{v / 1e9:.3f}G"
        if v >= 1e6:
            return f"{v / 1e6:.2f}M"
        if v >= 1e3:
            return f"{v / 1e3:.2f}K"
        return f"{int(v)}"
    if key.startswith("flops_"):
        if v >= 1e12:
            return f"{v / 1e12:.3f}T"
        if v >= 1e9:
            return f"{v / 1e9:.3f}G"
        if v >= 1e6:
            return f"{v / 1e6:.2f}M"
        return f"{int(v)}"
    if key == "gpu_peak_mb":
        return f"{v:.0f}"
    if key.endswith("_sec"):
        if v >= 3600:
            return f"{v / 3600:.2f}h"
        if v >= 60:
            return f"{v / 60:.2f}min"
        return f"{v:.1f}s"
    return f"{v:.4f}"


def aggregate_efficiency_runs(runs: List[dict]) -> Dict[str, str]:
    """Mean±std over seeds for each run-level efficiency scalar."""
    out: Dict[str, str] = {}
    for key in EFFICIENCY_METRIC_KEYS:
        vals = [r.get(key) for r in runs if r.get(key) is not None]
        nums = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not nums:
            out[key] = "NA"
        else:
            out[key] = _format_mean_std(nums)
    return out


def write_cost_report(
    root_run_dir: str,
    efficiency_by_name: Dict[str, List[dict]],
    section_order: List[str],
    experiment_seeds: Optional[List[int]] = None,
    *,
    filename: str = "cost.txt",
    aggregate_header: bool = False,
):
    path = os.path.join(root_run_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        if experiment_seeds:
            f.write(f"# experiment_seeds={experiment_seeds}\n\n")
        for name in section_order:
            runs = efficiency_by_name.get(name, [])
            if not runs:
                continue
            header = f"##{name}"
            if aggregate_header and experiment_seeds and len(experiment_seeds) > 1:
                header += f" (n={len(experiment_seeds)}, mean±std)"
            f.write(header + "\n")
            agg = aggregate_efficiency_runs(runs)
            f.write("metric\tvalue\n")
            for key in EFFICIENCY_METRIC_KEYS:
                f.write(f"{key}\t{agg.get(key, 'NA')}\n")
            f.write("\n")


def write_seed_test_report(run_dir: str, section: dict, seed: int, *, filename: str = "test.txt") -> None:
    """当前 seed 单次运行的 test.txt（写入 run_dir，不做跨 seed 聚合）。"""
    if not run_dir or section is None:
        return
    os.makedirs(run_dir, exist_ok=True)
    write_test_report(
        run_dir,
        [finalize_report_section(section)],
        experiment_seeds=[int(seed)],
        filename=str(filename),
        aggregate_header=False,
    )


def _format_gate_stats(
    stats: Dict[int, Dict[str, List[float]]],
    *,
    expert_alias_prefix: str = "e",
) -> List[str]:
    lines = []
    for cls_id in sorted(stats):
        item = stats[cls_id]
        if isinstance(item, dict) and "mean" in item and "std" in item:
            mean = item["mean"]
            std = item["std"]
            parts = [f"class_{cls_id}"]
            for idx, (m, s) in enumerate(zip(mean, std)):
                parts.append(f"{expert_alias_prefix}{idx}={m:.6f}±{s:.6f}")
            lines.append("\t".join(parts))
            continue

        for layer_name in sorted(item):
            layer_item = item[layer_name]
            mean = layer_item["mean"]
            std = layer_item["std"]
            parts = [f"class_{cls_id}", str(layer_name)]
            for idx, (m, s) in enumerate(zip(mean, std)):
                parts.append(f"{expert_alias_prefix}{idx}={m:.6f}±{s:.6f}")
            lines.append("\t".join(parts))
    return lines


def _format_task_router_probs(avg_task_probs) -> str:
    if not isinstance(avg_task_probs, list):
        return ""
    parts = []
    for idx, prob in enumerate(avg_task_probs):
        try:
            p = float(prob)
        except (TypeError, ValueError):
            continue
        parts.append(f"t{idx + 1}:{p:.4f}")
    return ",".join(parts)


def _format_task_router_debug(debug: dict) -> List[str]:
    if not isinstance(debug, dict):
        return []
    lines: List[str] = []
    mode = str(debug.get("mode", "unknown"))
    lines.append(f"# mode={mode}")
    if mode != "top2":
        return lines

    alpha = debug.get("alpha", "NA")
    delta = debug.get("delta_threshold", "NA")
    class_prob_mode = debug.get("class_prob_mode", "NA")
    lines.append(f"# alpha={alpha} delta_threshold={delta} class_prob_mode={class_prob_mode}")
    lines.append(
        "# routes total={total} single={single} dual={dual} dual_rate={rate}".format(
            total=debug.get("total_samples", "NA"),
            single=debug.get("single_route_total", "NA"),
            dual=debug.get("dual_route_total", "NA"),
            rate=debug.get("dual_route_rate", "NA"),
        )
    )
    lines.append(
        "# probs avg_top1={p1} avg_top2={p2} avg_margin={margin}".format(
            p1=debug.get("avg_top1_prob", "NA"),
            p2=debug.get("avg_top2_prob", "NA"),
            margin=debug.get("avg_top1_top2_margin", "NA"),
        )
    )

    per_true = debug.get("per_true_task", [])
    if isinstance(per_true, list) and per_true:
        lines.append("true_task\tnum_samples\tsingle\tdouble\tdouble_rate\ttop1_mean_task\tavg_task_probs")
        for item in per_true:
            if not isinstance(item, dict):
                continue
            top1_task = item.get("top1_task_by_mean_prob", "NA")
            if isinstance(top1_task, int) and top1_task >= 0:
                top1_task_text = str(top1_task + 1)
            else:
                top1_task_text = str(top1_task)
            lines.append(
                "{true_task}\t{num_samples}\t{single}\t{double}\t{double_rate}\t{top1_task}\t{avg_probs}".format(
                    true_task=int(item.get("true_task", -1)) + 1,
                    num_samples=item.get("num_samples", "NA"),
                    single=item.get("single_route_count", "NA"),
                    double=item.get("dual_route_count", "NA"),
                    double_rate=item.get("dual_route_rate", "NA"),
                    top1_task=top1_task_text,
                    avg_probs=_format_task_router_probs(item.get("avg_task_probs", [])),
                )
            )
    return lines


def save_router_stats(run_dir: str, profile_name: str, session_stats: List[dict]):
    path = os.path.join(run_dir, "router.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# routing stats (avg prob per expert)\n\n")
        f.write(f"# run {profile_name}\n")
        for session in session_stats:
            f.write(f"## session{session['session'] + 1}\n")
            f.write(f"# experts={session['num_experts']}\n")
            f.write(f"# expert_alias=task{session['session'] + 1}_e0..eN (global-style)\n")
            taskid_acc = session.get("taskid_acc")
            if isinstance(taskid_acc, (int, float)):
                f.write(f"# taskid_acc={float(taskid_acc):.6f}\n")
            dual_rate = session.get("task_router_dual_rate")
            if isinstance(dual_rate, (int, float)):
                f.write(f"# task_router_dual_rate={float(dual_rate):.6f}\n")
            task_router_debug = _format_task_router_debug(session.get("task_router_debug", {}))
            if task_router_debug:
                f.write("### task_router\n")
                for line in task_router_debug:
                    f.write(line + "\n")
                f.write("\n")
            for split in ("train", "test"):
                f.write(f"### {split}\n")
                expert_alias_prefix = f"task{session['session'] + 1}_e"
                for line in _format_gate_stats(session.get(split, {}), expert_alias_prefix=expert_alias_prefix):
                    f.write(line + "\n")
                f.write("\n")
