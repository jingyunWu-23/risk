#!/usr/bin/env python3
"""Batch-evaluate EgoPPO checkpoints from one fine-tune run and draw bar charts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EGO_DIR = MODEL_ROOT / "results" / "ego_enhanced_poet_finetune" / "poet_finetune_Jul_21_13_29_28-30-10" / "models" / "ego"
DEFAULT_ADV_DIR = MODEL_ROOT / "results" / "joint_Jul_01_14_59_59" / "models" / "adv"
DEFAULT_ROUND_LOG = MODEL_ROOT / "results" / "joint_Jul_01_14_59_59" / "joint_round_log.csv"
DEFAULT_CONFIG = MODEL_ROOT / "configs" / "carla_0915.yaml"

METRICS = [
    ("route_completion", "Road completion"),
    ("ego_route_distance_m", "Driven distance"),
    ("episode_reward", "Average reward"),
    ("crash", "Ego crash rate"),
    ("combined_efficiency", "Traffic efficiency"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ego-dir", default=str(DEFAULT_EGO_DIR))
    parser.add_argument(
        "--ego-checkpoint",
        action="append",
        default=None,
        help="Additional EgoPPO checkpoint path to evaluate. Repeatable.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--include-final", action="store_true", default=False)
    parser.add_argument(
        "--label-with-source",
        action="store_true",
        default=False,
        help="Prefix model labels with the checkpoint's experiment directory to avoid merging identical step numbers.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--adv-model-dir", default=str(DEFAULT_ADV_DIR))
    parser.add_argument("--joint-round-log", default=str(DEFAULT_ROUND_LOG))
    parser.add_argument("--adv-round-last", type=int, default=60)
    parser.add_argument("--adv-sample-count", type=int, default=10)
    parser.add_argument("--eval-seeds", default="101,102,103")
    parser.add_argument("--eval-episodes-per-adv", type=int, default=0)
    parser.add_argument("--eval-seed-base", type=int, default=101)
    parser.add_argument("--eval-seed-stride", type=int, default=100000)
    parser.add_argument("--num-adv", type=int, default=3)
    parser.add_argument("--num-natural", type=int, default=0)
    parser.add_argument("--spawn-start-index", type=int, default=0)
    parser.add_argument("--randomize-scenarios", action="store_true", default=False)
    parser.add_argument("--relative-offset-jitter", type=float, default=0.0)
    parser.add_argument("--randomize-lane-offsets", action="store_true", default=False)
    parser.add_argument("--randomize-lane-offset-min", type=int, default=0)
    parser.add_argument("--randomize-lane-offset-max", type=int, default=2)
    parser.add_argument("--randomize-spawn-start-index", action="store_true", default=False)
    parser.add_argument("--shuffle-spawn-points", action="store_true", default=False)
    parser.add_argument("--hdv-model", default="")
    parser.add_argument("--hdv-action", default="keep_lane")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--carla-rpc-timeout", type=float, default=300.0)
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES value for child eval commands.")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument(
        "--rarl-config",
        default=str(MODEL_ROOT.parent / "configs" / "compare_3sv.yaml"),
        help="Config forwarded when evaluating RARL TD3 checkpoints named checkpoint_N.pt or latest.pt.",
    )
    parser.add_argument("--purge-existing-actors-on-reset", action="store_true", default=False)
    parser.add_argument("--cleanup-destroy-mode", choices=["sequential", "batch"], default="sequential")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--rerun-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint[-_](\d+)\.pt", path.name)
    if not match and path.name == "latest.pt":
        sibling_steps = [
            checkpoint_step(item)
            for item in path.parent.glob("checkpoint*.pt")
            if item.name != "latest.pt" and re.fullmatch(r"checkpoint[-_]\d+\.pt", item.name)
        ]
        return max(sibling_steps, default=0)
    if not match:
        raise ValueError(path.name)
    return int(match.group(1))


def checkpoint_source(path: Path) -> str:
    parts = Path(path).parts
    if "models" in parts:
        model_index = parts.index("models")
        if model_index > 0:
            return parts[model_index - 1]
    if len(Path(path).parents) >= 3:
        return Path(path).parents[2].name
    return Path(path).parent.name


def checkpoint_label(path: Path, label_with_source: bool = False) -> str:
    step = checkpoint_step(path)
    if not label_with_source:
        return f"Ego-{step}"
    source = checkpoint_source(path)
    source = source.replace("poet_finetune_", "").replace("joint_", "joint-")
    return f"{source}-Ego-{step}"


def select_checkpoints(ego_dir: Path, interval: int, include_final: bool):
    checkpoints = sorted(
        [
            path
            for pattern in ("checkpoint-*.pt", "checkpoint_*.pt")
            for path in ego_dir.glob(pattern)
            if re.fullmatch(r"checkpoint[-_]\d+\.pt", path.name)
        ],
        key=checkpoint_step,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-*.pt or checkpoint_*.pt found under {ego_dir}")
    selected = []
    last_step = None
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        if last_step is None or step - last_step >= int(interval):
            selected.append(checkpoint)
            last_step = step
    if include_final and checkpoints[-1] not in selected:
        selected.append(checkpoints[-1])
    return selected


def select_all_checkpoints(args):
    selected = []
    try:
        selected = select_checkpoints(Path(args.ego_dir), int(args.checkpoint_interval), bool(args.include_final))
    except FileNotFoundError:
        if not args.ego_checkpoint:
            raise
    seen = {str(path.resolve()) for path in selected}
    for item in args.ego_checkpoint or []:
        checkpoint = Path(item)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Ego checkpoint not found: {checkpoint}")
        resolved = str(checkpoint.resolve())
        if resolved not in seen:
            selected.append(checkpoint)
            seen.add(resolved)
    return sorted(selected, key=checkpoint_step)


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%b_%d_%H_%M_%S")
        output_dir = MODEL_ROOT / "results" / "ego_adv_frozen_eval" / f"ego_sweep_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_command(cmd, cuda_visible_devices=None, log_path: Path | None = None):
    env = None
    if cuda_visible_devices is not None:
        import os

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    print(" ".join(str(part) for part in cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, check=True, env=env)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[LOG] {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            subprocess.run(cmd, check=True, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(f"[FAILED] child process log: {log_path}", flush=True)
            raise


def eval_one(args, checkpoint: Path, run_dir: Path):
    if args.skip_existing and (run_dir / "eval_complete.json").exists():
        print(f"[SKIP] existing eval: {run_dir}")
        return
    cmd = [
        sys.executable,
        str(MODEL_ROOT / "scripts" / "evaluate_finetuned_ego_against_adv.py"),
        "--config",
        str(args.config),
        "--ego-checkpoint",
        str(checkpoint),
        "--adv-model-dir",
        str(args.adv_model_dir),
        "--joint-round-log",
        str(args.joint_round_log),
        "--adv-round-last",
        str(args.adv_round_last),
        "--adv-sample-count",
        str(args.adv_sample_count),
        "--eval-seeds",
        str(args.eval_seeds),
        "--eval-episodes-per-adv",
        str(args.eval_episodes_per_adv),
        "--eval-seed-base",
        str(args.eval_seed_base),
        "--eval-seed-stride",
        str(args.eval_seed_stride),
        "--num-adv",
        str(args.num_adv),
        "--num-natural",
        str(args.num_natural),
        "--spawn-start-index",
        str(args.spawn_start_index),
        "--relative-offset-jitter",
        str(args.relative_offset_jitter),
        "--randomize-lane-offset-min",
        str(args.randomize_lane_offset_min),
        "--randomize-lane-offset-max",
        str(args.randomize_lane_offset_max),
        "--hdv-model",
        str(args.hdv_model),
        "--hdv-action",
        str(args.hdv_action),
        "--max-steps",
        str(args.max_steps),
        "--carla-rpc-timeout",
        str(args.carla_rpc_timeout),
        "--cleanup-destroy-mode",
        str(args.cleanup_destroy_mode),
        "--rarl-config",
        str(args.rarl_config),
        "--output-dir",
        str(run_dir),
    ]
    if args.no_cuda:
        cmd.append("--no-cuda")
    if args.purge_existing_actors_on_reset:
        cmd.append("--purge-existing-actors-on-reset")
    if args.randomize_scenarios:
        cmd.append("--randomize-scenarios")
    if args.randomize_lane_offsets:
        cmd.append("--randomize-lane-offsets")
    if args.randomize_spawn_start_index:
        cmd.append("--randomize-spawn-start-index")
    if args.shuffle_spawn_points:
        cmd.append("--shuffle-spawn-points")
    if args.dry_run:
        print(" ".join(str(part) for part in cmd))
        return
    run_command(cmd, cuda_visible_devices=args.cuda_visible_devices, log_path=run_dir / "eval.log")


def load_eval_run(label: str, run_dir: Path):
    summary_path = run_dir / "adv_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(str(summary_path))
    df = pd.read_csv(summary_path)
    df["model_label"] = label
    df["run_dir"] = str(run_dir)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        df["ego_checkpoint"] = manifest.get("ego_checkpoint", "")
    else:
        df["ego_checkpoint"] = ""
    for col in ["round", "adv_step", *[metric for metric, _ in METRICS]]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_episode_run(label: str, checkpoint: Path, run_dir: Path):
    path = run_dir / "episode_log.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    df["model_label"] = label
    df["model_source"] = checkpoint_source(checkpoint)
    df["checkpoint_step"] = checkpoint_step(checkpoint)
    df["ego_checkpoint"] = str(checkpoint)
    for metric, _title in METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    for col in ["round", "adv_step", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def format_mean_var(mean_value, var_value) -> str:
    if pd.isna(mean_value):
        return ""
    return f"{float(mean_value):.4f} ± {float(var_value):.4f}"


def scenario_mean_variance(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_label", "checkpoint_step", "round", "adv_step"]
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["episodes"] = int(len(sub))
        row["ego_checkpoint"] = str(sub["ego_checkpoint"].iloc[0]) if "ego_checkpoint" in sub.columns else ""
        for metric, title in METRICS:
            if metric not in sub.columns:
                continue
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_var"] = float(values.var(ddof=0)) if len(values) else np.nan
            row[f"{title}_mean_var"] = format_mean_var(row[f"{metric}_mean"], row[f"{metric}_var"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["checkpoint_step", "adv_step"])


def model_mean_variance(scenario_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model_label", "checkpoint_step"]
    for keys, sub in scenario_table.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["scenario_count"] = int(len(sub))
        row["episode_count"] = int(sub["episodes"].sum()) if "episodes" in sub.columns else 0
        row["ego_checkpoint"] = str(sub["ego_checkpoint"].iloc[0]) if "ego_checkpoint" in sub.columns else ""
        for metric, title in METRICS:
            mean_col = f"{metric}_mean"
            var_col = f"{metric}_var"
            if mean_col not in sub.columns:
                continue
            metric_mean = float(pd.to_numeric(sub[mean_col], errors="coerce").mean())
            metric_var = float(pd.to_numeric(sub[var_col], errors="coerce").mean())
            row[f"{metric}_mean"] = metric_mean
            row[f"{metric}_var"] = metric_var
            row[f"{title}_mean_var"] = format_mean_var(metric_mean, metric_var)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("checkpoint_step")


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def save_overall_bar(plt, overall: pd.DataFrame, metric: str, title: str, path: Path):
    mean_col = f"{metric}_mean"
    if mean_col not in overall.columns:
        return False
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(overall)), 4.8))
    x = np.arange(len(overall))
    ax.bar(x, overall[mean_col], color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel("Ego checkpoint")
    ax.set_ylabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels(overall["model_label"], rotation=45, ha="right")
    for idx, value in enumerate(overall[mean_col]):
        if pd.notna(value):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def save_scenario_grouped_bar(plt, combined: pd.DataFrame, metric: str, title: str, path: Path):
    mean_col = f"{metric}_mean"
    if mean_col not in combined.columns or "adv_step" not in combined.columns:
        return False
    pivot = combined.pivot_table(index="adv_step", columns="model_label", values=mean_col, aggfunc="mean").sort_index()
    if pivot.empty:
        return False
    model_order = (
        combined[["model_label", "checkpoint_step"]]
        .drop_duplicates()
        .sort_values("checkpoint_step")["model_label"]
        .tolist()
    )
    pivot = pivot[[label for label in model_order if label in pivot.columns]]
    fig, ax = plt.subplots(figsize=(max(11, 0.7 * len(pivot.columns)), 5.5))
    x = np.arange(len(pivot.index))
    width = min(0.8 / max(len(pivot.columns), 1), 0.08)
    for idx, column in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2.0) * width
        ax.bar(x + offset, pivot[column], width=width, label=column)
    ax.set_title(title)
    ax.set_xlabel("Adversarial checkpoint step")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(item)) for item in pivot.index], rotation=30, ha="right")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def save_adv_grouped_bars(plt, scenario_table: pd.DataFrame, metric: str, title: str, output_path: Path):
    mean_col = f"{metric}_mean"
    if mean_col not in scenario_table.columns or "adv_step" not in scenario_table.columns:
        return False
    adv_steps = sorted(pd.to_numeric(scenario_table["adv_step"], errors="coerce").dropna().astype(int).unique())
    if not adv_steps:
        return False
    model_order = (
        scenario_table[["model_label", "checkpoint_step"]]
        .drop_duplicates()
        .sort_values("checkpoint_step")["model_label"]
        .tolist()
    )
    ncols = 2
    nrows = int(np.ceil(len(adv_steps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(4.0 * nrows, 5.5)), squeeze=False)
    for ax, adv_step in zip(axes.flatten(), adv_steps):
        sub = scenario_table[scenario_table["adv_step"].astype(int) == int(adv_step)].copy()
        sub["model_label"] = pd.Categorical(sub["model_label"], categories=model_order, ordered=True)
        sub = sub.sort_values("model_label")
        x = np.arange(len(sub))
        ax.bar(x, sub[mean_col], color="#4c78a8")
        ax.set_title(f"Adv-{adv_step}")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model_label"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(title)
        for idx, value in enumerate(sub[mean_col]):
            if pd.notna(value):
                ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=6)
    for ax in axes.flatten()[len(adv_steps):]:
        ax.axis("off")
    fig.suptitle(f"{title} under each adversarial checkpoint", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def aggregate_and_plot(output_dir: Path, selected, label_with_source: bool = False):
    rows = []
    for checkpoint in selected:
        label = checkpoint_label(checkpoint, label_with_source)
        rows.append(load_episode_run(label, checkpoint, output_dir / "eval_runs" / label))
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(output_dir / "combined_episode_log.csv", index=False)
    scenario_table = scenario_mean_variance(combined)
    overall = model_mean_variance(scenario_table)
    scenario_table.to_csv(output_dir / "scenario_mean_variance.csv", index=False)
    overall.to_csv(output_dir / "model_mean_variance_numeric.csv", index=False)

    formatted_cols = ["model_label", "checkpoint_step", "scenario_count", "episode_count", "ego_checkpoint"]
    formatted_cols.extend([
        f"{title}_mean_var"
        for _metric, title in METRICS
        if f"{title}_mean_var" in overall.columns
    ])
    overall[formatted_cols].to_csv(output_dir / "model_comparison_mean_plus_variance.csv", index=False)

    pivot_metrics = [f"{metric}_mean" for metric, _title in METRICS if f"{metric}_mean" in scenario_table.columns]
    scenario_table[["model_label", "checkpoint_step", "round", "adv_step", "ego_checkpoint", *pivot_metrics]].to_csv(
        output_dir / "scenario_checkpoint_comparison.csv",
        index=False,
    )
    scenario_table.pivot_table(
        index=["round", "adv_step"],
        columns="model_label",
        values=pivot_metrics,
        aggfunc="mean",
    ).to_csv(output_dir / "scenario_metric_pivot.csv")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    generated = []
    for metric, title in METRICS:
        if f"{metric}_mean" not in overall.columns:
            continue
        path = figures_dir / f"overall_{metric}_bar.png"
        if save_overall_bar(plt, overall, metric, title, path):
            generated.append(path)
        scenario_path = figures_dir / f"scenario_mean_{metric}_bar.png"
        if save_scenario_grouped_bar(plt, scenario_table, metric, title, scenario_path):
            generated.append(scenario_path)
        adv_path = figures_dir / f"by_adv_{metric}_bar.png"
        if save_adv_grouped_bars(plt, scenario_table, metric, title, adv_path):
            generated.append(adv_path)
    return overall, generated


def main():
    args = parse_args()
    output_dir = make_output_dir(args)
    selected = select_all_checkpoints(args)

    manifest = {
        "ego_dir": str(args.ego_dir),
        "selected_checkpoints": [str(path) for path in selected],
        "selected_steps": [checkpoint_step(path) for path in selected],
        "selected_labels": [checkpoint_label(path, bool(args.label_with_source)) for path in selected],
        "args": vars(args),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Output dir: {output_dir}")
    print(f"Selected {len(selected)} checkpoints:")
    print(", ".join(str(checkpoint_step(path)) for path in selected))

    eval_root = output_dir / "eval_runs"
    eval_root.mkdir(parents=True, exist_ok=True)
    for checkpoint in selected:
        label = checkpoint_label(checkpoint, bool(args.label_with_source))
        eval_one(args, checkpoint, eval_root / label)

    if args.dry_run:
        return

    overall, figures = aggregate_and_plot(output_dir, selected, bool(args.label_with_source))
    print(f"Formatted table: {output_dir / 'model_comparison_mean_plus_variance.csv'}")
    print(f"Numeric table: {output_dir / 'model_mean_variance_numeric.csv'}")
    print(f"Scenario table: {output_dir / 'scenario_checkpoint_comparison.csv'}")
    for figure in figures:
        print(f"Figure: {figure}")
    display_cols = [
        "model_label",
        "combined_efficiency_mean",
        "ego_route_distance_m_mean",
        "episode_reward_mean",
        "crash_mean",
        "route_completion_mean",
    ]
    display_cols = [col for col in display_cols if col in overall.columns]
    print(overall[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
