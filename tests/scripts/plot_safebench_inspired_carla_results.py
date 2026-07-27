#!/usr/bin/env python3
"""Generate scenario-family tables and bar charts for SafeBench-inspired CARLA eval."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "results"
    / "safebench_inspired_carla_4scenes_10routes_15egos"
)

METRICS: List[Tuple[str, str, str]] = [
    ("crash", "Collision rate", "crash_rate"),
    ("route_completion", "Road completion", "route_completion"),
    ("ego_route_distance_m", "Driven distance", "ego_route_distance_m"),
    ("mean_speed", "Average speed", "mean_speed"),
    ("combined_efficiency", "Traffic efficiency", "combined_efficiency"),
    ("episode_reward", "Average reward", "episode_reward"),
]

SCENARIO_ORDER = ["follow_straight", "passing", "lane_change", "cut_in"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--episode-log", default="", help="Defaults to <output-dir>/combined_episode_log.csv.")
    return parser.parse_args()


def format_mean_var(mean_value, var_value) -> str:
    if pd.isna(mean_value):
        return ""
    return f"{float(mean_value):.4f} ± {float(var_value):.4f}"


def ensure_scenario_family(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "scenario_family" not in df.columns:
        if "template" in df.columns:
            df["scenario_family"] = df["template"]
        elif "scenario_name" in df.columns:
            df["scenario_family"] = df["scenario_name"]
        else:
            df["scenario_family"] = "scenario"
    df["scenario_family"] = df["scenario_family"].fillna("scenario").astype(str)
    return df


def model_sort_key(label: str, checkpoint_step) -> tuple:
    try:
        step = int(float(checkpoint_step))
    except (TypeError, ValueError):
        step = -1
    return step, str(label)


def scenario_family_mean_variance(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_scenario_family(df)
    rows = []
    group_cols = ["model_group", "model_label", "checkpoint_step", "scenario_family"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["episodes"] = int(len(sub))
        for metric, title, _slug in METRICS:
            values = pd.to_numeric(sub.get(metric), errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_var"] = float(values.var(ddof=0)) if len(values) else np.nan
            row[f"{title}_mean_var"] = format_mean_var(row[f"{metric}_mean"], row[f"{metric}_var"])
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["_scenario_order"] = table["scenario_family"].map(
        {name: idx for idx, name in enumerate(SCENARIO_ORDER)}
    ).fillna(len(SCENARIO_ORDER))
    table = table.sort_values(["checkpoint_step", "_scenario_order", "scenario_family"]).drop(columns=["_scenario_order"])
    return table


def write_formatted_table(table: pd.DataFrame, output_path: Path) -> None:
    cols = ["model_group", "model_label", "checkpoint_step", "scenario_family", "episodes"]
    cols.extend([f"{title}_mean_var" for _metric, title, _slug in METRICS])
    table[[col for col in cols if col in table.columns]].to_csv(output_path, index=False)


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


def model_order(table: pd.DataFrame) -> List[str]:
    ordered = table[["model_label", "checkpoint_step"]].drop_duplicates().copy()
    ordered["_key"] = [model_sort_key(row.model_label, row.checkpoint_step) for row in ordered.itertuples()]
    return ordered.sort_values("_key")["model_label"].tolist()


def scenario_order(table: pd.DataFrame) -> List[str]:
    seen = [item for item in SCENARIO_ORDER if item in set(table["scenario_family"])]
    extras = sorted(set(table["scenario_family"]) - set(seen))
    return seen + extras


def save_scenario_grouped_bar(plt, table: pd.DataFrame, metric: str, title: str, output_path: Path) -> bool:
    mean_col = f"{metric}_mean"
    if mean_col not in table.columns or table.empty:
        return False
    order = scenario_order(table)
    labels = model_order(table)
    pivot = table.pivot_table(index="scenario_family", columns="model_label", values=mean_col, aggfunc="mean")
    pivot = pivot.reindex(order)
    pivot = pivot[[label for label in labels if label in pivot.columns]]
    if pivot.empty:
        return False

    fig, ax = plt.subplots(figsize=(max(13.5, 0.9 * len(pivot.columns)), 6.2))
    x = np.arange(len(pivot.index))
    width = min(0.82 / max(len(pivot.columns), 1), 0.075)
    for idx, column in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2.0) * width
        color = "#f58518" if "train" in str(column) else None
        ax.bar(x + offset, pivot[column], width=width, label=column, color=color)
    ax.set_title(f"{title} by scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(name).replace("_", " ") for name in pivot.index], rotation=15, ha="right")
    ax.legend(loc="best", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def save_by_scenario_facets(plt, table: pd.DataFrame, metric: str, title: str, output_path: Path) -> bool:
    mean_col = f"{metric}_mean"
    if mean_col not in table.columns or table.empty:
        return False
    labels = model_order(table)
    scenarios = scenario_order(table)
    ncols = 2
    nrows = int(np.ceil(max(len(scenarios), 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, max(4.8 * nrows, 5.0)), squeeze=False)
    for ax, scenario in zip(axes.flatten(), scenarios):
        sub = table[table["scenario_family"] == scenario].copy()
        sub["model_label"] = pd.Categorical(sub["model_label"], categories=labels, ordered=True)
        sub = sub.sort_values("model_label")
        x = np.arange(len(sub))
        colors = ["#f58518" if group == "training_final" else "#4c78a8" for group in sub["model_group"]]
        ax.bar(x, sub[mean_col], color=colors)
        ax.set_title(str(scenario).replace("_", " "))
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model_label"], rotation=45, ha="right", fontsize=7)
    for ax in axes.flatten()[len(scenarios):]:
        ax.axis("off")
    fig.suptitle(f"{title} under each scenario", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def generate(output_dir: Path, episode_log: Path) -> List[Path]:
    df = pd.read_csv(episode_log, low_memory=False)
    df = ensure_scenario_family(df)
    for metric, _title, _slug in METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

    table = scenario_family_mean_variance(df)
    numeric_path = output_dir / "scenario_family_mean_variance.csv"
    formatted_path = output_dir / "scenario_family_comparison_mean_plus_variance.csv"
    table.to_csv(numeric_path, index=False)
    write_formatted_table(table, formatted_path)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    figures = []
    for metric, title, slug in METRICS:
        grouped_path = figures_dir / f"scenario_mean_{slug}_bar.png"
        if save_scenario_grouped_bar(plt, table, metric, title, grouped_path):
            figures.append(grouped_path)
        facet_path = figures_dir / f"by_scenario_{slug}_bar.png"
        if save_by_scenario_facets(plt, table, metric, title, facet_path):
            figures.append(facet_path)
    return [numeric_path, formatted_path, *figures]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    episode_log = Path(args.episode_log) if args.episode_log else output_dir / "combined_episode_log.csv"
    if not episode_log.exists():
        raise FileNotFoundError(str(episode_log))
    outputs = generate(output_dir, episode_log)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
