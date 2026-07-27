#!/usr/bin/env python3
"""Generate selected SafeBench/ScenGE Scenic scenarios with the LLM msgen path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


MODEL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENGE_ROOT = Path("/home/chenyuanwan/download/ScenGE-main")

BEHAVIOR_TO_SCENGE_SCENARIO: Dict[str, int] = {
    # ScenGE scenario ids are coarse SafeBench semantic classes. follow and cut_in
    # both use scenario 1 and are separated by repeated generation/LLM semantics.
    "straight_follow": 1,
    "follow": 1,
    "cut_in": 1,
    "lane_change": 3,
    "passing": 4,
}

DEFAULT_BEHAVIORS = ["straight_follow", "cut_in", "lane_change", "passing"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenge-root", default=str(DEFAULT_SCENGE_ROOT))
    parser.add_argument("--behaviors", nargs="+", default=DEFAULT_BEHAVIORS, choices=sorted(BEHAVIOR_TO_SCENGE_SCENARIO))
    parser.add_argument("--scenario-route-id", type=int, default=1)
    parser.add_argument("--route-pickle", default="")
    parser.add_argument("--llm-backend", choices=["openai", "huggingface", "hf"], default="openai")
    parser.add_argument("--llm-name", default="qwen3.7-plus")
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--openai-temperature", type=float, default=0.1)
    parser.add_argument("--openai-max-tokens", type=int, default=0)
    parser.add_argument("--embed-backend", choices=["openai", "huggingface", "hf"], default="openai")
    parser.add_argument("--embed-name", default="text-embedding-v4")
    parser.add_argument("--max-repair-rounds", type=int, default=4)
    parser.add_argument("--no-scenic-sample", action="store_true", default=False)
    parser.add_argument("--no-template-fallback", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def scenario_ids_for_behaviors(behaviors: List[str]) -> List[int]:
    return [BEHAVIOR_TO_SCENGE_SCENARIO[item] for item in behaviors]


def build_cmd(args, scenario_ids: List[int]) -> List[str]:
    scenge_root = Path(args.scenge_root).expanduser().resolve()
    script = scenge_root / "safebench" / "scenge" / "msgen" / "meta_scenario_generation.py"
    if not script.exists():
        raise FileNotFoundError(f"ScenGE msgen script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--llm-backend",
        args.llm_backend,
        "--llm_name",
        args.llm_name,
        "--embed-backend",
        args.embed_backend,
        "--embed_name",
        args.embed_name,
        "--scenario_ids",
        *[str(item) for item in scenario_ids],
        "--scenario-route-id",
        str(args.scenario_route_id),
        "--max-repair-rounds",
        str(args.max_repair_rounds),
    ]
    if args.openai_api_key:
        cmd.extend(["--openai-api-key", args.openai_api_key])
    if args.openai_base_url:
        cmd.extend(["--openai-base-url", args.openai_base_url])
    if args.openai_temperature is not None:
        cmd.extend(["--openai-temperature", str(args.openai_temperature)])
    if int(args.openai_max_tokens) > 0:
        cmd.extend(["--openai-max-tokens", str(args.openai_max_tokens)])
    if args.route_pickle:
        cmd.extend(["--route-pickle", args.route_pickle])
    if args.no_scenic_sample:
        cmd.append("--no-scenic-sample")
    if args.no_template_fallback:
        cmd.append("--no-template-fallback")
    return cmd


def write_manifest(args, scenario_ids: List[int], command: List[str]) -> Path:
    out_dir = MODEL_ROOT / "tests" / "results" / "scenge_llm_generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"manifest_{datetime.now().strftime('%b_%d_%H_%M_%S')}.json"
    payload = {
        "generator": "ScenGE msgen LLM/RAG",
        "scenge_root": str(Path(args.scenge_root).expanduser().resolve()),
        "behaviors": list(args.behaviors),
        "scenario_ids": scenario_ids,
        "scenario_route_id": int(args.scenario_route_id),
        "llm_backend": args.llm_backend,
        "llm_name": args.llm_name,
        "embed_backend": args.embed_backend,
        "embed_name": args.embed_name,
        "output_root": str(Path(args.scenge_root).expanduser().resolve() / "safebench" / "scenario" / "scenario_data" / "scenic_data"),
        "command": command,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return manifest_path


def main():
    args = parse_args()
    scenario_ids = scenario_ids_for_behaviors(args.behaviors)
    command = build_cmd(args, scenario_ids)
    manifest_path = write_manifest(args, scenario_ids, command)

    print("ScenGE LLM scenario generation")
    print(f"  behaviors:    {', '.join(args.behaviors)}")
    print(f"  scenario_ids: {' '.join(str(item) for item in scenario_ids)}")
    print(f"  scenge_root:  {Path(args.scenge_root).expanduser().resolve()}")
    print(f"  manifest:     {manifest_path}")
    print("  command:")
    print("  " + " ".join(command))
    if args.dry_run:
        return

    env = os.environ.copy()
    if args.openai_api_key:
        env["OPENAI_API_KEY"] = args.openai_api_key
    if args.openai_base_url:
        env["OPENAI_BASE_URL"] = args.openai_base_url
    result = subprocess.run(
        command,
        cwd=str(Path(args.scenge_root).expanduser().resolve()),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
