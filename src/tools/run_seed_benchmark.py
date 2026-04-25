#!/usr/bin/env python3
"""
Run base_model_training.train_quick for multiple seeds and aggregate
quick_summary.json metrics against a baseline.

Expected repository layout:
    <repo>/src/tools/run_seed_benchmark.py
    <repo>/src/base_model_training/train_quick.py

The wrapper automatically sets PYTHONPATH=<repo>/src when invoking
`python -m base_model_training.train_quick`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


@dataclass(frozen=True)
class RunConfig:
    seeds: list[int]
    train_path: Path
    test_path: Path
    pseudolabel_path: Path | None
    train_mode: str
    model_base: str
    batch_size: int
    output_dir: Path
    baseline_dir: Path | None
    log_level: str
    python_executable: str
    module: str
    repo_root: Path
    skip_train: bool
    skip_eval: bool
    overwrite: bool
    dry_run: bool
    extra_train_args: list[str]


def infer_repo_root(script_path: Path) -> Path:
    """Infer repo root, especially for <repo>/src/tools/run_seed_benchmark.py."""
    script_path = script_path.resolve()
    if script_path.parent.name == "tools" and script_path.parent.parent.name == "src":
        return script_path.parent.parent.parent

    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd.parent, cwd.parent.parent, script_path.parent, script_path.parent.parent]
    for candidate in candidates:
        if (candidate / "src" / "base_model_training" / "train_quick.py").exists():
            return candidate
        if (candidate / "base_model_training" / "train_quick.py").exists():
            # Allows running with repo_root=<repo>/src, but we normalize to parent.
            return candidate.parent
    return script_path.parent


def parse_seeds(values: list[str]) -> list[int]:
    seeds: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed must be provided")
    return seeds


def path_arg(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run train_quick for multiple seeds and aggregate quick_summary.json metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--baseline-dir", type=path_arg, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--module", default="base_model_training.train_quick")
    parser.add_argument(
        "--repo-root",
        type=path_arg,
        default=None,
        help="Repository root. If omitted, inferred from script location/current directory.",
    )

    # Same training arguments used by train_quick in the original command.
    parser.add_argument("--train-path", type=path_arg, required=True)
    parser.add_argument("--test-path", type=path_arg, required=True)
    parser.add_argument("--pseudolabel-path", type=path_arg, default=None)
    parser.add_argument("--train-mode", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output-dir", type=path_arg, required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = build_parser()
    args, extra_train_args = parser.parse_known_args(argv)

    if not args.skip_eval and args.baseline_dir is None:
        parser.error("--baseline-dir is required unless --skip-eval is used")

    repo_root = args.repo_root.resolve() if args.repo_root is not None else infer_repo_root(Path(__file__))

    return RunConfig(
        seeds=parse_seeds(args.seeds),
        train_path=args.train_path,
        test_path=args.test_path,
        pseudolabel_path=args.pseudolabel_path,
        train_mode=args.train_mode,
        model_base=args.model_base,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        baseline_dir=args.baseline_dir,
        log_level=args.log_level,
        python_executable=args.python_executable,
        module=args.module,
        repo_root=repo_root,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        extra_train_args=extra_train_args,
    )


def build_env(config: RunConfig) -> dict[str, str]:
    env = os.environ.copy()
    src_dir = config.repo_root / "src"
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) if not old_pythonpath else f"{src_dir}{os.pathsep}{old_pythonpath}"
    return env


def build_train_command(config: RunConfig, seed: int) -> list[str]:
    seed_output_dir = config.output_dir / f"seed_{seed}"
    cmd = [
        config.python_executable,
        "-m",
        config.module,
        "--train-path",
        str(config.train_path),
        "--test-path",
        str(config.test_path),
        "--train-mode",
        config.train_mode,
        "--model-base",
        config.model_base,
        "--batch-size",
        str(config.batch_size),
        "--output-dir",
        str(seed_output_dir),
        "--seed",
        str(seed),
        "--log-level",
        config.log_level,
    ]
    if config.pseudolabel_path is not None:
        cmd.extend(["--pseudolabel-path", str(config.pseudolabel_path)])
    cmd.extend(config.extra_train_args)
    return cmd


def run_training(config: RunConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(config)

    for seed in config.seeds:
        seed_output_dir = config.output_dir / f"seed_{seed}"
        summary_path = seed_output_dir / "quick_summary.json"

        if summary_path.exists() and not config.overwrite:
            print(f"[skip] seed={seed}: found {summary_path}. Use --overwrite to rerun.")
            continue

        seed_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_train_command(config, seed)
        print(f"\n[train] seed={seed}")
        print(" ".join(cmd))
        print(f"[repo-root] {config.repo_root}")
        print(f"[env] PYTHONPATH={config.repo_root / 'src'}")

        if config.dry_run:
            continue

        log_path = seed_output_dir / "train.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                env=env,
            )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Training failed for seed={seed} with exit code {completed.returncode}. See log: {log_path}"
            )
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Training finished for seed={seed}, but expected summary was not found: {summary_path}"
            )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def get_nested(data: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def as_float(value: Any, *, key: str, path: Path) -> float:
    if value is None:
        raise KeyError(f"Missing metric {key!r} in {path}")
    return float(value)


def safe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def metric_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values) if values else math.nan,
        "std": safe_stdev(values) if values else math.nan,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_results(config: RunConfig) -> dict[str, Any]:
    if config.baseline_dir is None:
        raise ValueError("baseline_dir is required for aggregation")

    rows: list[dict[str, Any]] = []
    labels: set[str] = set()

    for seed in config.seeds:
        base_path = config.baseline_dir / f"seed_{seed}" / "quick_summary.json"
        pseudo_path = config.output_dir / f"seed_{seed}" / "quick_summary.json"
        base = load_json(base_path)
        pseudo = load_json(pseudo_path)

        labels |= set((get_nested(base, "test_metrics.per_label", {}) or {}).keys())
        labels |= set((get_nested(pseudo, "test_metrics.per_label", {}) or {}).keys())

        row = {
            "seed": seed,
            "base_micro": as_float(get_nested(base, "test_metrics.micro.f1"), key="test_metrics.micro.f1", path=base_path),
            "pseudo_micro": as_float(get_nested(pseudo, "test_metrics.micro.f1"), key="test_metrics.micro.f1", path=pseudo_path),
            "base_macro": as_float(get_nested(base, "test_metrics.macro_f1"), key="test_metrics.macro_f1", path=base_path),
            "pseudo_macro": as_float(get_nested(pseudo, "test_metrics.macro_f1"), key="test_metrics.macro_f1", path=pseudo_path),
            "_base": base,
            "_pseudo": pseudo,
        }
        row["delta_micro"] = row["pseudo_micro"] - row["base_micro"]
        row["delta_macro"] = row["pseudo_macro"] - row["base_macro"]
        rows.append(row)

    per_seed = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]

    aggregate: dict[str, Any] = {}
    for key in ["base_micro", "pseudo_micro", "delta_micro", "base_macro", "pseudo_macro", "delta_macro"]:
        aggregate[key] = metric_summary([float(row[key]) for row in rows])

    aggregate["wins_micro"] = {"wins": sum(1 for r in rows if float(r["delta_micro"]) > 0), "n": len(rows)}
    aggregate["wins_macro"] = {"wins": sum(1 for r in rows if float(r["delta_macro"]) > 0), "n": len(rows)}

    per_label: list[dict[str, Any]] = []
    for label in sorted(labels):
        base_vals: list[float] = []
        pseudo_vals: list[float] = []
        deltas: list[float] = []
        for row in rows:
            bf = get_nested(row["_base"], f"test_metrics.per_label.{label}.f1")
            pf = get_nested(row["_pseudo"], f"test_metrics.per_label.{label}.f1")
            if bf is None or pf is None:
                continue
            bf = float(bf)
            pf = float(pf)
            base_vals.append(bf)
            pseudo_vals.append(pf)
            deltas.append(pf - bf)
        if deltas:
            per_label.append({
                "label": label,
                "n": len(deltas),
                "base_mean": mean(base_vals),
                "base_std": safe_stdev(base_vals),
                "pseudo_mean": mean(pseudo_vals),
                "pseudo_std": safe_stdev(pseudo_vals),
                "delta_mean": mean(deltas),
                "delta_std": safe_stdev(deltas),
                "wins": sum(1 for d in deltas if d > 0),
            })

    return {
        "seeds": config.seeds,
        "repo_root": str(config.repo_root),
        "baseline_dir": str(config.baseline_dir),
        "pseudo_dir": str(config.output_dir),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "per_label": per_label,
    }


def write_reports(config: RunConfig, report: dict[str, Any]) -> None:
    report_dir = config.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "seed_benchmark_summary.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    write_csv(
        report_dir / "per_seed_metrics.csv",
        report["per_seed"],
        ["seed", "base_micro", "pseudo_micro", "delta_micro", "base_macro", "pseudo_macro", "delta_macro"],
    )

    aggregate_rows = []
    for metric, stats in report["aggregate"].items():
        if metric.startswith("wins_"):
            aggregate_rows.append({"metric": metric, "n": stats["n"], "mean": "", "std": "", "wins": stats["wins"]})
        else:
            aggregate_rows.append({"metric": metric, "n": stats["n"], "mean": stats["mean"], "std": stats["std"], "wins": ""})
    write_csv(report_dir / "aggregate_metrics.csv", aggregate_rows, ["metric", "n", "mean", "std", "wins"])

    write_csv(
        report_dir / "per_label_delta_metrics.csv",
        report["per_label"],
        ["label", "n", "base_mean", "base_std", "pseudo_mean", "pseudo_std", "delta_mean", "delta_std", "wins"],
    )

    print(f"\n[reports] {report_dir}")
    print(f"- {json_path}")
    print(f"- {report_dir / 'per_seed_metrics.csv'}")
    print(f"- {report_dir / 'aggregate_metrics.csv'}")
    print(f"- {report_dir / 'per_label_delta_metrics.csv'}")


def fmt(value: float, signed: bool = False) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def print_console_summary(report: dict[str, Any]) -> None:
    print("\nseed | base_micro | pseudo_micro | delta_micro | base_macro | pseudo_macro | delta_macro")
    for row in report["per_seed"]:
        print(
            f"{row['seed']} | {fmt(row['base_micro'])} | {fmt(row['pseudo_micro'])} | "
            f"{fmt(row['delta_micro'], signed=True)} | {fmt(row['base_macro'])} | "
            f"{fmt(row['pseudo_macro'])} | {fmt(row['delta_macro'], signed=True)}"
        )

    print("\nAggregate")
    for key in ["base_micro", "pseudo_micro", "delta_micro", "base_macro", "pseudo_macro", "delta_macro"]:
        stats = report["aggregate"][key]
        print(f"{key}: mean={fmt(stats['mean'])} std={fmt(stats['std'])}")

    wm = report["aggregate"]["wins_micro"]
    wM = report["aggregate"]["wins_macro"]
    print(f"\nWins micro: {wm['wins']}/{wm['n']}")
    print(f"Wins macro: {wM['wins']}/{wM['n']}")

    print("\nPer-label delta mean")
    for row in report["per_label"]:
        print(
            f"{row['label']}: base_mean={fmt(row['base_mean'])} "
            f"pseudo_mean={fmt(row['pseudo_mean'])} "
            f"delta_mean={fmt(row['delta_mean'], signed=True)} "
            f"wins={row['wins']}/{row['n']}"
        )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)

    expected_module_file = config.repo_root / "src" / "base_model_training" / "train_quick.py"
    if not expected_module_file.exists():
        print(
            f"[warning] Expected training module not found at {expected_module_file}. "
            "Use --repo-root to point to the repository root if inference was wrong.",
            file=sys.stderr,
        )

    if not config.skip_train:
        run_training(config)

    if not config.skip_eval:
        report = aggregate_results(config)
        print_console_summary(report)
        write_reports(config, report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
