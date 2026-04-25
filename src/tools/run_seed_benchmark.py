#!/usr/bin/env python3
"""
Utility script to run seed-based GLiNER training/evaluation benchmarks and
aggregate results across seeds.

This script encapsulates two workflows:

1. Run base_model_training.train_quick once per seed, writing each run to
   <output-dir>/seed_<SEED>.
2. Compare the resulting quick_summary.json files against a baseline directory
   and produce per-seed and aggregate reports.

Example:
    python run_seed_benchmark.py \
      --seeds 13 42 99 17 23 \
      --train-path ../data/dd_corpus_small_train.json \
      --test-path ../data/dd_corpus_small_test.json \
      --pseudolabel-path ../artifacts/benchmarks/metadata_location_literal_top20_v1/refit_pseudolabels_top50.jsonl \
      --train-mode supervised_plus_pseudolabels \
      --model-base ../artifacts/base_model_training/quick_supervised_only_locprefix_expanded_clean_v2/best_quick_gliner_model \
      --batch-size 16 \
      --output-dir ../artifacts/benchmarks/metadata_location_literal_top50_v1_bs16_rerun_bestckpt \
      --baseline-dir ../artifacts/benchmarks/quick_supervised_only_locprefix_expanded_clean_v2_bs16_rerun_bestckpt \
      --log-level INFO

If you only want to aggregate existing runs:
    python run_seed_benchmark.py ... --skip-train
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    skip_train: bool
    skip_eval: bool
    overwrite: bool
    dry_run: bool
    extra_train_args: list[str]


def parse_seeds(values: list[str]) -> list[int]:
    """Parse seeds passed either as space-separated or comma-separated values."""
    seeds: list[int] = []
    for value in values:
        for part in value.split(','):
            part = part.strip()
            if part:
                seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError('at least one seed must be provided')
    return seeds


def path_arg(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Run base_model_training.train_quick for multiple seeds and '
            'aggregate quick_summary.json metrics.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Seed benchmark controls.
    parser.add_argument(
        '--seeds',
        nargs='+',
        required=True,
        help='Seeds to run, either space-separated or comma-separated. Example: --seeds 13 42 99 or --seeds 13,42,99',
    )
    parser.add_argument(
        '--baseline-dir',
        type=path_arg,
        default=None,
        help='Directory containing baseline seed_<SEED>/quick_summary.json files. Required unless --skip-eval is used.',
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Do not run training; only aggregate existing quick_summary.json files.',
    )
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Do not aggregate metrics after training.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Rerun a seed even if <output-dir>/seed_<SEED>/quick_summary.json already exists.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing them.',
    )
    parser.add_argument(
        '--python-executable',
        default=sys.executable,
        help='Python executable used to call the training module.',
    )
    parser.add_argument(
        '--module',
        default='base_model_training.train_quick',
        help='Training module invoked with python -m.',
    )

    # Same training arguments used in the original snippet.
    parser.add_argument('--train-path', type=path_arg, required=True)
    parser.add_argument('--test-path', type=path_arg, required=True)
    parser.add_argument('--pseudolabel-path', type=path_arg, default=None)
    parser.add_argument('--train-mode', required=True)
    parser.add_argument('--model-base', required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument(
        '--output-dir',
        type=path_arg,
        required=True,
        help='Benchmark root directory. Per-seed outputs are written to <output-dir>/seed_<SEED>.',
    )
    parser.add_argument('--log-level', default='INFO')

    return parser


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = build_parser()
    args, extra_train_args = parser.parse_known_args(argv)

    if args.skip_eval is False and args.baseline_dir is None:
        parser.error('--baseline-dir is required unless --skip-eval is used')

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
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        extra_train_args=extra_train_args,
    )


def build_train_command(config: RunConfig, seed: int) -> list[str]:
    seed_output_dir = config.output_dir / f'seed_{seed}'
    cmd = [
        config.python_executable,
        '-m',
        config.module,
        '--train-path',
        str(config.train_path),
        '--test-path',
        str(config.test_path),
        '--train-mode',
        config.train_mode,
        '--model-base',
        config.model_base,
        '--batch-size',
        str(config.batch_size),
        '--output-dir',
        str(seed_output_dir),
        '--seed',
        str(seed),
        '--log-level',
        config.log_level,
    ]
    if config.pseudolabel_path is not None:
        cmd.extend(['--pseudolabel-path', str(config.pseudolabel_path)])

    # Forward any additional arguments to train_quick. This makes the wrapper
    # resilient to future train_quick options while keeping the known arguments explicit.
    cmd.extend(config.extra_train_args)
    return cmd


def run_training(config: RunConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in config.seeds:
        seed_output_dir = config.output_dir / f'seed_{seed}'
        summary_path = seed_output_dir / 'quick_summary.json'

        if summary_path.exists() and not config.overwrite:
            print(f'[skip] seed={seed}: found {summary_path}. Use --overwrite to rerun.')
            continue

        seed_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_train_command(config, seed)
        print('\n[train] seed=', seed, sep='')
        print(' '.join(cmd))

        if config.dry_run:
            continue

        log_path = seed_output_dir / 'train.log'
        with log_path.open('w', encoding='utf-8') as log_file:
            completed = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        if completed.returncode != 0:
            raise RuntimeError(
                f'Training failed for seed={seed} with exit code {completed.returncode}. '
                f'See log: {log_path}'
            )

        if not summary_path.exists():
            raise FileNotFoundError(
                f'Training finished for seed={seed}, but expected summary was not found: {summary_path}'
            )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def get_nested(data: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted_key.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def as_float(value: Any, *, key: str, path: Path) -> float:
    if value is None:
        raise KeyError(f'Missing metric {key!r} in {path}')
    return float(value)


def safe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def metric_summary(values: list[float]) -> dict[str, float | int]:
    return {
        'n': len(values),
        'mean': mean(values) if values else math.nan,
        'std': safe_stdev(values) if values else math.nan,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: float, signed: bool = False) -> str:
    if math.isnan(value):
        return 'nan'
    return f'{value:+.4f}' if signed else f'{value:.4f}'


def aggregate_results(config: RunConfig) -> dict[str, Any]:
    if config.baseline_dir is None:
        raise ValueError('baseline_dir is required for aggregation')

    rows: list[dict[str, Any]] = []
    labels: set[str] = set()

    for seed in config.seeds:
        base_path = config.baseline_dir / f'seed_{seed}' / 'quick_summary.json'
        pseudo_path = config.output_dir / f'seed_{seed}' / 'quick_summary.json'

        base = load_json(base_path)
        pseudo = load_json(pseudo_path)

        labels |= set((get_nested(base, 'test_metrics.per_label', {}) or {}).keys())
        labels |= set((get_nested(pseudo, 'test_metrics.per_label', {}) or {}).keys())

        row = {
            'seed': seed,
            'base_micro': as_float(get_nested(base, 'test_metrics.micro.f1'), key='test_metrics.micro.f1', path=base_path),
            'pseudo_micro': as_float(get_nested(pseudo, 'test_metrics.micro.f1'), key='test_metrics.micro.f1', path=pseudo_path),
            'base_macro': as_float(get_nested(base, 'test_metrics.macro_f1'), key='test_metrics.macro_f1', path=base_path),
            'pseudo_macro': as_float(get_nested(pseudo, 'test_metrics.macro_f1'), key='test_metrics.macro_f1', path=pseudo_path),
            '_base': base,
            '_pseudo': pseudo,
        }
        row['delta_micro'] = row['pseudo_micro'] - row['base_micro']
        row['delta_macro'] = row['pseudo_macro'] - row['base_macro']
        rows.append(row)

    per_seed_public = [
        {k: v for k, v in row.items() if not k.startswith('_')}
        for row in rows
    ]

    aggregate: dict[str, Any] = {}
    for key in ['base_micro', 'pseudo_micro', 'delta_micro', 'base_macro', 'pseudo_macro', 'delta_macro']:
        vals = [float(row[key]) for row in rows]
        aggregate[key] = metric_summary(vals)

    aggregate['wins_micro'] = {
        'wins': sum(1 for row in rows if float(row['delta_micro']) > 0),
        'n': len(rows),
    }
    aggregate['wins_macro'] = {
        'wins': sum(1 for row in rows if float(row['delta_macro']) > 0),
        'n': len(rows),
    }

    per_label_rows: list[dict[str, Any]] = []
    for label in sorted(labels):
        base_vals: list[float] = []
        pseudo_vals: list[float] = []
        deltas: list[float] = []

        for row in rows:
            bf = get_nested(row['_base'], f'test_metrics.per_label.{label}.f1')
            pf = get_nested(row['_pseudo'], f'test_metrics.per_label.{label}.f1')
            if bf is None or pf is None:
                continue
            bf = float(bf)
            pf = float(pf)
            base_vals.append(bf)
            pseudo_vals.append(pf)
            deltas.append(pf - bf)

        if deltas:
            per_label_rows.append({
                'label': label,
                'n': len(deltas),
                'base_mean': mean(base_vals),
                'base_std': safe_stdev(base_vals),
                'pseudo_mean': mean(pseudo_vals),
                'pseudo_std': safe_stdev(pseudo_vals),
                'delta_mean': mean(deltas),
                'delta_std': safe_stdev(deltas),
                'wins': sum(1 for delta in deltas if delta > 0),
            })

    return {
        'seeds': config.seeds,
        'baseline_dir': str(config.baseline_dir),
        'pseudo_dir': str(config.output_dir),
        'per_seed': per_seed_public,
        'aggregate': aggregate,
        'per_label': per_label_rows,
    }


def write_reports(config: RunConfig, report: dict[str, Any]) -> None:
    report_dir = config.output_dir / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / 'seed_benchmark_summary.json'
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    per_seed_fields = [
        'seed',
        'base_micro',
        'pseudo_micro',
        'delta_micro',
        'base_macro',
        'pseudo_macro',
        'delta_macro',
    ]
    write_csv(report_dir / 'per_seed_metrics.csv', report['per_seed'], per_seed_fields)

    aggregate_rows = []
    for metric, stats in report['aggregate'].items():
        if metric.startswith('wins_'):
            aggregate_rows.append({
                'metric': metric,
                'n': stats['n'],
                'mean': '',
                'std': '',
                'wins': stats['wins'],
            })
        else:
            aggregate_rows.append({
                'metric': metric,
                'n': stats['n'],
                'mean': stats['mean'],
                'std': stats['std'],
                'wins': '',
            })
    write_csv(report_dir / 'aggregate_metrics.csv', aggregate_rows, ['metric', 'n', 'mean', 'std', 'wins'])

    write_csv(
        report_dir / 'per_label_delta_metrics.csv',
        report['per_label'],
        ['label', 'n', 'base_mean', 'base_std', 'pseudo_mean', 'pseudo_std', 'delta_mean', 'delta_std', 'wins'],
    )

    print(f'\n[reports] {report_dir}')
    print(f'- {json_path}')
    print(f'- {report_dir / "per_seed_metrics.csv"}')
    print(f'- {report_dir / "aggregate_metrics.csv"}')
    print(f'- {report_dir / "per_label_delta_metrics.csv"}')


def print_console_summary(report: dict[str, Any]) -> None:
    print('\nseed | base_micro | pseudo_micro | delta_micro | base_macro | pseudo_macro | delta_macro')
    for row in report['per_seed']:
        print(
            f"{row['seed']} | "
            f"{format_float(row['base_micro'])} | "
            f"{format_float(row['pseudo_micro'])} | "
            f"{format_float(row['delta_micro'], signed=True)} | "
            f"{format_float(row['base_macro'])} | "
            f"{format_float(row['pseudo_macro'])} | "
            f"{format_float(row['delta_macro'], signed=True)}"
        )

    print('\nAggregate')
    for key in ['base_micro', 'pseudo_micro', 'delta_micro', 'base_macro', 'pseudo_macro', 'delta_macro']:
        stats = report['aggregate'][key]
        print(f"{key}: mean={format_float(stats['mean'])} std={format_float(stats['std'])}")

    wins_micro = report['aggregate']['wins_micro']
    wins_macro = report['aggregate']['wins_macro']
    print(f"\nWins micro: {wins_micro['wins']}/{wins_micro['n']}")
    print(f"Wins macro: {wins_macro['wins']}/{wins_macro['n']}")

    print('\nPer-label delta mean')
    for row in report['per_label']:
        print(
            f"{row['label']}: "
            f"base_mean={format_float(row['base_mean'])} "
            f"pseudo_mean={format_float(row['pseudo_mean'])} "
            f"delta_mean={format_float(row['delta_mean'], signed=True)} "
            f"wins={row['wins']}/{row['n']}"
        )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)

    if not config.skip_train:
        run_training(config)

    if not config.skip_eval:
        report = aggregate_results(config)
        print_console_summary(report)
        write_reports(config, report)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
