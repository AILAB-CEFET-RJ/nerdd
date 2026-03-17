# Installation Guide

This repo is centered on the `src/` pipeline (training, pseudolabelling, and calibration).

This document covers environment setup only. For pipeline execution, see `docs/RUNBOOK.md`.

## Prerequisites

- Git
- Python 3.11+
- Pip
- Optional: virtualenv or venv

## Clone

```bash
git clone https://github.com/MLRG-CEFET-RJ/nerdd.git
cd nerdd
```

## Project Setup

Requires Python 3.11+.

Dependencies live in `src/requirements.txt`.

```bash
cd src
pip install -r requirements.txt
```

## Installation Check

Use a lightweight CLI help check to confirm the environment is usable:

```bash
cd src
python3 -m base_model_training.evaluate_gliner --help
```

## Notes

- `pywin32` is Windows-only. It is already marked with a platform guard in the requirements file.
- On NVIDIA Blackwell GPUs such as the RTX 5090, use a PyTorch build with `cu128` or newer. Older builds may fail with `no kernel image is available for execution on the device`.
- Operational commands and smoke runs live in `docs/RUNBOOK.md`.

## Troubleshooting

- If `pip install` fails, confirm you are in `src/` and using `src/requirements.txt`.
- If you are on Windows, consider using PowerShell or Git Bash for the commands above.
