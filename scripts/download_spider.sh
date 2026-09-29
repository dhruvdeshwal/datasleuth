#!/usr/bin/env bash
# Download Spider 1.0 + the official evaluator into eval/spider/.
# Usage: bash scripts/download_spider.sh [--testsuite]   (--testsuite adds ~1.3 GB of test-suite DBs)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
[ -x .venv/Scripts/python.exe ] && PY=.venv/Scripts/python.exe
[ -x .venv/bin/python ] && PY=.venv/bin/python
"$PY" -m eval.download "$@"
