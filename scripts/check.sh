#!/usr/bin/env bash
# Fast checks, the same in CI and locally: syntax and the pure-function tests.
# Needs only python3, pytest and numpy (the tests do not import torch or the models).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m compileall -q app tests
python3 -m pytest -q tests
