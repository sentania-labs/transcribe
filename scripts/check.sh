#!/usr/bin/env bash
# Fast checks, the same in CI and locally: syntax and the pure-function tests.
# Needs python3 with pytest, numpy, fastapi, httpx, python-multipart (no torch, no models).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m compileall -q app tests
python3 -m pytest -q tests
