#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -e .

cat <<'EOF'

Python dependencies are installed.

Before running real CARLA training, set CARLA_ROOT and PYTHONPATH, for example:

  export CARLA_ROOT=/path/to/CARLA_0.9.15
  export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg:$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH

Then start CARLA:

  $CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-rpc-port=2000

Smoke test:

  python scripts/train.py --config configs/compare_3sv.yaml --dry-run

Real training:

  python scripts/train.py --config configs/compare_3sv.yaml
EOF
