# Risk-Aware RL for CARLA 0.9.15

This is a from-scratch reproduction scaffold for:

Risk-Aware Reinforcement Learning for Non-Conservative Motion Planning in
Uncertain Autonomous Driving Environments.

Target platform:

- Ubuntu 20.04
- CARLA 0.9.15
- Python 3.8
- 10 Hz simulation
- 400000 environment steps
- 200 steps per episode

## Modules

- `rarl/envs/carla_env.py`: CARLA Gym-like environment.
- `rarl/belief/gm_bayes.py`: Gaussian Mixture Bayesian Belief Updater.
- `rarl/risk/risk_field.py`: time-varying road and vehicle risk field.
- `rarl/models/aca_lstm_td3.py`: ACA-LSTM actor and critic networks.
- `rarl/algos/td3.py`: TD3 training loop components.
- `scripts/train.py`: full training entry.
- `scripts/evaluate.py`: deterministic evaluation entry.

## Install

Create the environment on Ubuntu:

```bash
conda create -n rarl-carla python=3.8 -y
conda activate rarl-carla
pip install -r requirements.txt
```

Install CARLA Python API:

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.15
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg:$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH
```

Or install this project and its Python dependencies in editable mode:

```bash
bash scripts/setup_env.sh
```

Start CARLA:

```bash
$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
```

Train:

```bash
python scripts/train.py --config configs/default.yaml
```

Quick data-flow test without CARLA:

```bash
python scripts/train.py --config configs/default.yaml --dry-run
```

SafeBench-inspired straight-road LC template smoke test:

```bash
python tests/scripts/train_lc_straight_templates.py --template-name straight_follow --episodes 2 --batch-size 1 --no-cuda
```

Natural-traffic and frozen-adversary evaluation templates migrated from
`carla_evolution` live under `a/`. The two primary entry points are:

```bash
python a/scripts/run_ego_natural_seed_eval.py --backend mock --ego-checkpoint /path/to/checkpoint.pt
python a/scripts/evaluate_finetuned_ego_against_adv.py --backend mock --ego-checkpoint /path/to/checkpoint.pt
```

Evaluate:

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint runs/latest.pt
```

## Reproduction Contract

The default config fixes the requested scale:

- `max_timesteps: 400000`
- `episode_horizon: 200`
- expected episodes: `400000 / 200 = 2000`
- `sim_hz: 10`
- one episode lasts 20 simulated seconds.

The scaffold keeps every paper-level module present. The first runnable
version uses conservative approximations where exact paper constants are not
recoverable from text extraction; replace formulas in the module files as you
confirm them from the PDF equations.
