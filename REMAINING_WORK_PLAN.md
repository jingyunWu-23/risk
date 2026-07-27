# 剩余工作实施方案

本文档说明在当前 `carla_risk_aware_rl` 骨架基础上，还需要完成哪些工作，才能在 `CARLA 0.9.15 + Ubuntu 20.04 + Python 3.8` 上复现完整流程，并与我们的方法进行公平对比。

## 1. 实验原则

主对比实验必须保证环境难度一致。由于我们的方法中背景车一般为 3 辆，因此 risk-aware RL 方法的主对比实验也应设置为 3 辆背景车。

主实验统一条件：

```text
CARLA version: 0.9.15
OS: Ubuntu 20.04
Python: 3.8
simulation frequency: 10 Hz
episode length: 200 steps
training steps: 400000
background vehicles: 3
same Town
same route bank
same spawn seeds
same evaluation episodes
```

论文原始场景可以作为额外复现实验：

```text
Low-Speed Dense: 5-10 SVs, SV speed 5-8 m/s, AV target speed 12-14 m/s
High-Speed Sparse: 1-3 SVs, SV speed 18-22 m/s, AV target speed 22-25 m/s
```

## 2. 当前已完成内容

已新建独立工程：

```text
carla_risk_aware_rl/
```

已完成模块骨架：

```text
rarl/envs/carla_env.py          CARLA Gym-like 环境接口
rarl/belief/gm_bayes.py         Gaussian Mixture Bayesian Belief Updater
rarl/risk/risk_field.py         time-varying risk field
rarl/models/aca_lstm_td3.py     ACA-LSTM Actor + TD3 Critic
rarl/algos/td3.py               TD3 算法
rarl/algos/replay_buffer.py     Replay buffer
scripts/train.py                训练入口
scripts/evaluate.py             评估入口
configs/default.yaml            默认实验配置
```

已固定关键训练参数：

```yaml
max_timesteps: 400000
episode_horizon: 200
fixed_delta_seconds: 0.1
sim_hz: 10
```

已完成 dry-run 假环境下的非 torch 数据流检查：

```text
observation dimension: 62
GMBBU -> risk field -> reward -> observation 拼接正常
```

## 3. 下一步总体路线

推荐按下面顺序完成，不要一开始就追求所有细节完全还原。

```text
阶段 A: 接通真实 CARLA 环境
阶段 B: 固定 3 辆背景车公平对比场景
阶段 C: 接入完整观测、奖励、指标
阶段 D: 跑 400000 步训练
阶段 E: 做对比、消融、泛化实验
阶段 F: 整理论文图表和实验记录
```

## 4. 阶段 A：接通真实 CARLA 环境

目标：让 `CarlaRiskAwareEnv` 不依赖 `--dry-run`，可以真实 reset、step、close。

需要实现位置：

```text
rarl/envs/carla_env.py
```

需要完成函数：

```python
def _connect_carla(self):
    ...

def _spawn_ego_vehicle(self):
    ...

def _spawn_surrounding_vehicles(self):
    ...

def _apply_ego_action(self, action):
    ...

def _tick_world(self):
    ...

def _cleanup_actors(self):
    ...
```

CARLA 设置必须使用同步模式：

```python
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.1
settings.no_rendering_mode = True
world.apply_settings(settings)
```

动作建议先定义为 2 维：

```text
action[0]: acceleration command, range [-1, 1]
action[1]: steering command, range [-1, 1]
```

映射到 CARLA：

```text
acceleration >= 0 -> throttle
acceleration < 0  -> brake
steer             -> steer
```

## 5. 阶段 B：构建 3 辆背景车公平对比场景

目标：主实验所有方法使用同一套场景。

建议新增文件：

```text
configs/compare_3sv.yaml
rarl/envs/route_bank.py
```

`compare_3sv.yaml` 推荐设置：

```yaml
traffic:
  surrounding_vehicles_min: 3
  surrounding_vehicles_max: 3
  sv_speed_min: 5.0
  sv_speed_max: 12.0
  av_target_speed_min: 12.0
  av_target_speed_max: 14.0
  lane_change_probability: 0.2
  max_lane_changes_per_episode: 3
```

route bank 每条记录至少包含：

```json
{
  "scenario_id": 0,
  "town": "Town04",
  "ego_spawn_index": 31,
  "ego_route_end_index": 75,
  "sv_spawn_indices": [35, 40, 45],
  "sv_target_speeds": [8.0, 9.0, 10.0]
}
```

主实验建议固定：

```text
train route: 100 条
eval route: 30 条
seed: 0, 1, 2
```

## 6. 阶段 C：构造真实观测

第一版使用低维状态观测，不使用相机或激光雷达图像。

观测结构：

```text
obs = [ego_state, sv_belief_features, risk_features]
```

默认维度：

```text
ego_state: 8
max_svs: 10
sv_belief_feature: 5
risk_features: 4
total: 8 + 10 * 5 + 4 = 62
```

如果主实验固定 3 辆背景车，也可以把 `max_surrounding_vehicles` 改为 3：

```text
total: 8 + 3 * 5 + 4 = 27
```

为了和后续泛化到 5、8、10 辆背景车兼容，建议仍保留 `max_surrounding_vehicles: 10`，主实验只有 3 辆时其余位置补 0。

需要实现函数：

```python
def _read_ego_state(self):
    ...

def _read_surrounding_vehicle_observations(self):
    ...

def _to_ego_frame(self, ego_transform, vector_or_location):
    ...

def _build_observation(self):
    ...
```

ego state：

```text
[x_local, y_local, yaw_local, vx_local, vy_local, speed, ax_local, ay_local]
```

每辆周车原始输入：

```text
[rel_x, rel_y, rel_vx, rel_vy]
```

GMBBU 输出每辆周车：

```text
[mean_x, mean_y, mean_vx, mean_vy, uncertainty]
```

risk features：

```text
[road_risk, vehicle_risk, total_risk, risk_margin]
```

## 7. 阶段 D：完善奖励和指标

奖励函数建议先保持可解释：

```text
reward =
  progress_weight * route_progress_delta
+ speed_weight * speed_tracking
- risk_weight * total_risk
- collision_penalty * collision
- offroad_penalty * offroad
- action_smoothness_weight * action_change
```

需要统计的训练和评估指标：

```text
episode_reward
route_completion
collision
collision_rate
offroad
offroad_rate
avg_speed
min_distance
min_ttc
total_risk
episode_length
```

建议新增：

```text
rarl/envs/metrics.py
rarl/utils/logger.py
```

## 8. 阶段 E：训练流程

启动 CARLA：

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.15
$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
```

配置 Python API：

```bash
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg:$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH
```

安装依赖：

```bash
conda create -n rarl-carla python=3.8 -y
conda activate rarl-carla
pip install -r requirements.txt
```

先跑 smoke test：

```bash
python scripts/train.py --config configs/default.yaml --dry-run
```

真实训练：

```bash
python scripts/train.py --config configs/compare_3sv.yaml
```

训练规模：

```text
400000 steps / 200 steps per episode = 2000 episodes
```

建议每 20000 步保存一次 checkpoint，每 20000 步评估一次。

## 9. 阶段 F：实验设计

主对比实验：

```text
Ours
Risk-aware RL reproduction
TD3 baseline
LSTM-TD3 baseline
Rule-based baseline
```

所有方法统一：

```text
3 background vehicles
same seeds
same routes
same episode length
same training steps
same evaluation routes
```

消融实验：

```text
w/o GMBBU
w/o risk field
w/o ACA
w/o LSTM
```

泛化实验：

```text
background vehicles = 1, 3, 5, 8, 10
```

论文原始场景复现实验：

```text
Low-Speed Dense: 5-10 SVs
High-Speed Sparse: 1-3 SVs
```

## 10. 推荐完成顺序清单

```text
[ ] 1. 新增 compare_3sv.yaml
[ ] 2. 新增 route_bank.py，固定 train/eval routes
[ ] 3. 完成 CARLA actor spawn 和 cleanup
[ ] 4. 完成 ego action 到 VehicleControl 的映射
[ ] 5. 完成真实 ego_state 读取
[ ] 6. 完成真实 surrounding vehicle state 读取
[ ] 7. 完成 ego 坐标系转换
[ ] 8. 接入 GMBBU belief update
[ ] 9. 接入真实 road_risk / vehicle_risk
[ ] 10. 完成 collision/offroad/route_completion/min_ttc 指标
[ ] 11. 跑 1 episode smoke test
[ ] 12. 跑 1000 step short training
[ ] 13. 跑 400000 step full training
[ ] 14. 跑 fixed-route evaluation
[ ] 15. 跑 ours vs reproduction 主对比
[ ] 16. 跑消融实验
[ ] 17. 跑背景车数量泛化实验
```

## 11. 当前最优先任务

最优先不是调网络，而是把真实 CARLA 环境接通：

```text
reset -> spawn ego + 3 SVs -> build obs -> apply action -> tick -> compute reward/info -> done
```

只要这条链路跑通，后面的 TD3、GMBBU、risk field、ACA-LSTM 都已经有位置接入。
