# Droplets_DRL 使用说明（命令整理）

本文件整理当前仓库常用训练/评估命令，方便后续复现实验。

## 1. 先说明一个原则

- 默认配置都在 `config.py`。
- 但如果命令行传了参数，命令行参数会覆盖 `config.py` 当次运行的默认值。
- 想保持“全局默认”，就只改 `config.py`。
- 想做批量扫描，不改全局默认，优先用命令行覆盖参数。

## 2. 常见 `layout-mode`

- `free_layout`
- `fixed_grid_3x3`
- `free_layout_inflow`
- `free_layout_inflow_u_fixed`
- `fixed_grid_3x3_inflow`
- `fixed_grid_3x3_inflow_u_fixed`
- `gate3_layout_inflow`
- `gate3_layout_inflow_u_fixed`
- `logic_box_layout_inflow`
- `logic_box_layout_inflow_u_fixed`

当前逻辑门任务常用：
- `logic_box_layout_inflow_u_fixed`（水平来流固定，训练更稳）

## 3. 单路由训练（一个入口 -> 一个出口）

示例：`L1 -> R1`

```powershell
python train.py --layout-mode logic_box_layout_inflow_u_fixed --path-type logic_route --logic-route-mode single --source-port L1 --target-port R1 --total-timesteps 240000 --eval-freq 20000 --workers 1 --run-alias single_L1_R1
```

## 4. 单入口多出口训练（one-to-three）

示例：`L1 -> {T0, R0, B0}`，按 `mean` 选 best。

```powershell
python train.py --layout-mode logic_box_layout_inflow_u_fixed --path-type logic_route --logic-route-mode single_multi_target --source-port L1 --target-port T0 --target-port-set T0,R0,B0 --total-timesteps 240000 --eval-freq 20000 --workers 1 --run-alias one2three_mean
```

说明：
- `--target-port` 在 `single_multi_target` 下只是占位主目标。
- 实际目标集合由 `--target-port-set` 决定。
- 目标采样默认建议用 `LogicBoxConfig.TARGET_SAMPLE_MODE = "cycle"`，保证 `T0/R0/B0` 轮流训练，不会长期偏向某两个口。

### 4.1 固定几何、只调全局转速（one-shot omega-only）

目标：
- 圆柱 `x/y/r` 在一局中完全固定；
- 水平来流固定；
- 策略只输出一个全局 `omega`，通过转速大小和方向让同一入口去不同出口。

关键配置（`config.py`）：
- `LayoutModeConfig.LAYOUT_MODE = "logic_box_layout_inflow_u_fixed"`
- `LogicBoxConfig.ROUTE_MODE = "single_multi_target"`
- `LogicBoxConfig.FIXED_LAYOUT_ENABLE = True`
- `LogicBoxConfig.FIXED_GEOMETRY_ENABLE = True`
- `LogicBoxConfig.FIXED_LAYOUT_X/Y/R` 定义固定圆柱位置和半径
- `GlobalOmegaControlConfig.OPTIMIZE_OMEGA = True`
- `GlobalOmegaControlConfig.MODE = "continuous"`

训练示例：

```powershell
python train.py --layout-mode logic_box_layout_inflow_u_fixed --path-type logic_route --logic-route-mode single_multi_target --source-port L1 --target-port R0 --target-port-set R0,R1,R2 --total-timesteps 240000 --eval-freq 40000 --workers 1 --run-alias omega_only_fixed_geom
```

测试某个目标出口：

```powershell
python test.py --layout-mode logic_box_layout_inflow_u_fixed --model models\best\<your_tag>\best_model.zip --vecnorm models\best\<your_tag>\vecnormalize_best.pkl --logic-target-port R0
python test.py --layout-mode logic_box_layout_inflow_u_fixed --model models\best\<your_tag>\best_model.zip --vecnorm models\best\<your_tag>\vecnormalize_best.pkl --logic-target-port R1
python test.py --layout-mode logic_box_layout_inflow_u_fixed --model models\best\<your_tag>\best_model.zip --vecnorm models\best\<your_tag>\vecnormalize_best.pkl --logic-target-port R2
```

### 4.2 固定几何、实时连续调转速（dynamic omega）

目标：
- 圆柱 `x/y/r` 固定；
- 水平来流固定；
- 策略每个时间步只输出一个 `delta_omega`；
- 环境真实推进粒子，直到粒子出边界、碰撞或超时；
- 用目标出口是否命中作为主要奖励。

这个模式更接近磁性旋转油滴实验：转速可以实时改变，但变化率受限，不是瞬间跳到任意值。

关键配置（`config.py`）：
- `LayoutModeConfig.LAYOUT_MODE = "logic_box_layout_inflow_u_fixed_omega_rt"`
- `LogicBoxConfig.ROUTE_MODE = "single_multi_target"`
- `LogicBoxConfig.FIXED_LAYOUT_ENABLE = True`
- `LogicBoxConfig.FIXED_GEOMETRY_ENABLE = True`
- `LogicBoxConfig.FIXED_LAYOUT_X/Y/R` 定义固定圆柱位置和半径
- `GlobalOmegaControlConfig.OPTIMIZE_OMEGA = True`
- `GlobalOmegaControlConfig.OMEGA_MIN / OMEGA_MAX` 控制允许转速范围
- `DynamicOmegaControlConfig.MAX_DELTA_OMEGA` 控制每步最大转速变化
- `DynamicOmegaControlConfig.MAX_STEPS` 控制一局最多控制步数

当前默认最大转速设为 `1 Hz`：

```python
GlobalOmegaControlConfig.OMEGA_MIN = -2.0 * np.pi
GlobalOmegaControlConfig.OMEGA_MAX = 2.0 * np.pi
```

训练示例：

```powershell
python train.py --layout-mode logic_box_layout_inflow_u_fixed_omega_rt --path-type logic_route --logic-route-mode single_multi_target --source-port L1 --target-port R0 --target-port-set R0,R1,R2 --total-timesteps 500000 --eval-freq 20000 --workers 1 --run-alias omega_rt_fixed_geom
```

测试某个目标出口：

```powershell
python test.py --layout-mode logic_box_layout_inflow_u_fixed_omega_rt --model models\best\<your_tag>\best_model.zip --vecnorm models\best\<your_tag>\vecnormalize_best.pkl --logic-target-port R0 --rollout-steps 160
python test.py --layout-mode logic_box_layout_inflow_u_fixed_omega_rt --model models\best\<your_tag>\best_model.zip --vecnorm models\best\<your_tag>\vecnormalize_best.pkl --logic-target-port R1 --rollout-steps 160
python test.py --layout-mode logic_box_layout_inflow_u_fixed_omega_rt --model models\best\<your_tag>\best_model.zip --vecnorm models\best\<your_tag>\vecnormalize_best.pkl --logic-target-port R2 --rollout-steps 160
```

### 4.2.1 搜索通用布局、局内锁定布局、实时调转速

如果目标是“训练出一套通用 `x/y/r` 布局，并在每一局内部锁定它，只靠实时 `omega(t)` 把同一入口送到不同出口”，用：

```text
logic_box_layout_inflow_u_fixed_omega_rt_design
```

这个模式默认使用 `structure-control policy`，把结构和控制拆成两个时间尺度：

```text
慢变量结构：x/y/r 是一组全局可学习参数，不看目标出口，也不随粒子状态变化
快变量控制：每一步根据粒子状态、目标出口、当前 omega 输出 delta_omega
环境执行：每局开始锁定当前 x/y/r，局内只更新 omega(t)
训练过程中：PPO 仍会更新全局 x/y/r，因此能继续探索更好的通用布局
```

这比旧的 shared-geometry policy 更硬：旧策略虽然屏蔽目标，但几何 head 仍会随状态输出；
新策略直接把 `x/y/r` 变成目标无关的全局结构参数，更符合“同一个物理装置，只改变转速控制”的实验设定。

训练命令：

```powershell
python train.py `
  --layout-mode logic_box_layout_inflow_u_fixed_omega_rt_design `
  --path-type logic_route `
  --logic-route-mode single_multi_target `
  --source-port L1 `
  --target-port R0 `
  --target-port-set R0,R1,R2 `
  --num-cylinders 9 `
  --total-timesteps 1000000 `
  --eval-freq 20000 `
  --workers 1 `
  --run-alias omega_rt_design_L1_R012
```

如果需要退回旧 shared-geometry 策略，可加：

```powershell
--no-structure-control-policy
```

测试：

```powershell
python test.py --layout-mode logic_box_layout_inflow_u_fixed_omega_rt_design --model models\best\<your_tag>\best_model.zip --vecnorm models\best\<your_tag>\vecnormalize_best.pkl --logic-target-port R0 --rollout-steps 260
python test.py --layout-mode logic_box_layout_inflow_u_fixed_omega_rt_design --model models\best\<your_tag>\best_model.zip --vecnorm models\best\<your_tag>\vecnormalize_best.pkl --logic-target-port R1 --rollout-steps 260
python test.py --layout-mode logic_box_layout_inflow_u_fixed_omega_rt_design --model models\best\<your_tag>\best_model.zip --vecnorm models\best\<your_tag>\vecnormalize_best.pkl --logic-target-port R2 --rollout-steps 260
```

### 4.2.2 固定位置、只训练半径、实时调转速

如果先降低维度，让 9 个转子位置固定为 `LogicBoxConfig.FIXED_LAYOUT_X/Y`，只训练每个圆柱半径 `r_i`，并在局内实时控制统一 `omega(t)`，用：

```text
logic_box_layout_inflow_u_fixed_omega_rt_radius_design
```

这个模式的动作维度是：

```text
N 个半径动作 + 1 个 delta_omega
```

例如 `N=9` 时，动作维度从自由布局模式的 `28 = 9*3 + 1` 降到：

```text
10 = 9 + 1
```

训练命令：

```powershell
python train.py `
  --layout-mode logic_box_layout_inflow_u_fixed_omega_rt_radius_design `
  --path-type logic_route `
  --logic-route-mode single_multi_target `
  --source-port L1 `
  --target-port-set R0,R1,R2 `
  --num-cylinders 9 `
  --total-timesteps 1000000 `
  --eval-freq 20000 `
  --workers 1 `
  --run-alias radius_rt_L1_R012
```

测试：

```powershell
python test.py --layout-mode logic_box_layout_inflow_u_fixed_omega_rt_radius_design --model models\best\<your_tag>\best_model.zip --vecnorm models\best\<your_tag>\vecnormalize_best.pkl --logic-target-port R0 --rollout-steps 260
python test.py --layout-mode logic_box_layout_inflow_u_fixed_omega_rt_radius_design --model models\best\<your_tag>\best_model.zip --vecnorm models\best\<your_tag>\vecnormalize_best.pkl --logic-target-port R1 --rollout-steps 260
python test.py --layout-mode logic_box_layout_inflow_u_fixed_omega_rt_radius_design --model models\best\<your_tag>\best_model.zip --vecnorm models\best\<your_tag>\vecnormalize_best.pkl --logic-target-port R2 --rollout-steps 260
```

### 4.2.3 3x3 固定半径笔画训练

如果目标是先验证“固定 3x3 位置和半径，只靠实时统一转速 `omega(t)` 让粒子轨迹贴合一个简单笔画”，用独立脚本：

```text
train_3x3_stroke_omega.py
```

这个脚本不使用 logic-box 端口奖励，动作只有：

```text
delta_omega
```

默认输出统一放到：

```text
models/stroke/<run_tag>/
```

先建议训练最容易的 `soft_u`：

```powershell
python train_3x3_stroke_omega.py `
  --stroke soft_u `
  --radius 0.010 `
  --total-timesteps 300000 `
  --eval-freq 10000 `
  --run-alias first_soft_u
```

输出内容：

```text
models/stroke/<run_tag>/best/best_model.zip
models/stroke/<run_tag>/best/vecnormalize_best.pkl
models/stroke/<run_tag>/best/best.png
models/stroke/<run_tag>/best/best_omega_trace.csv
models/stroke/<run_tag>/frames/eval_*.png
models/stroke/<run_tag>/traces/eval_*_omega_trace.csv
```

### 4.2.4 2x2 west 四个子任务笔画训练

如果目标是先把手写 `west` 拆成四个更简单的子任务，并且使用 `2x2` 固定圆柱、水平来流、实时统一转速 `omega(t)`，用独立脚本：

```text
train_2x2_west_omega.py
```

核心约束：
- 圆柱数量固定为 `4`，排布为 `2x2`；
- 圆柱半径固定；
- 圆心距默认自动设置为 `6 * radius`，也就是 `3` 个圆柱直径；
- 有水平来流，默认 `--inflow-u 0.006`；
- 动作只有一个：`delta_omega`，表示每个控制步调整统一全局转速；
- 四个子任务分别是 `--stroke w/e/s/t`，建议分别训练。

训练命令示例：

```powershell
python train_2x2_west_omega.py `
  --stroke w `
  --radius 0.010 `
  --total-timesteps 300000 `
  --eval-freq 10000 `
  --run-alias west_w
```

四个子任务可以分别跑：

```powershell
python train_2x2_west_omega.py --stroke w --radius 0.010 --total-timesteps 300000 --eval-freq 10000 --run-alias west_w
python train_2x2_west_omega.py --stroke e --radius 0.010 --total-timesteps 300000 --eval-freq 10000 --run-alias west_e
python train_2x2_west_omega.py --stroke s --radius 0.010 --total-timesteps 300000 --eval-freq 10000 --run-alias west_s
python train_2x2_west_omega.py --stroke t --radius 0.010 --total-timesteps 300000 --eval-freq 10000 --run-alias west_t
```

输出默认在：

```text
models/stroke/west_2x2/<run_tag>/
```

其中：
- `best/best_model.zip`：当前最好模型；
- `best/vecnormalize_best.pkl`：归一化参数；
- `best/frames/best.png`：最好评估图；
- `best/traces/best_omega_trace.csv`：最好轨迹对应的转速时序；
- `frames/eval_*.png`：每次评估图；
- `traces/eval_*_omega_trace.csv`：每次评估的转速时序。

如果要手动改变圆心距，可以加 `--pitch`；不加时始终满足 `圆心距 = 3 * 直径`。

### 4.3 先找布局，再锁定布局训练实时转速

如果你还没有固定几何，使用这个两阶段脚本：

```text
A阶段：自由搜索同一套 x/y/r 布局
B阶段：锁定 A 阶段 best 的 x/y/r，只训练实时 omega(t)
```

命令：

```powershell
python train_find_layout_then_dynamic_omega.py `
  --source-port L1 `
  --targets R0,R1,R2 `
  --num-cylinders 9 `
  --stage-a-steps 600000 `
  --stage-b-steps 800000 `
  --eval-freq 20000 `
  --workers 1 `
  --run-prefix L1_R012_find_then_rt
```

输出：
- A 阶段模型：`models/best/*_A_find_geom/`
- B 阶段模型：`models/best/*_B_omega_rt/`
- 中间固定几何：`models/best/<run-prefix>_find_geom_then_omega_rt_bundle/fixed_geometry_from_stage_a.npz`
- 最终模型副本：`models/best/<run-prefix>_find_geom_then_omega_rt_bundle/final_dynamic_omega/`

如果 A 阶段已经训练过，只想从已有 A 模型继续做 B 阶段：

```powershell
python train_find_layout_then_dynamic_omega.py `
  --source-port L1 `
  --targets R0,R1,R2 `
  --skip-stage-a `
  --stage-a-model models\best\<stage_a_tag>\best_model.zip `
  --stage-a-vecnorm models\best\<stage_a_tag>\vecnormalize_best.pkl `
  --stage-b-steps 800000 `
  --eval-freq 20000 `
  --workers 1 `
  --run-prefix L1_R012_find_then_rt
```

## 5. 扫描圆柱数量 N 和最小半径 r_min（推荐）

新增脚本：`batch_sweep_logic_n_rmin.py`

```powershell
python batch_sweep_logic_n_rmin.py --layout-mode logic_box_layout_inflow_u_fixed --source-port L1 --targets T0,R0,B0 --num-cylinders-list 4,6,8,10 --min-active-r-list 0.0030,0.0035,0.0040 --eval-cycles 3 --eval-freq 20000 --workers 1 --run-prefix one2three_mean
```

常用可选项：
- `--logic-max-r 0.020`
- `--forbid-elimination` 或 `--allow-elimination`
- `--clean`（先清同名输出目录）
- `--dry-run`（只打印命令不执行）

如果你不想扫半径、只想使用 `config.py` 里的 `MIN_ACTIVE_R`：

```powershell
python batch_sweep_logic_n_rmin.py --layout-mode logic_box_layout_inflow_u_fixed --source-port L1 --targets T0,R0,B0 --num-cylinders-list 4,5,6,7 --use-config-min-active-r --eval-cycles 10 --eval-freq 20000 --workers 1 --run-prefix one2three_mean
```

说明：
- 加了 `--use-config-min-active-r` 后，脚本不会传 `--logic-min-active-r`。
- 训练直接读取 `config.py` 的 `LogicBoxConfig.MIN_ACTIVE_R`。

## 6. 多目标逐个训练（老脚本）

脚本：`batch_train_logic_routes.py`

```powershell
python batch_train_logic_routes.py --source-port L1 --targets T0,R0,R1,R2,B0 --layout-mode logic_box_layout_inflow_u_fixed --path-type logic_route --eval-cycles 3 --eval-freq 20000 --workers 1
```

说明：
- 这个脚本是“每个目标口单独训练一个模型”，不是 one-to-three 共享策略。

## 7. 课程式 one-to-three（三阶段自动续训）

脚本：`batch_curriculum_one2three.py`

默认三阶段：
- Stage1: `T0`
- Stage2: `T0,R0`
- Stage3: `T0,R0,B0`

```powershell
python batch_curriculum_one2three.py --layout-mode logic_box_layout_inflow_u_fixed --source-port L1 --eval-cycles 3,3,3 --eval-freq 20000 --workers 1 --run-prefix one2three_curriculum
```

可选：
- 自定义阶段目标：`--stage-targets "T0;T0,R0;T0,R0,B0"`
- 覆盖几何参数：`--num-cylinders`、`--logic-min-active-r`、`--logic-max-r`
- 消灭开关：`--logic-forbid-elimination` / `--logic-allow-elimination`
- 仅打印命令：`--dry-run`

### 7.1 你当前这套顺序（T0 -> B0 -> R0）

```powershell
python batch_curriculum_one2three.py `
  --layout-mode logic_box_layout_inflow_u_fixed `
  --source-port L1 `
  --stage-targets "T0;T0,B0;T0,B0,R0" `
  --eval-cycles "3,3,3" `
  --eval-freq 20000 `
  --workers 1 `
  --num-cylinders 10 `
  --run-prefix one2three_t0b0r0_autoN `
  --python-exe D:\CODING\Anaconda\envs\vortex-rl\python.exe
```

参数说明：
- `--num-cylinders 10`：总候选圆柱上限是 10（不是强制都要用）。
- `--python-exe ...`：指定用哪个 Python 环境跑 `train.py`，避免调用错环境（如缺 `torch`）。

### 7.2 若目标是“模型自己决定用几个圆柱”

必须满足：
- `config.py` 里 `LogicBoxConfig.FORBID_ELIMINATION = False`
- 不要在命令里传 `--logic-forbid-elimination`
- 若使用阶段激活功能，保持 `ACTIVE_CYL_LIMIT = 0`（或不传 `--active-cyl-stages`），避免人为限制数量

### 7.3 基于已有 best 模型继续优化（独立脚本）

脚本：`batch_continue_refine.py`

作用：
- 不改主环境逻辑，只是按顺序调用两次 `train.py`。
- Phase1：固定布局微调（以半径/转速为主）。
- Phase2：继续训练并解冻到自由布局（从 Phase1 best 接续）。
- 默认先只跑 Phase1（`--phase2-cycles` 默认是 `0`）。

示例：

```powershell
python batch_continue_refine.py `
  --layout-mode logic_box_layout_inflow_u_fixed `
  --path-type logic_route `
  --source-port L1 `
  --targets T0,R0,R2 `
  --init-model models\best\logic_box_layout_inflow_u_fixed_logic_route_L1_to_T0R0R2__bnd_local__one2three_mean_Nonvoerlap\best_model.zip `
  --init-vecnorm models\best\logic_box_layout_inflow_u_fixed_logic_route_L1_to_T0R0R2__bnd_local__one2three_mean_Nonvoerlap\vecnormalize_best.pkl `
  --phase1-cycles 25 `
  --phase1-lr 1e-4 `
  --eval-freq 20000 `
  --workers 1 `
  --run-prefix cont_refine `
```

说明：
- `phase1` 会使用 `--bootstrap-fixed-layout-from-init`，先把已有模型推断出的布局冻结后微调。
- 只做 Phase1 时，不需要传 `--phase2-cycles`（默认已跳过），也不需要 `--phase2-lr`。
- `phase2` 会从 `phase1` 的 best 继续训练，并启用 `--shared-xy-one-stage-enable` 进行自由布局续训。
- 如果需要补跑 Phase2，示例：`--phase2-cycles 30 --phase2-lr 5e-5`。
- 学习率建议：`phase1-lr=1e-4`，`phase2-lr=5e-5`（更稳，降低解冻后漂移）。
- `train.py` 也支持单独传 `--learning-rate`，用于普通训练或手动续训。
- 输出目录：
  - `models/best/...__<run-prefix>_p1_fixed/`
  - `models/best/...__<run-prefix>_p2_free/`

## 8. 测试与长轨迹可视化

默认会自动找 `models/best/<task_tag>/best_model.zip`。

```powershell
python test.py --layout-mode logic_box_layout_inflow_u_fixed --rollout-steps 600 --frame-stride 1 --fps 20
```

指定模型测试：

```powershell
python test.py --layout-mode logic_box_layout_inflow_u_fixed --model models\best\<your_tag>\best_model.zip --vecnorm models\best\<your_tag>\vecnormalize_best.pkl --rollout-steps 600 --fps 20 --out models\best\<your_tag>\rollout\long_rollout.gif
```

常用显示参数：
- `--show-target-point`
- `--hide-target-path`
- `--hide-logic-seeds`
- `--show-port-only-overlays`

## 9. 输出目录命名规则

`train.py` 会按任务自动生成 tag，主要由这些字段组成：

- `layout-mode`
- `path-type`
- source/target（或 target-set）
- bounds 标识（`bnd_local` / `bnd_global`）
- `RUN_ALIAS` 或 `--run-alias`

输出根目录：
- `models/best/<task_tag>/`
- `models/eval/<task_tag>/`

## 10. 当前与你问题最相关的关键参数（`config.py`）

- 圆柱总数：`StokesCylinderConfig.NUM_CYLINDERS`
- 全局物理半径下限：`StokesCylinderConfig.MIN_R`
- 逻辑模式半径上限：`LogicBoxConfig.MAX_R`
- 逻辑模式最小激活半径：`LogicBoxConfig.MIN_ACTIVE_R`
- 是否禁止“消灭圆柱”：`LogicBoxConfig.FORBID_ELIMINATION`
- 阶段激活上限（0=不限制）：`LogicBoxConfig.ACTIVE_CYL_LIMIT`
- 非激活圆柱锁定半径：`LogicBoxConfig.INACTIVE_LOCK_R`
- 多目标 best 指标：`LogicBoxConfig.MULTI_TARGET_BEST_METRIC`
- 多目标集合：`LogicBoxConfig.TARGET_PORT_SET`

## 11. 一个实用建议

做对比实验时，每次都改 `--run-alias`（或 `RUN_ALIAS`），避免结果互相覆盖。

## 12. task1+task2 合并版（同一布局 + 可切换三入口映射）

目标：
- 同一套 `x/y` 布局下，三入口同时评估三条路由；
- 通过切换 `r/omega`（以及可选来流）在不同“映射模式”之间切换。

配置（`config.py`）：
- `LogicBoxConfig.ROUTE_MODE = "multi_map_switch"`
- `LogicBoxConfig.REWARD_MODE = "port_only"`
- `LogicBoxConfig.MULTI_TARGET_BEST_METRIC = "mean"`
- `LogicBoxConfig.HARD_MIN_TRAIN_ENABLE = False`
- 在 `LogicBoxConfig.MULTI_ROUTE_SETS` 中定义多个映射模式（每个模式是一组 3 条 `(Lx, ?)` 路由）。

训练示例：

```powershell
python train.py --layout-mode logic_box_layout_inflow_u_fixed --path-type logic_route --logic-route-mode multi_map_switch --total-timesteps 240000 --eval-freq 20000 --workers 1 --run-alias multi3x3_switch_mean
```

测试指定映射模式：

```powershell
python test.py --layout-mode logic_box_layout_inflow_u_fixed --logic-route-set-idx 0
python test.py --layout-mode logic_box_layout_inflow_u_fixed --logic-route-set-idx 1
```

## 13. A/B 交替脚本（同一目录归档）

脚本：`batch_ab_multiswitch.py`

作用：
- A 阶段：放开 `x/y/r/omega`（`--shared-xy-one-stage-enable`）找布局；
- B 阶段：从 A 阶段 best 启动并 `--bootstrap-fixed-layout-from-init`，冻结 `x/y` 只训控制码；
- 自动执行 `A0 -> B0 -> A1 -> B1`；
- 把每阶段产物统一复制到一个目录：`models/ab_runs/<run-prefix>__<layout>__<path>/`。

示例：

```powershell
python batch_ab_multiswitch.py `
  --layout-mode logic_box_layout_inflow_u_fixed `
  --path-type logic_route `
  --route-mode multi_map_switch `
  --stage-cycles 15,15,4,8 `
  --stage-lrs 1e-4,5e-5,5e-5,3e-5 `
  --eval-freq 20000 `
  --workers 1 `
  --run-prefix multi3x3_ab_mean `
  --run-tests
```

说明：
- `--stage-cycles` / `--stage-lrs` 顺序固定为 `[A0,B0,A1,B1]`。
- 任一阶段填 `0` 可跳过该阶段（例如只跑 `A0,B0`）。
- `--run-tests` 会在最终模型上按 route-set index 逐个调用 `test.py`，GIF 也归档到同一 bundle 目录。

## 14. 实验参数导出（圆柱位置/直径/转速/来流）

脚本：`export_experiment_layout.py`

作用：
- 从 `best_model.zip` 解码当前最优布局；
- 输出每个圆柱的位置、半径、直径；
- 输出全局转速 `omega` 的 `rad/s`、`Hz`、`rpm`；
- 输出来流大小，以及来流相对圆柱表面转速的无量纲比值；
- 默认把 logic box 的高度映射为实验平台宽度。

示例：实验平台宽度 `1.4 cm`，并行三粒子模式 `multi_map`：

```powershell
python export_experiment_layout.py `
  --layout-mode logic_box_layout_inflow_u_fixed `
  --logic-route-mode multi_map `
  --num-cylinders 6 `
  --run-alias <your_run_alias> `
  --platform-width-cm 1.4
```

如果是可重构并行模式 `multi_map_switch`，指定某个 route-set：

```powershell
python export_experiment_layout.py `
  --layout-mode logic_box_layout_inflow_u_fixed `
  --logic-route-mode multi_map_switch `
  --logic-route-set-idx 0 `
  --num-cylinders 6 `
  --run-alias <your_run_alias> `
  --platform-width-cm 1.4
```

指定模型路径：

```powershell
python export_experiment_layout.py `
  --model models\best\<task_tag>\best_model.zip `
  --vecnorm models\best\<task_tag>\vecnormalize_best.pkl `
  --layout-mode logic_box_layout_inflow_u_fixed `
  --logic-route-mode multi_map `
  --num-cylinders 6 `
  --platform-width-cm 1.4
```

输出文件：
- `experiment_layout.csv`：实验简表，一行一个圆柱，只保留 `idx`、左下角原点位置 `x/y(mm)`、直径 `diameter_mm`。
- `experiment_layout_controls.csv`：全局控制量，只写一行，包括 `omega`、来流、来流/圆柱表面速度比、缩放比例。
- `experiment_layout.json`：完整数据，包含仿真坐标、居中坐标、半径、直径、route 信息和元数据。

默认位置：和 `best_model.zip` 在同一个文件夹。

换算说明：
- 默认 `--scale-ref logic_box_y`，即 `LogicBoxConfig.BOX_Y_RANGE` 的高度对应实验平台宽度 `1.4 cm`。
- 如果要用整张绘图区高度对应 `1.4 cm`，加 `--scale-ref render_y`。
