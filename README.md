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

### 4.1 固定几何、只调全局转速（omega-only）

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
