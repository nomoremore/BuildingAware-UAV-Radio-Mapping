# FARM 场景 11 适配版 LANC 实验

本项目把 FARM `scene_11` 的三维 ARM `.npy` 文件转换成点级样本，训练 Building-aware Simplified LANC，并输出预测指标、真实图、误差图、二维高度切片和三维无线信号地图。

## 数据位置

默认读取：

```powershell
.\11\tx_positions_512.csv
.\11\11\freq35\pattern_120\*.npy
```

当前默认设置：

- 频率：`freq35`
- 天线：`pattern_120`
- 高度：`100/110/120/130m`
- 标签：FARM `uint8` 编码值归一化后的 `signal_norm`

注意：当前没有把 FARM 标签强行转换成 dBm，因为完整解码公式还未确认。

## 运行

快速 smoke test：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_experiment --farm-root .\11 --quick --epochs 3 --model lanc_v2
```

V2 单场景完整训练：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_experiment --farm-root .\11 --epochs 50 --stride 16 --model lanc_v2
```

复现 V1 baseline：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_experiment --farm-root .\11 --epochs 50 --stride 16 --model lanc_v1
```

如果想强制 CPU：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_experiment --farm-root .\11 --quick --cpu --model lanc_v2
```

## 输出

结果会写到：

```text
runs/farm_lanc_v2_scene11_YYYYMMDD_HHMMSS/
```

主要文件：

- `farm_lanc_pairs.csv`：FARM 转换后的点级 LANC 样本。
- `splits.json`：按 `tx_id` 划分的训练/验证/测试集。
- `metrics.csv`：点级误差和覆盖图误差汇总。
- `predictions_test.csv`：测试集逐点预测结果。
- `coverage_map_pred.csv`：预测三维无线信号地图。
- `coverage_map_true.csv`：FARM 真值三维无线信号地图。
- `coverage_map_error.csv`：预测图与真值图的误差。
- `coverage_metrics.csv`：完整三维图层面的 NMSE、RMSE、PSNR、SSIM 和 mean error。
- `coverage_metrics_by_height.csv`：按高度层统计的 NMSE、RMSE、PSNR、SSIM。
- `candidate_pool.csv`：后续主动选点候选池。
- `BuildingAwareLANCV2.pt`：训练好的 V2 模型权重。
- `training_curves.png`：训练曲线。
- `coverage_slice_pred_z*.png`：不同高度的预测二维覆盖切片。
- `coverage_slice_true_z*.png`：不同高度的 FARM 真值二维覆盖切片。
- `coverage_slice_abs_error_z*.png`：不同高度的绝对误差切片。
- `coverage_3d_pred.png`：预测三维无线信号地图。
- `coverage_3d_true.png`：FARM 真值三维无线信号地图。
- `coverage_3d_abs_error.png`：三维绝对误差图。
- `coverage_metrics_summary.png`：NMSE、RMSE、PSNR、SSIM 的可视化摘要。

## 当前模型

`lanc_v1` 使用四个分支：

```text
几何传播分支 + 频率分支 + 天线方向分支 + 建筑环境分支 -> signal_norm
```

`lanc_v2` 仍然保持四分支结构，只增强输入条件：

- 几何分支加入 Tx/Rx 坐标和 Tx 高度。
- 天线分支加入 `yaw_sin/yaw_cos`。
- 不使用第五个 config 分支，不使用残差校正，不使用 tx/config embedding。

建筑物特征当前从 FARM 的 `value == 0` 占据区域近似提取，后续可以替换成更标准的三维建筑物地图或 OSM/LiDAR 特征。

## Scene 12 主动采样实验

在 scene 11 预训练模型基础上，把 scene 12 当作新目标场景：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_active_experiment `
  --source-run runs\farm_lanc_v2_scene11_20260711_195624 `
  --target-farm-root .\12 `
  --target-scene-id 12 `
  --rounds 5 `
  --initial-ratio 0.05 `
  --budget-ratio 0.02 `
  --strategies random proposed
```

快速调试：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_active_experiment `
  --source-run runs\farm_lanc_v2_scene11_20260711_195624 `
  --target-farm-root .\12 `
  --target-scene-id 12 `
  --rounds 1 `
  --initial-ratio 0.02 `
  --budget-ratio 0.01 `
  --quick
```

主动实验输出到：

```text
runs/active_scene11_to_scene12_YYYYMMDD_HHMMSS/
```

关键文件：

- `zero_shot_metrics.csv`：scene 11 模型直接预测 scene 12 的零样本误差。
- `active_round_metrics.csv`：每轮主动采样和微调后的 NMSE、RMSE、PSNR、SSIM。
- `active_selected_rx_points.csv`：每轮选中的 UAV 三维测点。
- `coverage_map_zero_shot.csv`：未微调前的 scene 12 覆盖图。
- `coverage_map_after_random.csv`：随机选点微调后的覆盖图。
- `coverage_map_after_proposed.csv`：主动策略微调后的覆盖图。
- `coverage_3d_zero_shot_pred.png`：scene 12 零样本预测三维图。
- `coverage_3d_after_proposed_abs_error.png`：主动策略微调后的三维误差图。
- `active_curve_rmse.png`、`active_curve_psnr_ssim.png`、`active_curve_ssim.png`：主动采样性能曲线。
- `fine_tuned_BuildingAwareLANCV2_proposed.pt`：主动策略微调后的模型权重。

注意：主动采样前 scene 12 标签不直接给模型看；被选中的 `rx_id` 才从 FARM 真值中揭示标签，用来模拟 UAV 实测。
