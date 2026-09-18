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

## 主动采样策略

主动采样以 `rx_id` 为一个 UAV 三维测点。选中某个 `rx_id` 后，该位置下所有 TX/Yaw 配置的 FARM 标签会一起揭示，用来模拟一次无人机空间测量。每次评估只使用尚未采样的 `rx_id`，避免把已测点自身的准确率算进提升。

当前主线策略是 `proposed_v3`：

```text
每轮预算 = 覆盖配额 + 建筑利用配额
覆盖配额：4 个高度层轮询 + 8x8 空间分层 + 空间分散
建筑利用配额：0.50 building + 0.20 diversity
             + 0.15 prediction_spread + 0.15 gradient
```

`rho` 控制覆盖配额比例，预定义取值为 `0.4/0.5/0.6`。scene 12 只用于开发阶段选择 `rho`；冻结策略后，scene 13 只做独立验证，不再根据 scene 13 结果修改 `rho` 或评分权重。

`prediction_spread_norm` 是同一接收点在多个 TX/Yaw 配置间的预测标准差，表示“多配置预测分歧”，不是严格的贝叶斯不确定性。

### 快速 smoke test

先确认 V3 的选点、微调、指标和图像输出链路正常：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_active_experiment `
  --source-run runs\farm_lanc_v2_scene11_20260711_195624 `
  --target-farm-root .\12 `
  --target-scene-id 12 `
  --rounds 1 `
  --initial-ratio 0.02 `
  --budget-ratio 0.01 `
  --quick `
  --strategies random building_only proposed_v2_j proposed_v3
```

主动实验输出到：

```text
runs/active_scene11_to_scene12_YYYYMMDD_HHMMSS/
```

关键文件：

- `zero_shot_metrics.csv`：预训练模型直接预测目标场景的零样本误差。
- `active_round_metrics.csv`：每轮主动采样和微调后的 NMSE、RMSE、PSNR、SSIM。
- `active_selected_rx_points.csv`：每轮选中的 UAV 三维测点。
- `proposed_v3_selection_diagnostics.csv`：V3 每轮的预算、覆盖配额、高度分布、空间分区覆盖数和评分统计。
- `candidate_pool_zero_shot.csv`：包含 `prediction_spread_norm`、建筑特征和选点评分的候选池。
- `strategy_comparison_summary.csv`：各策略最终一轮的指标汇总和 RMSE 排名。
- `coverage_3d_after_proposed_v3_pred.png`：V3 微调后的三维预测图。
- `coverage_3d_after_proposed_v3_abs_error.png`：V3 微调后的三维绝对误差图。
- `fine_tuned_BuildingAwareLANCV2_proposed_v3.pt`：V3 微调后的模型权重。

## 最终论文实验流程

完整实验主线为：

```text
scene 11 预训练模型
→ scene 12 六随机种子策略开发并选择 rho
→ 冻结 proposed_v3
→ scene 13 六随机种子独立验证
→ scene 13 三个采样预算验证
→ 论文表格和图表汇总
```

### 1. Scene 12 策略开发

先用 dry-run 检查六个随机种子的命令：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_paper_experiments `
  --suite stability `
  --strategy-mode development `
  --source-run runs\farm_lanc_v2_scene11_20260711_195624 `
  --target-farm-root .\12 `
  --target-scene-id 12 `
  --dry-run
```

正式运行 `seed=42/2024/3407/1234/5678/9012`。development 模式同时比较全部 baseline，以及 `proposed_v3_rho04/rho05/rho06`：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_paper_experiments `
  --suite stability `
  --strategy-mode development `
  --source-run runs\farm_lanc_v2_scene11_20260711_195624 `
  --target-farm-root .\12 `
  --target-scene-id 12 `
  --skip-existing
```

从批量结果中按六个种子的平均 encoded RMSE 选择 `rho`。若前两名 RMSE 差值小于 `0.01`，选择 PSNR 和 SSIM 退化更小者。选中后只保留一个冻结策略名 `proposed_v3`，并停止在 scene 12 上继续调权。

### 2. Scene 13 独立验证

最终 stability 模式使用六个随机种子，并包含 `random`、`uniform_grid`、`weak_only`、`building_only`、`proposed`、`proposed_v2_j`、`proposed_v3`：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_paper_experiments `
  --suite stability `
  --strategy-mode final `
  --source-run runs\farm_lanc_v2_scene11_20260711_195624 `
  --target-farm-root .\13 `
  --target-scene-id 13 `
  --skip-existing
```

### 3. Scene 13 采样预算实验

三种预算为 `initial 2% + 每轮 1%`、`initial 5% + 每轮 2%`、`initial 10% + 每轮 2%`，比较 `random`、`building_only`、`proposed_v3`：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_paper_experiments `
  --suite budget `
  --strategy-mode final `
  --source-run runs\farm_lanc_v2_scene11_20260711_195624 `
  --target-farm-root .\13 `
  --target-scene-id 13 `
  --skip-existing
```

批量实验输出到：

```text
runs/paper_batch_YYYYMMDD_HHMMSS/
```

关键文件：

- `batch_manifest.csv`：每个子实验的 seed、预算、策略、状态、输出目录和日志位置。
- `logs/*.log`：每个子实验的完整运行日志。
- `active_runs/`：每个子实验生成的主动采样 run 目录。

### 4. 最终论文汇总

把 scene 12 开发结果、scene 13 验证结果和批量实验汇总成论文表图：

```powershell
.\.venv\Scripts\python.exe -m lanc.summarize_paper_results `
  --scene12-run runs\active_scene11_to_scene12_YYYYMMDD_HHMMSS `
  --scene13-run runs\active_scene11_to_scene13_YYYYMMDD_HHMMSS `
  --batch-manifest runs\paper_batch_YYYYMMDD_HHMMSS\batch_manifest.csv `
  --main-strategy proposed_v3
```

汇总输出在：

```text
runs/paper_final_summary_YYYYMMDD_HHMMSS/
```

论文可用文件：

- `strategy_mean_std_scene13.csv`：scene 13 各策略的均值、标准差和逐 seed RMSE 胜率。
- `budget_curve_summary.csv`：三个采样预算下的最终误差。
- `zero_shot_vs_active_final.csv`：zero-shot 与主动采样微调的最终对比。
- `strategy_rmse_mean_std.png`：各策略 RMSE 均值和标准差图。
- `strategy_psnr_ssim_mean_std.png`：各策略 PSNR/SSIM 对比图。
- `sampling_budget_curve.png`：采样率与误差关系图。
- `final_zero_shot_vs_active.png`：zero-shot 与主动采样微调对比图。

`building_only` 必须保留在主线比较和论文主图中。验收关注六种子平均性能、RMSE 稳定性、PSNR/SSIM 退化幅度和预算曲线，不声称随机采样下每个 seed、每个预算都必然获胜。

## 已归档实验结果

适合 GitHub 分享的精选实验记录位于：

```text
experiment_records/
```

其中包含 scene 11 预训练、scene 12 策略开发、scene 13 独立验证和最终论文汇总的 CSV、PNG、模型权重与实验结论。完整 `runs/` 和原始 FARM 数据不进入 Git。

按时间顺序整理、适合向导师汇报的详细说明位于：

```text
docs/实验进展汇报.md
```
