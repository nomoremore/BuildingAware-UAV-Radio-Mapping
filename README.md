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
  --strategies random uniform_grid weak_only building_only proposed proposed_v2_a proposed_v2_b proposed_v2_c proposed_v2_j
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
  --quick `
  --strategies random uniform_grid weak_only proposed_v2_a
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
- `strategy_comparison_summary.csv`：各策略最终一轮的指标汇总和 RMSE 排名。
- `strategy_comparison_rmse.png`：各策略最终 RMSE 对比图。
- `strategy_comparison_psnr_ssim.png`：各策略最终 PSNR/SSIM 对比图。
- `coverage_3d_zero_shot_pred.png`：scene 12 零样本预测三维图。
- `coverage_3d_after_proposed_abs_error.png`：主动策略微调后的三维误差图。
- `active_curve_rmse.png`、`active_curve_psnr_ssim.png`、`active_curve_ssim.png`：主动采样性能曲线。
- `fine_tuned_BuildingAwareLANCV2_proposed.pt`：主动策略微调后的模型权重。

注意：主动采样前 scene 12 标签不直接给模型看；被选中的 `rx_id` 才从 FARM 真值中揭示标签，用来模拟 UAV 实测。

## 最终论文实验流程

当前建议把代码实验收口为：

```text
scene 11 预训练模型
→ scene 12 主动采样策略开发
→ scene 13 独立泛化验证
→ 多随机种子稳定性实验
→ 不同采样率实验
→ 论文表格和图表汇总
```

主线策略固定为 `proposed_v2_j`。后续不要再根据 scene 13 结果调整模型结构或策略权重，否则 scene 13 就不再是干净的独立验证场景。

### 1. 批量实验 dry-run

先只打印将要运行的命令，确认不会误跑：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_paper_experiments `
  --suite stability `
  --source-run runs\farm_lanc_v2_scene11_20260711_195624 `
  --target-farm-root .\13 `
  --target-scene-id 13 `
  --dry-run
```

### 2. 多随机种子稳定性实验

默认在 scene 13 上运行 `seed=42/2024/3407`，策略为：

```text
random, uniform_grid, weak_only, proposed, proposed_v2_j
```

命令：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_paper_experiments `
  --suite stability `
  --source-run runs\farm_lanc_v2_scene11_20260711_195624 `
  --target-farm-root .\13 `
  --target-scene-id 13 `
  --skip-existing
```

### 3. 不同采样率实验

默认采样预算为：

```text
initial 2% + 每轮 1%
initial 5% + 每轮 2%
initial 10% + 每轮 2%
```

命令：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_paper_experiments `
  --suite budget `
  --source-run runs\farm_lanc_v2_scene11_20260711_195624 `
  --target-farm-root .\13 `
  --target-scene-id 13 `
  --skip-existing
```

批量实验输出在：

```text
runs/paper_batch_YYYYMMDD_HHMMSS/
```

关键文件：

- `batch_manifest.csv`：每个子实验的配置、状态、输出目录和日志位置。
- `logs/*.log`：每个子实验的完整运行日志。
- `active_runs/`：每个子实验生成的主动采样 run 目录。

### 4. 最终论文汇总

把 scene 12 开发结果、scene 13 单次验证结果和批量实验结果汇总成论文表图：

```powershell
.\.venv\Scripts\python.exe -m lanc.summarize_paper_results `
  --scene12-run runs\active_scene11_to_scene12_20260711_223226 `
  --scene13-run runs\active_scene11_to_scene13_20260712_114611 `
  --batch-manifest runs\paper_batch_YYYYMMDD_HHMMSS\batch_manifest.csv `
  --main-strategy proposed_v2_j
```

汇总输出在：

```text
runs/paper_final_summary_YYYYMMDD_HHMMSS/
```

论文可用文件：

- `strategy_mean_std_scene13.csv`：scene 13 多随机种子的策略均值和标准差。
- `budget_curve_summary.csv`：不同采样率下的最终误差。
- `zero_shot_vs_active_final.csv`：zero-shot 与主动采样微调的最终对比。
- `strategy_rmse_mean_std.png`：不同策略 RMSE 均值和标准差图。
- `strategy_psnr_ssim_mean_std.png`：不同策略 PSNR/SSIM 图。
- `sampling_budget_curve.png`：采样率与误差关系图。
- `final_zero_shot_vs_active.png`：zero-shot 与主动采样微调对比图。

默认情况下，`building_only` 不进入主线论文图表；如果需要把它也画进主图，可以在汇总命令中加入：

```powershell
--include-building-only
```

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
