# FARM 场景 11 适配版 LANC 最小实验

这个项目现在只保留一个最小闭环：从 FARM `scene_11` 的三维 ARM `.npy` 文件生成点级样本，训练一个 Building-aware Simplified LANC，并输出预测指标、二维高度切片和三维无线信号地图。

## 数据位置

默认读取：

```powershell
.\11\tx_positions_512.csv
.\11\11\freq35\pattern_120\*.npy
```

第一版只使用：

- 频率：`freq35`
- 天线：`pattern_120`
- 高度：`100/110/120/130m`
- 标签：FARM `uint8` 编码值归一化后的 `signal_norm`

注意：当前没有把 FARM 标签强行转换成 dBm，因为完整解码公式还未确认。

## 运行

快速 smoke test：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_experiment --farm-root .\11 --quick --epochs 3
```

单场景完整测试：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_experiment --farm-root .\11 --epochs 50 --stride 16
```

如果想强制 CPU：

```powershell
.\.venv\Scripts\python.exe -m lanc.run_experiment --farm-root .\11 --quick --cpu
```

## 输出

结果会写到：

```text
runs/farm_lanc_scene11_YYYYMMDD_HHMMSS/
```

主要文件：

- `farm_lanc_pairs.csv`：FARM 转换后的点级 LANC 样本。
- `splits.json`：按 `tx_id` 划分的训练/验证/测试集。
- `metrics.csv`：归一化编码误差和 uint8 编码误差。
- `predictions_test.csv`：测试集逐点预测结果。
- `coverage_map_pred.csv`：按接收点聚合后的预测三维无线信号地图。
- `coverage_map_true.csv`：FARM 真值三维无线信号地图。
- `coverage_map_error.csv`：预测图与真值图的误差。
- `coverage_metrics.csv`：完整三维图层面的 NMSE、RMSE、PSNR、SSIM。
- `coverage_metrics_by_height.csv`：按高度层统计的 NMSE、RMSE、PSNR、SSIM。
- `candidate_pool.csv`：后续主动选点候选池。
- `BuildingAwareLANC.pt`：训练好的模型权重。
- `training_curves.png`：训练曲线。
- `coverage_slice_pred_z*.png`：不同高度的预测二维覆盖切片。
- `coverage_slice_true_z*.png`：不同高度的 FARM 真值二维覆盖切片。
- `coverage_slice_abs_error_z*.png`：不同高度的绝对误差切片。
- `coverage_3d_pred.png`：预测三维无线信号地图。
- `coverage_3d_true.png`：FARM 真值三维无线信号地图。
- `coverage_3d_abs_error.png`：三维绝对误差图。
- `coverage_metrics_summary.png`：NMSE、RMSE、PSNR、SSIM 的可视化摘要。

## 当前模型

Building-aware Simplified LANC 使用四个分支：

```text
几何传播分支 + 频率分支 + 天线方向分支 + 建筑环境分支 -> signal_norm
```

建筑物特征当前从 FARM 的 `value == 0` 占据区域近似提取，后续可以替换成更标准的三维建筑物地图或 OSM/LiDAR 特征。
