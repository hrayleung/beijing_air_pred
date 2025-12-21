# 北京多站点空气质量预测

基于 UCI PRSA 北京数据集的多步空气质量预测。项目包含 EDA、可复现的预处理流程、基线模型，以及更强的时空模型 WG-DGTM。任务参数固定：回溯窗 168 小时、预测窗 24 小时、12 个站点、6 个污染物目标。

## 仓库结构
- `PRSA_Data_20130301-20170228/`：原始 12 个监测站 CSV。
- `eda_beijing_air_quality.py`：EDA，输出到 `eda_output/`。
- `preprocessing_pipeline_v2.1.py`：主预处理流水线，生成 `processed/`（详见 `processed/README.md`）。
- `baseline/`：基线模型包（配置、数据加载、模型、训练、评估），输出到 `baseline/results/`。
- `model/`：WG-DGTM 强模型包，独立配置与脚本，输出到 `model/results/`。
- 生成物（视为构建产物）：`processed/`、`processed_scaled/`、`eda_output/`、`baseline/results/`、`model/results/`。
- 其他文档：`DOCUMENTATION.md`（EDA/预处理细节）、`PROJECT_REPORT*.md`（报告）、`CHANGELOG_v2*.md`。

## 环境准备
- 建议 Python 3.9+。
- 安装依赖（基线与强模型共用）：
  ```bash
  python -m pip install -r baseline/requirements.txt
  ```

## 运行指南
- **EDA**：生成描述性报告与图表（输出在 `eda_output/`）。
  ```bash
  python eda_beijing_air_quality.py
  ```

- **预处理**：构建可直接训练的数据集（覆盖 `processed/` 中现有内容）。
  ```bash
  python preprocessing_pipeline_v2.1.py
  ```
  主要产物：深度模型用的 NPZ (`processed/P1_deep`)、LightGBM 表格 CSV (`processed/tabular_lgbm`)、图邻接矩阵 (`processed/graphs`)，配置记录于 `processed/metadata.json`。

- **基线模型**：训练/评估基线，结果存于 `baseline/results/`。
  ```bash
  # 朴素持久性基线
  python -m baseline.scripts.run --model naive --config baseline/configs/default.yaml

  # LSTM 基线
  python -m baseline.scripts.run --model lstm --config baseline/configs/lstm.yaml

  # 调试检查（轻量自检）
  python -m baseline.scripts.debug_checks --config baseline/configs/default.yaml
  ```

- **WG-DGTM 强模型**：训练与评估动态图 + TCN 模型。
  ```bash
  # 训练
  python -m model.scripts.run_train --config model/configs/wgdgtm.yaml

  # 评估指定 checkpoint
  python -m model.scripts.run_eval --config model/configs/wgdgtm.yaml --ckpt model/results/checkpoints/best.pt
  ```

## 数据说明
- 任务参数固定：回溯 `168`、预测 `24`、站点 `12`、目标 `6`，站点顺序已在 `processed/metadata.json` 固化。
- 目标保持原始量纲；输入经 RobustScaler 归一化（缩放器位于 `processed/P1_deep/scaler.pkl`）。
- 损失与指标需使用 `Y_mask` 忽略缺失目标（参见 `baseline/evaluation/masked_metrics.py`）。
- 特征工程、数据划分与文件格式细节见 `DOCUMENTATION.md` 与 `processed/README.md`。

## 输出与校验
- 基线指标：`baseline/results/metrics_summary.csv`、`baseline/results/metrics_per_pollutant.csv`。
- 强模型产物：`model/results/` 下的 checkpoint、日志与图表。
- 如修改预处理，需重新生成 `processed/` 并核对 `processed/metadata.json` 以及掩码指标的一致性。
