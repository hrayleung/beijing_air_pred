# 北京多站点空气质量预测

基于 UCI PRSA 北京数据集的多步空气质量预测。项目包含 EDA、可复现的预处理流程、基线模型，以及更强的时空模型 WG-DGTM。任务参数固定：回溯窗 168 小时、预测窗 24 小时、12 个站点、6 个污染物目标。

## 仓库结构
- 原始数据：`PRSA_Data_20130301-20170228/`（12 站点 CSV）。
- 核心脚本：`eda_beijing_air_quality.py`（EDA）、`preprocessing_pipeline_v2.1.py`（预处理）。
- 基线包 `baseline/`：`configs/`、`data/`、`models/`、`training/`、`evaluation/`、`scripts/`，输出 `baseline/results/`。
- 强模型包 `model/`：`configs/`、`data/`、`modules/`、`models/`、`training/`、`evaluation/`、`scripts/`，输出 `model/results/`。
- 生成物：`processed/`、`processed_scaled/`、`eda_output/`、`baseline/results/`、`model/results/`（视为构建产物）。
- 文档：`DOCUMENTATION.md`（EDA/预处理）、`processed/README.md`（数据格式）、`PROJECT_REPORT*.md`、`CHANGELOG_v2*.md`。

目录速览：
```
.
├── eda_beijing_air_quality.py
├── preprocessing_pipeline_v2.1.py
├── baseline/
│   ├── configs/           # 默认与模型特定配置
│   ├── data/              # NPZ/CSV 加载、图构造
│   ├── models/            # naive/seasonal/lgbm/lstm/tcn/stgcn/gwnet
│   ├── training/          # PyTorch 与 LightGBM 训练器
│   ├── evaluation/        # 掩码指标与可视化
│   └── scripts/           # run / debug_checks 入口
├── model/
│   ├── configs/           # wgdgtm 配置
│   ├── data/              # 数据集与加载器
│   ├── modules/           # 图构建、注意力、TCN 等组件
│   ├── models/            # WG-DGTM 组装
│   ├── training/          # 训练循环、调度、早停
│   ├── evaluation/        # 指标与推理脚本
│   └── scripts/           # run_train / run_eval 入口
├── processed/
│   ├── P1_deep/           # 深度模型 NPZ + scaler
│   ├── P2_simple/         # 朴素基线 NPZ
│   ├── tabular_lgbm/      # LightGBM CSV
│   ├── graphs/            # 相关性邻接矩阵
│   ├── metadata.json      # 处理配置与顺序
│   └── README.md          # 详细字段说明
└── DOCUMENTATION.md
```

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
- 任务参数固定：回溯 `168`、预测 `24`、站点 `12`、目标 `6`，站点顺序记录在 `processed/metadata.json`。
- 输入经 RobustScaler 归一化，目标保持原始量纲；缩放器位于 `processed/P1_deep/scaler.pkl`。
- 掩码：`Y` 缺失位置为 0，必须结合 `Y_mask` 计算损失与指标（参考 `baseline/evaluation/masked_metrics.py`）。
- 输出结构：`processed/P1_deep/train|val|test.npz` 含 `X`、`Y`、`X_mask`、`Y_mask`；`processed/tabular_lgbm/*.csv` 为全局 LightGBM 特征；`processed/graphs/*.npy` 为相关性邻接矩阵。
- 更多特征工程、划分与字段说明见 `DOCUMENTATION.md` 与 `processed/README.md`。

## 输出与校验
- 基线指标：`baseline/results/metrics_summary.csv`、`baseline/results/metrics_per_pollutant.csv`。
- 强模型产物：`model/results/` 下的 checkpoint、日志与图表。
- 如修改预处理，需重新生成 `processed/` 并核对 `processed/metadata.json` 以及掩码指标的一致性。
