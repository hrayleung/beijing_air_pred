# PRSA 北京多站点空气质量预测：从 EDA、预处理到基线与自定义模型（WG‑DGTM）的复现实验报告（v2.1）

## 摘要

本报告基于 UCI PRSA Beijing Multi‑Site Air Quality（2013–2017）数据集，系统给出：  
(1) 数据特征分析（EDA）；(2) 无泄露预处理流水线（v2.1）；(3) 基线模型套件（B0–B6）训练与评估；(4) 自定义强模型 **WG‑DGTM**（Wind‑Gated Dynamic Graph + TCN）的设计动机、结构细节与实验结果。  

任务定义为：利用过去 7 天（`L=168` 小时）的多站点观测，预测未来 24 小时（`H=24`）12 个站点、6 类污染物（`D=6`）的浓度。所有指标均在 TEST 集上，且严格按 `Y_mask` 对缺失位置进行屏蔽（masked metrics）。  

在当前实验快照中，WG‑DGTM 在 **macro_MAE**（按污染物等权）上达到 **179.533**，优于最佳基线 LightGBM 的 **182.237**；且在 6 个污染物的 MAE 上均优于“对应污染物最佳基线”。详细结果与图表见第 7 节。

---

## 1. 数据集与任务定义

### 1.1 数据来源与文件结构

原始数据位于 `PRSA_Data_20130301-20170228/`，包含 12 个站点 CSV（每站 35,064 行小时级记录；总计 420,768 行）。文件清单与列结构可由 `eda_output/station_summary.csv` 复核。时间范围为 **2013‑03‑01 00:00** 至 **2017‑02‑28 23:00**。

### 1.2 变量体系与单位

污染物（6，预测目标，顺序固定：`[PM2.5, PM10, SO2, NO2, CO, O3]`）：

| 变量 | 单位 | 说明 |
|---|---|---|
| PM2.5 | μg/m³ | 细颗粒物 |
| PM10 | μg/m³ | 可吸入颗粒物 |
| SO2 | μg/m³ | 二氧化硫 |
| NO2 | μg/m³ | 二氧化氮 |
| CO |（原数据单位）| 一氧化碳（量纲显著大于其他污染物） |
| O3 | μg/m³ | 臭氧 |

气象与其他输入变量：
- 气象（5）：TEMP（°C）、PRES（hPa）、DEWP（°C）、RAIN（mm）、WSPM（m/s）。
- 风向：原始为离散方位 `wd`，在预处理中编码为 `wd_sin`/`wd_cos`。
- 时间周期特征：`hour_sin/hour_cos` 与 `month_sin/month_cos`（周期编码）。

### 1.3 固定任务参数

| 参数 | 含义 | 取值 |
|---|---|---:|
| `N` | 站点数 | 12 |
| `L` | 输入窗口 | 168 小时 |
| `H` | 预测步长 | 24 小时 |
| `D` | 预测目标维度 | 6 |
| `F` | 输入特征维度 | 17 |

任务与数据语义以 `processed/metadata.json` 为准。

---

## 2. EDA：数据特征与统计结论（含必要图表）

EDA 由 `eda_beijing_air_quality.py` 实现，输出位于 `eda_output/`（含 `eda_report.html` 与多张 PNG/CSV）。本节总结与建模相关的关键统计结论。

### 2.1 缺失值结构与分布

缺失值分析结果来自 `eda_output/missingness_by_station_feature.csv`。总体规律为：
- 缺失集中在污染物观测，气象变量缺失很少；
- `wd`（风向）存在少量缺失，从而在派生的 `wd_sin/wd_cos` 上对应为缺失。

按“全站点聚合后的缺失率”统计（每特征共 12×35,064=420,768 个位置）：

| 特征 | 缺失率(%) |
|---|---:|
| CO | 4.92 |
| O3 | 3.16 |
| NO2 | 2.88 |
| SO2 | 2.14 |
| PM2.5 | 2.08 |
| PM10 | 1.53 |
| wd | 0.43 |
| TEMP | 0.09 |
| PRES | 0.09 |
| DEWP | 0.10 |
| RAIN | 0.09 |
| WSPM | 0.08 |

缺失可视化（按站点×特征热力图）：

![](eda_output/missingness_analysis.png)

### 2.2 描述统计：长尾分布与尺度差异

`eda_output/stats_overall.csv` 给出全数据（跨站点聚合）统计量。污染物呈现明显长尾，且 CO 的数量级显著大于其他污染物（对损失函数与评估的尺度敏感性提出要求）。

| 变量 | 均值 | 中位数(p50) | p95 | max |
|---|---:|---:|---:|---:|
| PM2.5 | 79.79 | 55 | 242 | 999 |
| PM10 | 104.60 | 82 | 279 | 999 |
| SO2 | 15.83 | 7 | 60 | 500 |
| NO2 | 50.64 | 43 | 117 | 290 |
| CO | 1230.77 | 900 | 3500 | 10000 |
| O3 | 57.37 | 45 | 177 | 1071 |

> 注：上述极大值包含传感器封顶值（如 999/10000），预处理阶段将其识别并置为 NaN（见第 3 节）。

### 2.3 时序规律：季节性与日周期

空气污染具有显著季节性（冬季更高、夏季更低），同时具备日周期结构：PM2.5 往往在夜间/清晨更高，O3 在下午更高（光化学反应更强）。该结构体现为长周期（季节）与短周期（日内）并存的多尺度时间依赖。

![](eda_output/distributions_seasonality.png)

### 2.4 空间异质性：站点差异与相关结构

站点间均值与方差存在差异：例如 PM2.5 均值最低的站点为 Dingling（65.99），最高为 Dongsi（86.19）；CO 均值最低为 Dingling（904.90），最高为 Wanshouxigong（1370.40）（见 `eda_output/stats_by_station.csv`）。这提示空间建模需要同时考虑“站点固有差异”与“跨站点耦合”。

站点对比图：

![](eda_output/station_comparison.png)

站点相关结构（示例：PM2.5 相关矩阵）：

![](eda_output/correlation_matrix.png)

---

## 3. 预处理方法（v2.1，严格防泄露）

预处理脚本为 `preprocessing_pipeline_v2.1.py`，输出至 `processed/`，关键约定见 `processed/README.md` 与日志 `processed/reports/preprocessing_log.txt`。本节按流水线步骤给出可复核的细节说明。

### 3.1 设计原则（Leakage‑safe）

1. **Split‑first**：先按时间边界切分 train/val/test，再做插补、缩放、图构建与窗口生成，避免未来信息泄露。  
2. **TRAIN‑only 统计量**：插补回填值（中位数）与缩放器（RobustScaler）仅由 TRAIN 拟合。  
3. **图构建仅用 TRAIN**：站点图结构由 TRAIN 的相关统计构建。  
4. **窗口不跨边界**：监督窗口生成严格在 split 内完成。

### 3.2 Step‑by‑Step 处理细节与可复核数值

**Step 1：加载原始数据**  
- 每站点 35,064 小时记录，总计 420,768 行（见 `processed/reports/preprocessing_log.txt`）。  
- 站点顺序固定为字母序，并贯穿所有输出（见 `processed/metadata.json` 的 `station_list`）。

**Step 2：封顶值（cap values）识别与处理**  
为避免传感器封顶（例如 `PM2.5=999`、`PM10=999`、`CO=10000`）对统计与训练造成偏置，将这些值转为 `NaN`。本次快照共识别到 61 个封顶值（明细见 `processed/reports/cap_values_report.csv`），例如：

| 站点 | 变量 | cap 值 | 数量 |
|---|---|---:|---:|
| Wanshouxigong | PM2.5 | 999 | 1 |
| Changping / Guanyuan / Shunyi | PM10 | 999 | 1×3 |
| 多站点 | CO | 10000 | 合计 57 |

**Step 3：特征工程（17 维输入特征）**  
输出特征顺序见 `processed/feature_list.json`，包括：
- 风向编码：`wd` 映射为角度后取 `sin/cos`（`wd_sin`/`wd_cos`），以保持方向的连续性；
- 时间周期编码：  
  `hour_sin = sin(2π·hour/24)`，`hour_cos = cos(2π·hour/24)`；  
  `month_sin = sin(2π·month/12)`，`month_cos = cos(2π·month/12)`。

**Step 4：构建原始张量（time × station × feature）**  
构建 `data_tensor` 形状为 `(T=35064, N=12, F=17)`；缺失总量为 **75,909**，占 **1.06%**（见 `processed/reports/preprocessing_log.txt`）。

**Step 5：时间切分（Split‑first）**  
边界与长度（小时）：
- TRAIN：26,304 小时（2013‑03‑01 至 2016‑02‑29）  
- VAL：5,880 小时（2016‑03‑01 至 2016‑10‑31）  
- TEST：2,880 小时（2016‑11‑01 至 2017‑02‑28）

**Step 6：TRAIN‑only 中位数统计（用于插补回填）**  
对 TRAIN 内每个站点×特征计算中位数，得到 `medians` 形状 `(12,17)`（见 `preprocessing_pipeline_v2.1.py` 中 `compute_train_statistics`）。

**Step 7：缺失插补（两条管线）**

- **P1（深度学习/严格因果）**：`causal_impute`  
  - 仅使用 **forward‑fill**（向前填充），不允许 back‑fill；  
  - 对序列开头仍缺失的位置使用 TRAIN 中位数回填；  
  - 同时输出观测掩码 `mask = 1[not NaN]`。  

- **P2（简单基线/非因果）**：`non_causal_impute`  
  - 线性插值（`limit=6`）+ ffill + bfill；  
  - 该路径会使用未来信息（仅用于简单基线的便利性），但 **仍输出 `Y_mask`**（v2.1 FIX C），以保证评估时屏蔽缺失。

**Step 8：缩放（RobustScaler）**
- 对输入 `X` 使用 RobustScaler（基于中位数与 IQR），仅在 TRAIN 拟合后应用到 val/test。  
- 形式化地：`x_scaled = (x - median_train) / IQR_train`，其中 `IQR_train = q75_train - q25_train`。  
- 本次快照 `SCALE_TARGETS=False`，即 `Y` 以原始单位存储（见 `processed/metadata.json` 与 `processed/P1_deep/scaler_params.json`）。  
- 若未来启用 `SCALE_TARGETS=True`，v2.1 约定只在 `Y_mask==1` 的位置缩放目标，避免“缺失 0 值”被错误缩放（见 `preprocessing_pipeline_v2.1.py` 的 `apply_target_scaler`）。

**Step 9：监督窗口生成（L=168, H=24）**
窗口约定（以 `t` 为 origin 时刻、为输入最后一个时间点）：
- `X` 覆盖 `[t-L+1, …, t]`
- `Y` 覆盖 `[t+1, …, t+H]`

输出形状（以 P1 为例）：
- `X`：`(samples, 168, 12, 17)`（scaled）
- `Y`：`(samples, 24, 12, 6)`（raw；缺失位置置 0）
- `Y_mask`：同形状（缺失屏蔽）

样本数（见 `processed/reports/preprocessing_log.txt`）：
- train：26,113  
- val：5,689  
- test：2,689

**Step 10：LightGBM 表格数据（严格因果）**
输出至 `processed/tabular_lgbm/`，特征包含：
- 滞后特征：`lag_k = x(t-k)`，并明确 **不包含 lag0**；
- 滚动统计：`roll{w}_mean/std`（因果滚动窗口）；
- 时间特征：hour/month sin/cos + dayofweek；
- `station`（类别特征）与 `station_id`。

为避免早期样本因 lag/roll 不足导致 NaN 污染，valid_start 使用：  
`min_origin_idx = max(L-1, max_lag, max_roll)`（见日志 FIX B）。  
本快照输出规模（见 `processed/reports/preprocessing_log.txt`）：
- train：313,344 行，317 列  
- val：68,256 行  
- test：32,256 行

**Step 11：站点图构建（TRAIN only）**
以 TRAIN 的 PM2.5 Pearson 相关构建图：
- 生成 full correlation（12×12）与 top‑k（k=4）稀疏图；
- 仅保留正相关边（可复核 `preprocessing_pipeline_v2.1.py` 的 `GRAPH_USE_POSITIVE_ONLY` 逻辑）；
- 对角置 1，并保持对称写入。  
日志显示 top‑k 图非零项为 78（含对角；见 `processed/reports/preprocessing_log.txt`）。

**Step 12：验证测试（必过）**
v2.1 自带一致性检查（见 `processed/reports/preprocessing_log.txt`）：
1. `X` 确实已缩放；  
2. 因果插补无 back‑fill；  
3. `Y` 与 raw 目标严格一致（200 样本抽检）；  
4. 窗口不跨 split；  
5. 站点顺序与邻接矩阵轴一致。

---

## 4. 基线模型套件（B0–B6）：方法与协议

### 4.1 模型列表与数据来源

| 编号 | 模型 | 输入数据 | 备注 |
|---:|---|---|---|
| B0 | Naive Persistence | `processed/P1_deep` | `y(t+h)=y(t)`（对 scaled 输入做逆变换） |
| B1 | Seasonal Naive 24h | `processed/P1_deep` | `y(t+h)=y(t+h-24)`（对 scaled 输入做逆变换） |
| B2 | LightGBM | `processed/tabular_lgbm` | 多输出（24×6）预测 |
| B3 | LSTM（Direct） | `processed/P1_deep` | 端到端多步 |
| B4 | TCN | `processed/P1_deep` | 因果卷积 |
| B5 | STGCN | `processed/P1_deep` + `graphs` | 固定图 |
| B6 | Graph WaveNet | `processed/P1_deep` + `graphs` | 自适应图 |

> 注：预处理同时产出 `processed/P2_simple`（非因果插补，可能引入未来信息），但仓库当前基线入口 `baseline/scripts/run.py` 默认读取 `processed/P1_deep` 以保持严格因果设定。

### 4.2 评估协议（Masked metrics）

基线评估入口：`baseline/evaluation/evaluate.py`。预测、标签与掩码形状必须一致：`(S, 24, 12, 6)`。  
指标均按掩码 `Y_mask` 屏蔽（见 `baseline/evaluation/masked_metrics.py`），并包含“多步输出差异检查”（避免输出在 24 个 horizon 上坍塌为常数）。

指标定义（设预测为 `ŷ`，真实值为 `y`，掩码为 `m∈{0,1}`）：
- `MAE = Σ(|ŷ-y|·m) / Σ(m)`
- `RMSE = sqrt( Σ((ŷ-y)^2·m) / Σ(m) )`
- `sMAPE = 100 · Σ( |ŷ-y|/(|ŷ|+|y|+ε) · m ) / Σ(m)`

macro 指标（`macro_MAE/macro_RMSE/macro_sMAPE`）为上述指标在 6 个污染物维度上的简单平均（等权）。

### 4.3 基线复现命令

```bash
# EDA
python eda_beijing_air_quality.py

# 预处理
python preprocessing_pipeline_v2.1.py

# 运行全部基线（多卡 DataParallel）
unset CUDA_VISIBLE_DEVICES
python -m baseline.scripts.run --model all --config baseline/configs/default.yaml
```

---

## 5. 自定义模型 WG‑DGTM：设计动机、结构细节与训练策略

自定义模型实现位于 `model/`，训练与评估入口分别为：
- 训练：`python -m model.scripts.run_train --config model/configs/wgdgtm.yaml`
- 评估：`python -m model.scripts.run_eval --config model/configs/wgdgtm.yaml --ckpt <path>`

### 5.1 为什么要这样设计（动机与归纳偏置）

WG‑DGTM 的设计围绕 PRSA 任务的三个核心挑战：

1. **时空耦合（Spatial coupling）**  
   多站点污染物并非独立：区域排放与扩散导致站点间强相关（见第 2.4 节相关矩阵）。因此需要图结构或等价的跨站点信息传播机制。

2. **传播方向与时变性（Directed & time‑varying transport）**  
   固定相关图只能刻画“长期平均耦合”，无法表达“随时间变化的传播方向”，而风速/风向是污染传输的重要驱动。故引入 **动态有向图**，并用风场对其进行门控（wind gating），把物理先验注入模型。

3. **长序列 + 直接多步预测（L=168, H=24）**  
   7 天输入窗口要求长感受野且必须保持因果性；多步输出若缺少 horizon 条件，容易出现 **horizon collapse**（不同预测步输出趋同）。因此采用 **扩张因果 TCN** 作为时序主干，并使用 **horizon embedding** 作为解码条件。

此外，污染物尺度高度不均（CO 的量级远大于其他污染物）。若直接用未加权 MAE 训练，梯度将被 CO 主导，导致其他污染物性能受损。因此 WG‑DGTM 使用 **按污染物标准差加权的 masked MAE** 训练，兼顾尺度公平与缺失屏蔽。

上述动机在 `model/DESIGN_NOTE.md` 中给出更集中的阐述。

### 5.2 模型总体结构（精确定义）

输入/输出：
- 输入 `X`：`(B, L=168, N=12, F=17)`（scaled）
- 输出 `Ŷ`：`(B, H=24, N=12, D=6)`（raw）

严格设定（Setting A）：模型仅使用 lookback 窗口内的历史特征，不引入未来气象/协变量；因此可在相同信息集下与基线公平对比（见 `model/README.md`）。

WG‑DGTM 由五个模块组成（`model/models/wgdgtm.py`）：

1. **特征编码器**（`model/modules/feature_encoder.py`）  
   对每个时间步与站点做逐点编码：  
   `h(t,i) = Dropout(GELU(LayerNorm(Wx·x(t,i))))`

2. **风门控动态有向图构建**（`model/modules/dynamic_graph.py`）  
   在每个时间步构建邻接矩阵 `A_t`，由三部分融合：
   - 静态先验图 `A_static`：由 TRAIN 的相关 top‑k 图得到（预处理输出 `processed/graphs/adjacency_corr_topk.npy`）；
   - 可学习图 `A_learn`：由站点嵌入 `E` 生成：`A_learn = softmax(E E^T)`；
   - 动态图 `A_dyn(t)`：由节点状态注意力生成：  
     `A_dyn(t) = softmax( (Wq h(t)) (Wk h(t))^T / sqrt(d) + λ·g(t) )`  
     其中 `g(t,i)` 为风门控标量（见 5.3）。

	   最终融合（并保证权重正、便于优化）：

	   `A_t = RowNormalize( softplus(α)·A_static + softplus(β)·A_learn + softplus(γ)·A_dyn(t) + I )`

	   设计理由（Why this fusion）：`A_static` 提供稳健空间先验并降低过拟合风险；`A_learn` 弥补固定相关图无法表达的长期残差关系；`A_dyn(t)` 使连边随状态与风场自适应变化以刻画时变传播；`softplus` 约束融合权重为正以提升可解释性与训练稳定性；加入 `I` 保证自环；`RowNormalize` 将每行视为“从源节点出发的权重分布”，可稳定消息传递并便于解释为有向扩散强度。

3. **空间消息传递**（`model/modules/spatial_layer.py`）  
   在每个时间步进行有向传播：  
   `z(t) = (A_t · h(t)) · W_s`

4. **时间主干：扩张因果 TCN**（`model/modules/tcn.py`）  
   将 `z(t,i)` 视为每站点序列，使用多层 dilation=2^l 的因果卷积块获得长感受野；取最后时刻表征 `r_last(i)` 用于解码。

5. **horizon‑aware 解码器**（`model/modules/horizon_decoder.py`）  
   为每个预测步 `h` 引入可学习嵌入 `e_h`，以条件化输出并避免多步坍塌：  
   `Ŷ(t+h,i) = MLP([r_last(i), e_h])`

可选升级：
- **Residual forecasting**（`model/modules/residual_baseline.py`）：输出预测残差并叠加持久性基线 `y(t)`；
- **Multi‑head decoder**（`model/modules/multihead_decoder.py`）：每个污染物一个小 head，降低跨污染物负迁移。

### 5.3 风门控（wind gating）的物理含义与实现细节

预处理输入为 scaled 值，但风门控需要物理意义一致的量纲。WG‑DGTM 在前向中仅对风相关通道进行逆缩放（`model/models/wgdgtm.py::_wind_uvs`），并计算：
- `u = WSPM · wd_cos`，`v = WSPM · wd_sin`
- `g(t,i) = sigmoid(MLP([u, v, WSPM])) ∈ (0,1)`

随后将 `λ·g(t,i)` 作为“从节点 i 出发的注意力偏置”加入 dynamic attention logits，实现方向性与强度的时变调制（详见 `model/DESIGN_NOTE.md`）。

### 5.4 损失函数、权重与优化设置

训练损失：**std‑weighted masked MAE**（`model/losses/masked_losses.py`）：

- 在 TRAIN 的观测位置（`Y_mask==1`）上计算每污染物标准差 `std[d]`；
- 权重定义：`w[d] = 1 / (std[d] + eps)`；
- 损失：
  `Loss = sum( |Ŷ - Y| · Y_mask · w ) / sum(Y_mask)`

本次快照的 `std/weights` 输出于 `model/results/metrics/target_std_weights.json`（示例：CO 的 `std≈1125.53`，对应权重显著更小，从而避免训练被 CO 主导）。

训练器：`model/training/trainer.py`（AdamW、梯度裁剪、早停）。默认配置见 `model/configs/wgdgtm.yaml`：
- `epochs=50`，`batch_size=64`，`lr=1e-3`，`weight_decay=1e-4`
- `grad_clip=5.0`，早停 `patience=8`
- 多 GPU：可见 GPU>1 时启用 `torch.nn.DataParallel`（`training.use_data_parallel=true`）

### 5.5 现象驱动的两项改动：Residual forecasting 与 Multi‑head 输出

WG‑DGTM 提供两项可选改动（`model/modules/residual_baseline.py`、`model/modules/multihead_decoder.py`）。其核心目的并非“为了更复杂”，而是针对已观测到的**现象与数据特性**补足短板、提高可解释性与训练稳定性。

#### 5.5.1 Residual forecasting：相对 persistence 学习预测残差

**动机（来自基线现象）**：基线结果表明短期（1–6 小时）污染物具有显著惯性，简单持久性在 `h=1` 上往往非常强。例如（TEST，masked）：
- PM2.5：`naive_persistence` 的 `MAE_h1=13.46`，优于 LightGBM（20.94）与 TCN（33.92）。
- CO：`naive_persistence` 的 `MAE_h1=318.09`，亦优于 LightGBM（453.58）与 TCN（840.67）。

因此，将模型直接拟合 `y(t+h)` 等价于同时学习：
(i) 容易的“惯性部分”（接近 `y(t)`）；(ii) 更难的“变化部分”（受风场输送、扩散、化学反应等影响）。Residual forecasting 将任务重参数化为：

`ŷ(t+h) = y_base(t+h) + Δ̂(t+h)`，其中 `y_base(t+h)=y(t)`（persistence）。

**预期收益**：残差 `Δ(t+h)` 在短期通常幅度更小、分布更平稳，有助于降低输出尺度、改善梯度条件数并提升优化稳定性；对量纲大且惯性强的 CO 尤其有利。

**典型权衡**：若训练目标对 24 个 horizon 等权，残差学习会鼓励模型“贴近基线”，从而显著改善 `h=1/h=6`，但可能牺牲远期（如 `h=24`）对趋势/转折的刻画能力。本报告第 7.2.1 节中 residual+multihead 的“`h1` 大幅改善、`h24` 变差”正对应这一权衡。

#### 5.5.2 Multi‑head output：每个污染物一个小输出头

**动机（来自任务异质性）**：6 个污染物构成典型多任务学习问题，但其生成机理与统计尺度差异显著（例如 CO 量纲大且惯性强；O3 具有显著日周期并受光化学过程影响；PM2.5/PM10 更受输送与沉降影响）。共享输出头（shared head）在这种异质多任务下容易出现负迁移：某些污染物的梯度会干扰另一些污染物的拟合与校准。

Multi‑head 的做法是“共享表征、分离映射”：
- 共享骨干（动态图 + TCN）学习共同的时空结构；
- 输出层对每个污染物使用独立的小 head（参数量小但更专门），降低跨污染物干扰并改善末端校准。

从结果侧亦可观察到这种异质性：基线中 PM2.5/PM10/SO2 的最优模型更偏向 TCN，而 NO2/CO/O3 更偏向 LightGBM（见第 7.2.2 节表格），提示“单一共享映射”并非各目标的共同最优。

#### 5.5.3 与实验结果的对应关系与改进方向

在当前快照中，residual+multihead 显著改善短期宏平均误差（`macro_MAE@h1: 132.35 → 67.73`），但远期劣化（`macro_MAE@h24: 216.65 → 237.72`）。该现象与残差学习在 horizon 等权目标下偏向 persistence 的理论预期一致。可行的改进方向包括：
- **horizon‑weighted loss / 分段权重**：提高远期 horizon 的损失权重，以平衡短期与远期性能；
- **残差基线改造**：用季节性基线或更强的基线替代纯 persistence，以减少“贴基线”带来的远期偏置；
- **head 容量与正则**：调整 multi‑head 的容量、dropout 与权重衰减，缓解过拟合与长步泛化退化。

#### 5.5.4 学术化表述（可直接引用，中英对照）

**中文（可直接写入论文/报告）**：基于基线实验所揭示的短期强惯性特征（例如 PM2.5 与 CO 在 `h=1` 下 persistence 已取得最优或近最优 MAE），我们将多步预测任务重新参数化为相对 persistence 的残差预测，即 \(\hat y_{t+h}=y_t+\widehat{\Delta}_{t+h}\)。该设定使模型聚焦于幅度更小、更平稳的变化项 \(\Delta_{t+h}\)，从而改善短期优化条件并提升 `h=1–6` 的预测精度；同时，鉴于 6 类污染物在生成机理、尺度与噪声结构上的显著异质性，我们采用“共享骨干 + 分污染物输出头”的 multi‑head 解码器以缓解负迁移并提升末端校准。实验上，该 residual+multihead 变体在短步宏平均误差上显著获益（`macro_MAE@h1: 132.35 → 67.73`），但也呈现远期误差劣化（`macro_MAE@h24: 216.65 → 237.72`）这一典型权衡，提示后续可通过 horizon‑weighted 损失或更强的残差基线来平衡中长步预测性能。

**English (ready-to-use paragraph)**: Motivated by the strong short-term persistence observed in the baselines (e.g., persistence achieves the best or near-best MAE at `h=1` for PM2.5 and CO), we reparameterize multi-horizon forecasting as residual learning relative to persistence, \(\hat y_{t+h}=y_t+\widehat{\Delta}_{t+h}\). This formulation focuses the model on smaller and more stationary deviations \(\Delta_{t+h}\), stabilizing optimization and improving short-horizon accuracy (`h=1–6`). Moreover, given the pronounced heterogeneity across the six pollutants in terms of mechanisms, scales, and noise characteristics, we adopt a shared spatio-temporal backbone with pollutant-specific output heads (multi-head decoder) to mitigate negative transfer and improve calibration. Empirically, the residual+multihead variant yields a marked gain at short horizons (`macro_MAE@h1: 132.35 → 67.73`) but degrades long-horizon performance (`macro_MAE@h24: 216.65 → 237.72`), suggesting horizon-weighted objectives or stronger residual baselines as future remedies.

---

## 6. 实验流程（Procedures）与环境

### 6.1 环境（dl conda）

本报告对应运行环境（来自 `conda run -n dl`）：
- Python 3.11.14，PyTorch 2.8.0+cu128（CUDA 12.8），LightGBM 4.6.0
- GPU：4× NVIDIA GeForce RTX 4090 D

### 6.2 一键复现命令

```bash
# 1) EDA
python eda_beijing_air_quality.py

# 2) 预处理 v2.1（会覆盖 processed/）
python preprocessing_pipeline_v2.1.py

# 3) 基线（B0–B6）
unset CUDA_VISIBLE_DEVICES
python -m baseline.scripts.run --model all --config baseline/configs/default.yaml

# 4) 自定义模型 WG-DGTM
python -m model.scripts.run_train --config model/configs/wgdgtm.yaml
python -m model.scripts.run_eval  --config model/configs/wgdgtm.yaml --ckpt model/results/checkpoints/best.pt
```

> 说明：当前仓库快照中，WG‑DGTM 的输出位于 `model/results/`（而非 `model/results/wgdgtm/` 子目录）；升级版 residual+multihead 的输出位于 `model/results/wgdgtm_residual_multihead/`。

---

## 7. 结果与分析（TEST 集，Masked 指标 + 图表）

### 7.1 基线结果（B0–B6）

基线输出目录：`baseline/results/`。核心汇总表：
- `baseline/results/model_comparison.csv`（overall 指标 + 关键 horizon MAE）
- `baseline/results/metrics_overall.csv`（含 macro 平均）
- `baseline/results/metrics_per_pollutant.csv`（按污染物）

**Overall 指标对比（来自 `baseline/results/model_comparison.csv`）：**

| 模型 | MAE ↓ | RMSE ↓ | sMAPE(%) ↓ | MAE@h1 ↓ | MAE@h6 ↓ | MAE@h12 ↓ | MAE@h24 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| lightgbm | 180.994 | 550.906 | 30.18 | 87.205 | 155.131 | 188.603 | 225.830 |
| naive_persistence | 195.757 | 619.812 | 32.09 | 60.819 | 162.775 | 208.991 | 255.227 |
| tcn | 197.514 | 641.416 | 31.73 | 158.952 | 173.278 | 197.184 | 235.474 |
| lstm | 227.942 | 767.073 | 36.51 | 209.124 | 216.625 | 228.896 | 245.183 |
| gwnet | 249.602 | 788.987 | 39.48 | 246.335 | 247.532 | 249.442 | 252.665 |
| seasonal_naive_24h | 254.928 | 747.602 | 39.49 | 254.494 | 254.583 | 254.923 | 255.227 |
| stgcn | 333.515 | 1030.602 | 46.92 | 333.646 | 333.867 | 333.394 | 333.377 |

由于 CO 量纲显著大于其他污染物，overall MAE/RMSE 容易被 CO 误差主导。为衡量“各污染物等权”的整体性能，可使用 macro 指标（按污染物平均；见 `baseline/results/metrics_overall.csv`）：

| 模型 | macro_MAE ↓ | macro_RMSE ↓ | macro_sMAPE(%) ↓ |
|---|---:|---:|---:|
| lightgbm | 182.237 | 266.762 | 30.174 |
| naive_persistence | 197.094 | 302.169 | 32.090 |
| tcn | 198.905 | 303.803 | 31.734 |
| lstm | 229.522 | 365.927 | 36.512 |
| gwnet | 251.325 | 376.730 | 39.474 |
| seasonal_naive_24h | 256.660 | 364.831 | 39.481 |
| stgcn | 335.951 | 477.210 | 46.974 |

**MAE‑H 曲线（所有基线）：**  
![](baseline/results/plots/all_models_mae_vs_horizon.png)

**RMSE‑H 曲线（所有基线）：**  
![](baseline/results/plots/all_models_rmse_vs_horizon.png)

**训练曲线与样例预测（基线）：**  
![](baseline/results/plots/lstm_loss_curve.png)
![](baseline/results/plots/tcn_loss_curve.png)
![](baseline/results/plots/stgcn_loss_curve.png)
![](baseline/results/plots/gwnet_loss_curve.png)
![](baseline/results/plots/lightgbm_sample_predictions.png)
![](baseline/results/plots/seasonal_naive_sanity.png)

### 7.2 自定义模型 WG‑DGTM：TEST 指标与可视化

#### 7.2.1 Macro 指标（按污染物等权）

WG‑DGTM 评估输出目录：`model/results/**/metrics/`。macro 汇总见：
- 原始 WG‑DGTM：`model/results/metrics/macro_avg_metrics.csv`
- residual+multihead：`model/results/wgdgtm_residual_multihead/metrics/macro_avg_metrics.csv`

| run | macro_MAE ↓ | macro_RMSE ↓ | macro_sMAPE(%) ↓ |
|---|---:|---:|---:|
| WG‑DGTM | 179.533 | 269.082 | 29.422 |
| WG‑DGTM (residual+multihead) | 184.897 | 278.924 | 30.026 |

为解释不同 horizon 行为，可将关键 horizon 的 per‑pollutant `MAE_h*` 取均值得到 `macro_MAE_h*`：

| 模型 | macro_MAE ↓ | macro_MAE@h1 ↓ | macro_MAE@h6 ↓ | macro_MAE@h12 ↓ | macro_MAE@h24 ↓ |
|---|---:|---:|---:|---:|---:|
| lightgbm（最佳基线） | 182.237 | 87.819 | 156.200 | 189.895 | 227.379 |
| WG‑DGTM | 179.533 | 132.354 | 155.645 | 182.838 | 216.645 |
| residual+multihead | 184.897 | 67.734 | 155.974 | 193.889 | 237.722 |

> 该对比提示：residual+multihead 显著改善短期（h=1）宏平均误差，但在长步（h=24）上劣化；原始 WG‑DGTM 则在中长步更优，从而整体 macro_MAE 更低。

#### 7.2.2 按污染物指标与对比（MAE）

原始 WG‑DGTM 的 per‑pollutant 指标见 `model/results/metrics/metrics_per_pollutant.csv`：

| 污染物 | WG‑DGTM MAE ↓ |
|---|---:|
| PM2.5 | 52.279 |
| PM10 | 60.752 |
| SO2 | 9.187 |
| NO2 | 23.023 |
| CO | 916.276 |
| O3 | 15.678 |

与 LightGBM（最佳 overall 基线）相比，WG‑DGTM 在全部污染物上均有改善（单位为原始单位）：

| 污染物 | LightGBM | WG‑DGTM | Δ（WG‑DGTM−LightGBM） |
|---|---:|---:|---:|
| PM2.5 | 56.505 | 52.279 | -4.226 |
| PM10 | 65.302 | 60.752 | -4.550 |
| SO2 | 9.411 | 9.187 | -0.224 |
| NO2 | 23.949 | 23.023 | -0.926 |
| CO | 922.082 | 916.276 | -5.806 |
| O3 | 16.171 | 15.678 | -0.493 |

进一步地，以“每个污染物的最优基线”作为参照（PM2.5/PM10/SO2 的最佳基线为 TCN，其余为 LightGBM），WG‑DGTM 仍实现一致的小幅提升：

| 污染物 | 最优基线 | 最优基线 MAE | WG‑DGTM MAE | Δ（WG‑DGTM−最优基线） |
|---|---|---:|---:|---:|
| PM2.5 | tcn | 52.795 | 52.279 | -0.515 |
| PM10 | tcn | 61.877 | 60.752 | -1.124 |
| SO2 | tcn | 9.305 | 9.187 | -0.118 |
| NO2 | lightgbm | 23.949 | 23.023 | -0.926 |
| CO | lightgbm | 922.082 | 916.276 | -5.806 |
| O3 | lightgbm | 16.171 | 15.678 | -0.493 |

三模型对比（LightGBM / TCN / WG‑DGTM，按污染物 MAE）：

![](model/results/plots/mae_compare_baselines.png)

#### 7.2.3 必要图表：误差曲线、样例预测、训练曲线

原始 WG‑DGTM：

![](model/results/plots/error_vs_horizon.png)
![](model/results/plots/prediction_vs_truth.png)
![](model/results/plots/train_history.png)

residual+multihead：

![](model/results/wgdgtm_residual_multihead/plots/error_vs_horizon.png)
![](model/results/wgdgtm_residual_multihead/plots/prediction_vs_truth.png)
![](model/results/wgdgtm_residual_multihead/plots/train_history.png)

#### 7.2.4 验证集训练过程（用于可复核的“结果来源”说明）

训练历史保存在：
- `model/results/logs/train_history.csv`
- `model/results/wgdgtm_residual_multihead/logs/train_history.csv`

其最优验证集 macro_MAE（越低越好）为：
- WG‑DGTM：70.979（epoch 22）
- residual+multihead：69.004（epoch 8）

> 注：验证集最优并不保证测试集最优；本快照中 residual+multihead 在 val 上更优，但在 test 的中长步误差更大（见 7.2.1）。

---

## 8. 结论

1. 数据层面：污染物缺失率显著高于气象变量；分布具长尾且 CO 量纲远大于其他污染物；存在强季节性与日周期；站点间既有异质性也有显著耦合。  
2. 方法层面：v2.1 预处理通过 split‑first、TRAIN‑only 统计量与严格窗口约定实现无泄露；`Y_mask` 的引入保证缺失位置不会污染训练与评估。  
3. 结果层面：基线中 LightGBM 在 overall MAE 上最优；自定义 WG‑DGTM 在 macro_MAE 上进一步提升，并在 6 个污染物 MAE 上均优于对应最佳基线，体现出“风门控动态有向图 + TCN”的有效性。residual+multihead 可显著改善短期，但在本快照中牺牲了长步预测性能。
