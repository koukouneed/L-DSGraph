# L-DSGraph: Line-level Dual-Source Graph Neural Network for Software Fault Localization

> **L-DSGraph：面向语句级软件缺陷定位的双源图神经网络**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

[中文版本](#chinese) | [English Version](#english)

---

<span id="english"></span>

## English Version

### Table of Contents

1. [Overview](#overview)
2. [Environment Requirements](#environment-requirements)
3. [Installation Guide](#installation-guide)
4. [Core Features](#core-features)
5. [Project Structure](#project-structure)
6. [Usage Examples](#usage-examples)
7. [Experiment Reproduction](#experiment-reproduction)
8. [Configuration Parameters](#configuration-parameters)
9. [Evaluation Metrics](#evaluation-metrics)
10. [FAQ & Troubleshooting](#faq--troubleshooting)
11. [Citation](#citation)

---

### Overview

**L-DSGraph** (Line-level Dual-Source Graph Neural Network) is a graph neural network-based framework for software fault localization (FL) that operates at the statement/line level. The model jointly leverages three complementary information sources to pinpoint defective code lines:

- **SBFL Features** (Spectrum-Based Fault Localization): Coverage matrices and aggregated ranking scores from multiple SBFL formulas (e.g., Ochiai, Zoltar).
- **Lexical Features**: Token-level information captured via hash-based feature hashing (default) or learnable token embeddings.
- **Syntactic Features**: Fine-grained Abstract Syntax Tree (AST) node types encoded as learnable embeddings.

The model constructs a graph from AST edges and uses a gated message-passing mechanism with GRU-based propagation to rank all statement nodes by their likelihood of being faulty.

#### Supported Models

| Model Key | Description |
|-----------|-------------|
| `l_dsgraph` | Core L-DSGraph model with gated fusion (SBFL + Lexical + AST) |
| `l_dsgraph_ablation` | Extended L-DSGraph supporting ablation variants and fusion strategies |
| `gcn` | Graph Convolutional Network baseline |
| `gat` | Graph Attention Network baseline |
| `ggat` / `gramus` | GraMuS GAT model (dense adjacency) |
| `spggat` / `spgramus` | GraMuS GAT model (sparse adjacency) |
| `depgraph` / `depgraphc` / `depgraphc2` / `depgrapho` | Dependency Graph variants with GRU propagation |
| `grace` / `grace_lite` | Transformer-based Grace model |
| `gnet4fl` | GraphSAGE-based GNET4FL model |
| `deepfl` | DEEP-FL multi-group feature model |
| `cnnfl` / `rnnfl` | CNN/RNN sequence-model baselines |

---

### Environment Requirements

| Component | Version / Specification |
|-----------|------------------------|
| Python | 3.9 or higher |
| PyTorch | 2.0+ with CUDA support (GPU recommended) |
| CUDA | 11.7+ (for GPU training) |
| OS | Windows / Linux / macOS |
| Memory | 16 GB RAM minimum, 32 GB recommended |
| GPU Memory | 8 GB VRAM minimum (for `hidden_dim=128`) |
| Disk Space | ~5 GB for raw data + processed dataset |

**Python Dependencies:**

```
torch>=2.0.0
numpy
scipy
pandas
networkx>=3.0
pyyaml
scikit-learn
openpyxl
```

---

### Installation Guide

#### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/ConDefects.git
cd ConDefects/L-DSGraph
```

#### Step 2: Create a Virtual Environment

```bash
# Using conda (recommended)
conda create -n ldsgraph python=3.9
conda activate ldsgraph

# Or using venv
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

#### Step 3: Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy pandas networkx pyyaml scikit-learn openpyxl
```

#### Step 4: Download the Dataset

The preprocessed dataset (`processed_data.pt`, ~700 MB) is hosted on cloud storage. Download and place the files under the `dataset/` directory.

| File | Size | Description |
|------|------|-------------|
| `dataset/processed_data.pt` | ~700 MB | Preprocessed binary dataset |
| `dataset/difficulty.txt` | ~50 KB | Bug difficulty labels |
| `dataset/lengs.txt` | ~50 KB | Statement length statistics |

> **Download Link:** [Quark Netdisk](https://pan.quark.cn/s/3619519bae7a),[google drive](https://drive.google.com/file/d/1RcwLZThOYSKJDuYX-XVtzXuw9uZKf3yo/view?usp=sharing)



After downloading, the `dataset/` directory should contain:

```
L-DSGraph/
└── dataset/
    ├── processed_data.pt
    ├── difficulty.txt
    └── lengs.txt
```

---

### Core Features

#### 1. Multi-Source Feature Fusion

L-DSGraph integrates three complementary feature types:

- **SBFL Features**: Coverage matrix (passed/failed test counts) + aggregated SBFL formula scores. Supports multiple SBFL variants: `mergeSBFLALL`, `mergeSBFL1.01`, `mergeSBFL0`, `mergeSBFL0.99`.
- **Lexical Features**: Configurable via `hash_num_bins` (default 64-d hash vector) or learned token embeddings (`token_emb` variant).
- **AST Syntax Features**: Fine-grained AST node type embeddings (`ast_emb_dim`).

#### 2. Flexible Fusion Strategies

| Strategy | Key | Description |
|----------|-----|-------------|
| Gated Fusion | `gate` (default) | Learnable sigmoid gate controlling core ↔ syntax information flow |
| Balanced Gated | `balanced_gate` | Dual-gate mechanism: `g * h_core + (1-g) * h_syn` |
| Concatenation | `concat` | Simple concatenation + linear projection |
| Cross-Attention | `use_cross_attn_fusion` | SBFL-Token cross-attention for core feature construction |

#### 3. Ablation Framework

The `l_dsgraph_ablation` model supports systematic ablation:

| Variant | Effect |
|---------|--------|
| `full` | All features enabled (default) |
| `wo_syntax` | Remove AST syntax features |
| `wo_lexical` | Remove lexical (hash/embedding) features |
| `wo_log1p` | Skip log1p normalization on hash features |
| `concat` | Use concatenation instead of gated fusion |
| `token_emb` | Use learnable token embeddings instead of hash |

#### 4. Cross-Validation & Stratification

- **K-Fold** (`cv_method: kfold`): Standard K-fold cross-validation with stratified grouping by bug/project.
- **Leave-One-Out** (`cv_method: leave_one_out`): Leave-one-project-out evaluation for cross-project testing.
- **Stratification**: By difficulty level (fixed/adaptive) or statement count (fixed/adaptive).

#### 5. Rich Evaluation Metrics

- Top-K accuracy (Top-1, Top-3, Top-5, Top-10)
- Mean Reciprocal Rank (MRR)
- Mean Average Rank (MAR)
- Mean First Rank (MFR)
- EXAM Score (percentage of code to examine)
- Statistical tests: Wilcoxon signed-rank test, Vargha-Delaney Â₁₂ effect size
- Per-difficulty breakdown
- Statement-root node shortcut analysis

---

### Project Structure

```
L-DSGraph/
├── DataConfig.py          # Data loading, preprocessing, and configuration
├── ModelAll.py            # All neural network model architectures
├── utils.py               # Training utilities, evaluation, statistics
├── train.py               # Main training and experiment orchestration script
├── package_data.py        # Data packaging script (raw → binary)
├── configs/
│   ├── default.yaml       # Default configuration (baselines)
│   ├── baseline.yaml      # Baseline experiment configurations
│   ├── ablation.yaml      # Ablation study configurations
│   └── ablation1diff.yaml # Ablation with PDG edge type
├── dataset/
│   ├── processed_data.pt  # Packaged binary data (generated)
│   ├── difficulty.txt     # Bug difficulty labels (generated)
│   └── lengs.txt          # Statement lengths (generated)
└── model/                 # Saved model checkpoints (generated)
```

---

### Usage Examples

#### Quick Start: Train the L-DSGraph Model

```bash
# Train with default configuration
python train.py --config configs/default.yaml

# Train specific experiments by name
python train.py --config configs/ablation.yaml --experiments ablation_full ablation_wo_syntax
```

#### Run a Single Baseline Model

```bash
# GCN
python train.py --config configs/baseline.yaml --experiments cnn_baseline

# GraMuS
python train.py --config configs/baseline.yaml --experiments gramus_h128_lr0001

# Grace
python train.py --config configs/baseline.yaml --experiments grace_h64_lr001
```

#### Run Ablation Studies

```bash
# Full model
python train.py --config configs/ablation.yaml --experiments ablation_full

# Without syntax features
python train.py --config configs/ablation.yaml --experiments ablation_wo_syntax

# Without lexical features
python train.py --config configs/ablation.yaml --experiments ablation_wo_lexical

# Concatenation fusion
python train.py --config configs/ablation.yaml --experiments ablation_concat

# Multiple variants combined
python train.py --config configs/ablation.yaml \
    --experiments ablation_wo_syntax_wo_lexical ablation_concat_wo_log1p
```

#### Run SBFL Purity Baseline

```bash
# Evaluate pure SBFL (no GNN) - Zoltar formula
python train.py --config configs/default.yaml --experiments E0_pure_sbfl_zoltar

# Evaluate pure SBFL - Ochiai formula
python train.py --config configs/default.yaml --experiments E0_pure_sbfl_ochiai
```

#### Custom Experiment Configuration

Add entries to `configs/default.yaml` under `experiments:`. Example:

```yaml
experiments:
  - name: "my_custom_exp"
    gnn_type: "l_dsgraph_ablation"
    variant: ["token_emb"]
    stratify_by: "difficulty_adaptive"
    sbfl_type: "mergeSBFLALL"
    hidden_dim: 128
    lr: 0.001
    epochs: 120
    cv_method: "kfold"
    k_folds: 5
    num_steps: 5
    dropout: 0.3
    ast_emb_dim: 16
    token_emb_dim: 16
    use_fusion_norm: true
    fusion_norm_type: "layer"
    fusion_type: "balanced_gate"
    use_edge_type: true
    use_cross_attn_fusion: true
    cross_attn_heads: 4
```

---

### Experiment Reproduction

#### Reproducing Paper Results

The complete experiment workflow consists of two stages:

**Stage 1: Run Baseline Models**

```bash
# Run all baselines
python train.py --config configs/baseline.yaml
```

**Stage 2: Run L-DSGraph Experiments**

```bash
# Full model
python train.py --config configs/ablation.yaml --experiments ablation_full

# Ablation variants
python train.py --config configs/ablation.yaml \
    --experiments ablation_wo_syntax ablation_wo_lexical ablation_concat ablation_wo_log1p
```

#### Output Artifacts

After training, results are saved to:

| Path | Content |
|------|---------|
| `model/*.pth` | Model checkpoints |
| `model/training_results_*.xlsx` | Excel workbook with results, per-config sheets, statistical tests, and per-program statement rankings |
| Console output | Per-epoch metrics and final fold-averaged results |

The Excel output includes multiple sheets:
- **Training Results**: Summary of all experiments
- **Per-config sheets**: Detailed per-configuration results
- **Statistical Tests**: Wilcoxon and Â₁₂ results
- **Per-Program Statement Ranks**: Line-level rank details

---

### Configuration Parameters

#### Data Configuration (`data:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_data_path` | str | `"dataset/"` | Fine-grained AST data directory |
| `sbfl_data_path` | str | `"dataset/"` | SBFL scores and statement-level data directory |
| `gramus_data_path` | str | `""` | GraMuS-format data directory |
| `difficulty_file` | str | `"dataset/difficulty.txt"` | Bug difficulty labels file |
| `data_source` | str | `"pickle"` | `"reference_path"` (raw), `"pickle"` (binary), or `"gramus"` |
| `sbfl_type` | str | `"mergeSBFLALL"` | SBFL formula variant |
| `use_original_coverage_matrix` | bool | `true` | Include raw coverage matrix |
| `use_sbfl` | bool | `true` | Include SBFL features |
| `use_full_ast` | bool | `true` | Use full AST (`ASTtreefull.json`) |
| `hash_num_bins` | int | `64` | Hash vector dimensionality for lexical features |
| `use_dense_adj` | bool | `false` | Use dense adjacency matrix (`true`) or sparse (`false`) |
| `edge_type` | str | `"ast"` | Edge type: `"ast"` |
| `bidirectional` | bool | `true` | Bidirectional edges |

#### Training Configuration (`training:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | `0` | Random seed |
| `batch_size` | int | `60` | Batch size |
| `patience` | int | `30` | Early stopping patience |
| `use_validation` | bool | `true` | Use validation set |
| `save_model` | bool | `true` | Save model checkpoints |
| `model_dir` | str | `"model"` | Model save directory |
| `weight_decay` | float | `0.001` | Weight decay (L2 regularization) |
| `loss_alpha` | float | `0.7` | ListMLE loss weight |
| `loss_beta` | float | `0.3` | BCE loss weight |
| `l1_lambda` | float | `0.0005` | L1 regularization strength |
| `scheduler_T0` | int | `20` | CosineAnnealingWarmRestarts T_0 |
| `scheduler_T_mult` | int | `2` | CosineAnnealingWarmRestarts T_mult |
| `grad_clip` | float | `1.0` | Gradient clipping max norm |

#### Experiment Configuration (`experiments:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | (required) | Experiment name for logging |
| `gnn_type` | str | (required) | Model type key (see [Supported Models](#supported-models)) |
| `hidden_dim` | int | `128` | Hidden dimension |
| `lr` | float | `0.001` | Learning rate |
| `epochs` | int | `120` | Maximum training epochs |
| `num_heads` | int | `5` | Number of attention heads |
| `cv_method` | str | `"kfold"` | Cross-validation: `"kfold"` or `"leave_one_out"` |
| `k_folds` | int | `5` | Number of K-fold splits |
| `variant` | str/list | `null` | Ablation variant(s) |
| `sbfl_type` | str | `"mergeSBFLALL"` | SBFL feature set |
| `sbfl_mode` | str | `null` | `"pure_sbfl"`, `"zero"`, `"random"`, `"single_formula"`, or `null` |
| `sbfl_formula` | str | `null` | Single SBFL formula name (e.g., `"zoltar"`, `"ochiai"`) |
| `num_steps` | int | `5` | Message-passing steps |
| `dropout` | float | `0.3` | Dropout rate |
| `ast_emb_dim` | int | `16` | AST node type embedding dimension |
| `token_emb_dim` | int | `16` | Token embedding dimension |
| `stratify_by` | str | `"difficulty"` | `"difficulty"`, `"difficulty_adaptive"`, `"stmt_count"` |
| `use_fusion_norm` | bool | `false` | Enable fusion normalization |
| `fusion_norm_type` | str | `"none"` | `"layer"`, `"l2"`, `"layer_fusion"` |
| `fusion_type` | str | `"auto"` | `"gate"`, `"balanced_gate"`, `"concat"` |
| `use_edge_type` | bool | `false` | Enable edge type gating |
| `use_cross_attn_fusion` | bool | `false` | Enable SBFL-Token cross-attention |
| `cross_attn_heads` | int | `4` | Cross-attention heads |
| `edge_gate_bias_init` | float | `-1` | Edge gate bias initialization |
| `edge_gate_l1_lambda` | float | `0.01` | Edge gate L1 regularization |
| `contrastive_weight` | float | `0.0` | Contrastive loss weight |
| `ranking_weight` | float | `0.0` | Pairwise ranking loss weight |
| `use_weighted_labels` | bool | `false` | Weighted labels (statement root vs. child nodes) |

---

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Top-K Accuracy** | Fraction of bugs where a faulty line appears in the top-K ranked lines |
| **MRR** | Mean Reciprocal Rank: average of `1/rank` for the first faulty line |
| **MAR** | Mean Average Rank: average rank across all faulty lines |
| **MFR** | Mean First Rank: average rank of the first faulty line found |
| **EXAM Score** | Percentage of code lines that must be examined to find the fault |
| **Shortcut Degree** | Difference between full-model Top-K and statement-root-only Top-K |
| **Â₁₂** | Vargha-Delaney Â₁₂ effect size between GNN and SBFL rankings (0.5 = no difference, >0.5 favors GNN) |
| **Wilcoxon p-value** | Statistical significance of GNN vs. SBFL comparisons |

---

### FAQ & Troubleshooting

<details>
<summary><b>Q: CUDA Out of Memory error</b></summary>

Reduce `batch_size` in `training:` section (e.g., from 60 to 30). For larger graphs, reduce `hidden_dim` from 128 to 64. Use `use_dense_adj: false` to save memory.
</details>

<details>
<summary><b>Q: How to run on CPU only?</b></summary>

The code automatically falls back to CPU if CUDA is not available. No configuration changes needed. Training will be significantly slower.
</details>

<details>
<summary><b>Q: What do the variant options mean?</b></summary>

See the [Ablation Framework](#5-ablation-framework) table. Variants can be combined as a list, e.g., `["token_emb", "concat"]` uses token embeddings with concatenation fusion.
</details>

<details>
<summary><b>Q: How to interpret "shortcut degree"?</b></summary>

Shortcut degree measures how much the model relies on non-statement-root AST nodes (children/grandchildren). A high shortcut degree suggests the model uses deep AST structure beyond surface-level statement nodes. A low or negative value suggests minimal benefit from deep AST.
</details>

---

### Citation

If you use this code in your research, please cite:

```bibtex
@article{ldsgraph2025,
  title     = {L-DSGraph: Line-level Dual-Source Graph Neural Network for Software Fault Localization},
  author    = {Your Names Here},
  journal   = {TBD},
  year      = {2025},
  note      = {Code repository: https://github.com/your-org/ConDefects}
}
```

For baseline model references:

```bibtex
@inproceedings{gramus2023,
  title     = {GraMuS: Graph-based Mutation-aided Spectrum-based Fault Localization},
  author    = {...},
  booktitle = {Proceedings of ...},
  year      = {2023}
}

@inproceedings{grace2022,
  title     = {GRACE: Graph-based Code Representation for Fault Localization},
  author    = {...},
  booktitle = {Proceedings of ...},
  year      = {2022}
}
```

---

---

<span id="chinese"></span>

## 中文版本

### 目录

1. [项目概述](#项目概述)
2. [环境配置要求](#环境配置要求)
3. [安装步骤](#安装步骤)
4. [核心功能说明](#核心功能说明)
5. [项目结构](#项目结构)
6. [使用示例](#使用示例)
7. [实验复现流程](#实验复现流程)
8. [配置参数说明](#配置参数说明)
9. [评估指标](#评估指标)
10. [常见问题解决](#常见问题解决)
11. [引用信息](#引用信息)

---

### 项目概述

**L-DSGraph**（Line-level Dual-Source Graph Neural Network，语句级双源图神经网络）是一个基于图神经网络的软件缺陷定位（Fault Localization, FL）框架，在语句/代码行级别进行缺陷定位。该模型联合利用三种互补信息源来精确定位有缺陷的代码行：

- **SBFL 特征**（基于频谱的缺陷定位）：覆盖矩阵及多种 SBFL 公式（如 Ochiai、Zoltar）的聚合排序分数。
- **词法特征**：通过基于哈希的特征哈希（默认）或可学习的 Token 嵌入来捕获 Token 级别的信息。
- **语法特征**：将细粒度抽象语法树（AST）节点类型编码为可学习的嵌入向量。

模型构建了一个基于 AST 边的图，然后使用基于门控的消息传递机制和 GRU 传播来对所有语句节点按其包含缺陷的可能性进行排序。

#### 支持的模型

| 模型名称 | 说明 |
|----------|------|
| `l_dsgraph` | 核心 L-DSGraph 模型，采用门控融合（SBFL + 词法 + AST） |
| `l_dsgraph_ablation` | 扩展版 L-DSGraph，支持消融变体和多种融合策略 |
| `gcn` | 图卷积网络基线 |
| `gat` | 图注意力网络基线 |
| `ggat` / `gramus` | GraMuS GAT 模型（稠密邻接） |
| `spggat` / `spgramus` | GraMuS GAT 模型（稀疏邻接） |
| `depgraph` / `depgraphc` / `depgraphc2` / `depgrapho` | 带 GRU 传播的依赖图变体 |
| `grace` / `grace_lite` | 基于 Transformer 的 Grace 模型 |
| `gnet4fl` | 基于 GraphSAGE 的 GNET4FL 模型 |
| `deepfl` | DEEP-FL 多组特征模型 |
| `cnnfl` / `rnnfl` | CNN/RNN 序列模型基线 |

---

### 环境配置要求

| 组件 | 版本 / 规格 |
|------|-------------|
| Python | 3.9 或更高版本 |
| PyTorch | 2.0+，需 CUDA 支持（推荐使用 GPU） |
| CUDA | 11.7+（用于 GPU 训练） |
| 操作系统 | Windows / Linux / macOS |
| 内存 | 最低 16 GB RAM，推荐 32 GB |
| GPU 显存 | 最低 8 GB VRAM（`hidden_dim=128` 时） |
| 磁盘空间 | 约 5 GB（原始数据 + 处理后数据集） |

**Python 依赖包：**

```
torch>=2.0.0
numpy
scipy
pandas
networkx>=3.0
pyyaml
scikit-learn
openpyxl
```

---

### 安装步骤

#### 第一步：克隆仓库

```bash
git clone https://github.com/your-org/ConDefects.git
cd ConDefects/L-DSGraph
```

#### 第二步：创建虚拟环境

```bash
# 使用 conda（推荐）
conda create -n ldsgraph python=3.9
conda activate ldsgraph

# 或使用 venv
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

#### 第三步：安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy pandas networkx pyyaml scikit-learn openpyxl
```

#### 第四步：下载数据集

预处理后的数据集（`processed_data.pt`，约 700 MB）托管在云存储上。下载后将文件放入 `dataset/` 目录。

| 文件 | 大小 | 说明 |
|------|------|------|
| `dataset/processed_data.pt` | ~700 MB | 预处理后的二进制数据集 |
| `dataset/difficulty.txt` | ~50 KB | Bug 难度标签 |
| `dataset/lengs.txt` | ~50 KB | 语句长度统计 |

> **下载链接：** [夸克网盘](https://pan.quark.cn/) / [百度网盘](https://pan.baidu.com/)
>
> *(请替换为实际的分享链接)*

下载完成后，`dataset/` 目录应包含：

```
L-DSGraph/
└── dataset/
    ├── processed_data.pt
    ├── difficulty.txt
    └── lengs.txt
```

---

### 核心功能说明

#### 1. 多源特征融合

L-DSGraph 整合三种互补特征类型：

- **SBFL 特征**：覆盖矩阵（通过/失败测试计数）+ 聚合的 SBFL 公式分数。支持多种 SBFL 变体：`mergeSBFLALL`、`mergeSBFL1.01`、`mergeSBFL0`、`mergeSBFL0.99`。
- **词法特征**：通过 `hash_num_bins`（默认 64 维哈希向量）或可学习的 Token 嵌入（`token_emb` 变体）进行配置。
- **AST 语法特征**：细粒度 AST 节点类型嵌入（`ast_emb_dim`）。

#### 2. 灵活的融合策略

| 策略 | 配置键 | 说明 |
|------|--------|------|
| 门控融合 | `gate`（默认） | 可学习的 sigmoid 门控，控制核心特征 ↔ 语法特征的信息流 |
| 均衡门控 | `balanced_gate` | 双门控机制：`g * h_core + (1-g) * h_syn` |
| 拼接融合 | `concat` | 简单拼接 + 线性投影 |
| 交叉注意力 | `use_cross_attn_fusion` | SBFL-Token 交叉注意力，用于构建核心特征 |

#### 3. 消融实验框架

`l_dsgraph_ablation` 模型支持系统性的消融实验：

| 变体 | 效果 |
|------|------|
| `full` | 所有特征启用（默认） |
| `wo_syntax` | 移除 AST 语法特征 |
| `wo_lexical` | 移除词法（哈希/嵌入）特征 |
| `wo_log1p` | 跳过哈希特征的 log1p 归一化 |
| `concat` | 使用拼接融合替代门控融合 |
| `token_emb` | 使用可学习的 Token 嵌入替代哈希特征 |

#### 4. 交叉验证与分层抽样

- **K 折交叉验证**（`cv_method: kfold`）：标准 K 折交叉验证，按 Bug/项目进行分层分组。
- **留一法**（`cv_method: leave_one_out`）：留一项目评估，用于跨项目测试。
- **分层方式**：按难度级别（固定/自适应）或语句数量（固定/自适应）。

#### 5. 丰富的评估指标

- Top-K 准确率（Top-1、Top-3、Top-5、Top-10）
- 平均倒数排名（MRR）
- 平均排名（MAR）
- 平均首次排名（MFR）
- EXAM 分数（需检查代码的百分比）
- 统计检验：Wilcoxon 符号秩检验、Vargha-Delaney Â₁₂ 效应量
- 按难度细分分析
- 语句根节点捷径分析

---

### 项目结构

```
L-DSGraph/
├── DataConfig.py          # 数据加载、预处理和配置
├── ModelAll.py            # 所有神经网络模型架构
├── utils.py               # 训练工具、评估、统计检验
├── train.py               # 主训练和实验编排脚本
├── package_data.py        # 数据打包脚本（原始 → 二进制）
├── configs/
│   ├── default.yaml       # 默认配置（基线）
│   ├── baseline.yaml      # 基线实验配置
│   ├── ablation.yaml      # 消融实验配置
│   └── ablation1diff.yaml # PDG 边类型的消融实验
├── dataset/
│   ├── processed_data.pt  # 打包的二进制数据（自动生成）
│   ├── difficulty.txt     # Bug 难度标签（自动生成）
│   └── lengs.txt          # 语句长度（自动生成）
└── model/                 # 模型检查点保存目录（自动生成）
```

---

### 使用示例

#### 快速开始：训练 L-DSGraph 模型

```bash
# 使用默认配置训练
python train.py --config configs/default.yaml

# 按名称训练特定实验
python train.py --config configs/ablation.yaml --experiments ablation_full ablation_wo_syntax
```

#### 运行单一基线模型

```bash
# GCN
python train.py --config configs/baseline.yaml --experiments cnn_baseline

# GraMuS
python train.py --config configs/baseline.yaml --experiments gramus_h128_lr0001

# Grace
python train.py --config configs/baseline.yaml --experiments grace_h64_lr001
```

#### 运行消融实验

```bash
# 完整模型
python train.py --config configs/ablation.yaml --experiments ablation_full

# 无语法特征
python train.py --config configs/ablation.yaml --experiments ablation_wo_syntax

# 无词法特征
python train.py --config configs/ablation.yaml --experiments ablation_wo_lexical

# 拼接融合
python train.py --config configs/ablation.yaml --experiments ablation_concat

# 多变体组合
python train.py --config configs/ablation.yaml \
    --experiments ablation_wo_syntax_wo_lexical ablation_concat_wo_log1p
```

#### 运行纯 SBFL 基线

```bash
# 评估纯 SBFL（不使用 GNN）- Zoltar 公式
python train.py --config configs/default.yaml --experiments E0_pure_sbfl_zoltar

# 评估纯 SBFL - Ochiai 公式
python train.py --config configs/default.yaml --experiments E0_pure_sbfl_ochiai
```

#### 自定义实验配置

在 `configs/default.yaml` 的 `experiments:` 部分添加条目。示例：

```yaml
experiments:
  - name: "my_custom_exp"
    gnn_type: "l_dsgraph_ablation"
    variant: ["token_emb"]
    stratify_by: "difficulty_adaptive"
    sbfl_type: "mergeSBFLALL"
    hidden_dim: 128
    lr: 0.001
    epochs: 120
    cv_method: "kfold"
    k_folds: 5
    num_steps: 5
    dropout: 0.3
    ast_emb_dim: 16
    token_emb_dim: 16
    use_fusion_norm: true
    fusion_norm_type: "layer"
    fusion_type: "balanced_gate"
    use_edge_type: true
    use_cross_attn_fusion: true
    cross_attn_heads: 4
```

---

### 实验复现流程

完整的实验工作流包含两个阶段：

**阶段一：运行基线模型**

```bash
# 运行所有基线
python train.py --config configs/baseline.yaml
```

**阶段二：运行 L-DSGraph 实验**

```bash
# 完整模型
python train.py --config configs/ablation.yaml --experiments ablation_full

# 消融变体
python train.py --config configs/ablation.yaml \
    --experiments ablation_wo_syntax ablation_wo_lexical ablation_concat ablation_wo_log1p
```

#### 输出产物

训练完成后，结果保存到以下位置：

| 路径 | 内容 |
|------|------|
| `model/*.pth` | 模型检查点 |
| `model/training_results_*.xlsx` | Excel 工作簿，包含结果汇总、按配置分表、统计检验和逐程序语句排名 |
| 控制台输出 | 每 epoch 指标及最终折平均结果 |

Excel 输出包含多个工作表：
- **训练结果**（Training Results）：所有实验汇总
- **按配置分表**：每个配置的详细结果
- **统计检验**（Statistical Tests）：Wilcoxon 和 Â₁₂ 结果
- **逐程序语句排名**（Per-Program Statement Ranks）：代码行级别排名详情

---

### 配置参数说明

#### 数据配置（`data:`）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root_data_path` | str | `"dataset/"` | 细粒度 AST 数据目录 |
| `sbfl_data_path` | str | `"dataset/"` | SBFL 分数和语句级数据目录 |
| `gramus_data_path` | str | `""` | GraMuS 格式数据目录 |
| `difficulty_file` | str | `"dataset/difficulty.txt"` | Bug 难度标签文件 |
| `data_source` | str | `"pickle"` | `"reference_path"`（原始）、`"pickle"`（二进制）或 `"gramus"` |
| `sbfl_type` | str | `"mergeSBFLALL"` | SBFL 公式变体 |
| `use_original_coverage_matrix` | bool | `true` | 是否包含原始覆盖矩阵 |
| `use_sbfl` | bool | `true` | 是否包含 SBFL 特征 |
| `use_full_ast` | bool | `true` | 使用完整 AST（`ASTtreefull.json`） |
| `hash_num_bins` | int | `64` | 词法特征哈希向量维度 |
| `use_dense_adj` | bool | `false` | 使用稠密邻接矩阵（`true`）或稀疏（`false`） |
| `edge_type` | str | `"ast"` | 边类型：`"ast"` |
| `bidirectional` | bool | `true` | 是否使用双向边 |

#### 训练配置（`training:`）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seed` | int | `0` | 随机种子 |
| `batch_size` | int | `60` | 批次大小 |
| `patience` | int | `30` | 早停耐心值 |
| `use_validation` | bool | `true` | 是否使用验证集 |
| `save_model` | bool | `true` | 是否保存模型检查点 |
| `model_dir` | str | `"model"` | 模型保存目录 |
| `weight_decay` | float | `0.001` | 权重衰减（L2 正则化） |
| `loss_alpha` | float | `0.7` | ListMLE 损失权重 |
| `loss_beta` | float | `0.3` | BCE 损失权重 |
| `l1_lambda` | float | `0.0005` | L1 正则化强度 |
| `scheduler_T0` | int | `20` | CosineAnnealingWarmRestarts T_0 |
| `scheduler_T_mult` | int | `2` | CosineAnnealingWarmRestarts T_mult |
| `grad_clip` | float | `1.0` | 梯度裁剪最大范数 |

#### 实验配置（`experiments:`）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | （必填） | 实验名称，用于日志记录 |
| `gnn_type` | str | （必填） | 模型类型（见[支持的模型](#支持的模型)） |
| `hidden_dim` | int | `128` | 隐藏层维度 |
| `lr` | float | `0.001` | 学习率 |
| `epochs` | int | `120` | 最大训练轮数 |
| `num_heads` | int | `5` | 注意力头数 |
| `cv_method` | str | `"kfold"` | 交叉验证方式：`"kfold"` 或 `"leave_one_out"` |
| `k_folds` | int | `5` | K 折交叉验证折数 |
| `variant` | str/list | `null` | 消融变体 |
| `sbfl_type` | str | `"mergeSBFLALL"` | SBFL 特征集 |
| `sbfl_mode` | str | `null` | `"pure_sbfl"`、`"zero"`、`"random"`、`"single_formula"` 或 `null` |
| `sbfl_formula` | str | `null` | 单一 SBFL 公式名（如 `"zoltar"`、`"ochiai"`） |
| `num_steps` | int | `5` | 消息传递步数 |
| `dropout` | float | `0.3` | Dropout 比率 |
| `ast_emb_dim` | int | `16` | AST 节点类型嵌入维度 |
| `token_emb_dim` | int | `16` | Token 嵌入维度 |
| `stratify_by` | str | `"difficulty"` | `"difficulty"`、`"difficulty_adaptive"`、`"stmt_count"` |
| `use_fusion_norm` | bool | `false` | 是否启用融合归一化 |
| `fusion_norm_type` | str | `"none"` | `"layer"`、`"l2"`、`"layer_fusion"` |
| `fusion_type` | str | `"auto"` | `"gate"`、`"balanced_gate"`、`"concat"` |
| `use_edge_type` | bool | `false` | 是否启用边类型门控 |
| `use_cross_attn_fusion` | bool | `false` | 是否启用 SBFL-Token 交叉注意力 |
| `cross_attn_heads` | int | `4` | 交叉注意力头数 |
| `edge_gate_bias_init` | float | `-1` | 边门控偏置初始化值 |
| `edge_gate_l1_lambda` | float | `0.01` | 边门控 L1 正则化系数 |
| `contrastive_weight` | float | `0.0` | 对比学习损失权重 |
| `ranking_weight` | float | `0.0` | 成对排序损失权重 |
| `use_weighted_labels` | bool | `false` | 是否使用加权标签（语句根 vs. 子节点） |

---

### 评估指标

| 指标 | 说明 |
|------|------|
| **Top-K 准确率** | 有缺陷的行出现在排名前 K 的行中的 Bug 比例 |
| **MRR** | 平均倒数排名：所有 Bug 中首个缺陷行的 `1/rank` 平均值 |
| **MAR** | 平均排名：所有缺陷行排名的平均值 |
| **MFR** | 平均首次排名：找到的第一个缺陷行的平均排名 |
| **EXAM 分数** | 为找到缺陷所需检查的代码行百分比 |
| **捷径程度** | 完整模型 Top-K 与仅语句根节点 Top-K 的差异 |
| **Â₁₂** | Vargha-Delaney Â₁₂ 效应量，衡量 GNN 与 SBFL 排名差异（0.5 = 无差异，>0.5 表示 GNN 更优） |
| **Wilcoxon p 值** | GNN 与 SBFL 比较的统计显著性 |

---

### 常见问题解决

<details>
<summary><b>Q: CUDA 显存不足（Out of Memory）错误</b></summary>

在 `training:` 部分减小 `batch_size`（例如从 60 降至 30）。对于较大的图，可将 `hidden_dim` 从 128 降至 64。设置 `use_dense_adj: false` 以节省内存。
</details>

<details>
<summary><b>Q: 如何仅在 CPU 上运行？</b></summary>

如果 CUDA 不可用，代码会自动回退到 CPU，无需修改配置。但训练速度会显著降低。
</details>

<details>
<summary><b>Q: 变体选项是什么意思？</b></summary>

请参阅[消融实验框架](#5-消融实验框架)表格。变体可作为列表组合，例如 `["token_emb", "concat"]` 表示使用 Token 嵌入和拼接融合。
</details>

<details>
<summary><b>Q: 如何理解 "捷径程度"（shortcut degree）？</b></summary>

捷径程度衡量模型在多大程度上依赖非语句根 AST 节点（子节点/孙子节点）。高捷径程度表明模型利用了超越表层语句节点的深层 AST 结构。低值或负值表明深层 AST 带来的收益很小。
</details>

---

### 引用信息

如果您在研究中使用此代码，请引用：

```bibtex
@article{ldsgraph2025,
  title     = {L-DSGraph: Line-level Dual-Source Graph Neural Network for Software Fault Localization},
  author    = {Your Names Here},
  journal   = {TBD},
  year      = {2025},
  note      = {代码仓库: https://github.com/your-org/ConDefects}
}
```

基线模型引用：

```bibtex
@inproceedings{gramus2023,
  title     = {GraMuS: Graph-based Mutation-aided Spectrum-based Fault Localization},
  author    = {...},
  booktitle = {Proceedings of ...},
  year      = {2023}
}

@inproceedings{grace2022,
  title     = {GRACE: Graph-based Code Representation for Fault Localization},
  author    = {...},
  booktitle = {Proceedings of ...},
  year      = {2022}
}
```
