# 基于 STL-10 数据集的图像分类

## 项目功能

本项目实现了一个面向 STL-10 的 CNN 图像分类实验框架，支持：

- 统一配置驱动的训练与评估
- 按阶段组织的控制变量实验
- 单次实验与批量实验运行
- 训练曲线、混淆矩阵、PCA、t-SNE 等分析可视化
- 结果汇总与 LaTeX 表格导出

## 实验设计

项目包括 4 个阶段的控制变量实验，所有主实验统一采用：

- 固定 epoch
- 使用 cosine scheduler
- 不使用 early stopping
- 每组实验默认使用 3 个随机种子：`42, 20260505, 123`
- 用验证集选择 baseline，用测试集做最终汇总

四个阶段分别为：

1. `stage1`：优化器与学习率
2. `stage2`：数据增强
3. `stage3`：正则化全因子实验
4. `stage4`：模型深度与池化方式

## 项目结构

- `configs/base.py`：统一基础配置
- `configs/experiments/`：四个阶段的实验生成器与注册入口
- `model/`：CNN 模型定义
- `scripts/train.py`：单实验训练入口
- `scripts/analysis.py`：分析入口，包含评估、可视化与表格导出
- `scripts/run.sh`：按阶段批量训练
- `scripts/eval.sh`：按阶段批量评估并汇总
- `scripts/summarize_stage.py`：聚合多 seed 结果
- `utils/dataloader.py`：数据加载与组合增强
- `utils/visualization.py`：训练曲线、PCA、t-SNE、混淆矩阵等绘图

## 实验方案

### 数据准备

先从 `STL10/train` 划分验证集到 `STL10/val`：

```bash
python utils/split.py
```

### 环境配置

```bash
conda create -n cnn python=3.11 -y
conda activate cnn
pip install -r requirements.txt
```

### 运行方式

训练某个阶段：

```bash
bash scripts/run.sh stage1
```

从上一个阶段最佳实验继续派生 baseline：

```bash
bash scripts/run.sh stage2 s1_optsgd_lr1e-2_seed42
```

训练单个实验：

```bash
python scripts/train.py --experiment s1_optsgd_lr1e-2_seed42
```

评估并汇总某个阶段：

```bash
bash scripts/eval.sh stage1
```

该命令会依次执行：

- 按阶段批量评估验证集和测试集指标
- 为每个实验生成 `t-SNE` 图
- 汇总阶段结果到报告文件

指定基线：

```bash
bash scripts/eval.sh stage3 s2_augcrop_seed42
```

生成单个实验的训练曲线：

```bash
python scripts/analysis.py train --model s1_optsgd_lr1e-2_seed42
```

生成单个或多个实验的 t-SNE 图：

```bash
python scripts/analysis.py tsne --model s1_optsgd_lr1e-2_seed42
python scripts/analysis.py tsne --stage stage1
```

生成混淆矩阵：

```bash
python scripts/analysis.py cm --model s1_optsgd_lr1e-2_seed42
```

生成 PCA 可视化：

```bash
python scripts/analysis.py pca --model s1_optsgd_lr1e-2_seed42
```

生成多实验对比图：

```bash
python scripts/analysis.py compare --model s1_optsgd_lr1e-2_seed42 s1_optadamw_lr1e-3_seed42
```

导出实验结果表格：

```bash
python scripts/analysis.py table --model s1_optsgd_lr1e-2_seed42 s1_optadamw_lr1e-3_seed42
python scripts/analysis.py table --table-kind stage-summary --stage stage1
```

### 实验复现方案

1. 先执行数据划分：`python utils/split.py`
2. 安装环境依赖：`pip install -r requirements.txt`
3. 运行阶段训练：`bash scripts/run.sh <stage>`
4. 运行阶段评估：`bash scripts/eval.sh <stage> [baseline]`
5. 查看输出目录中的模型、日志、评估结果和图表

## 阶段实验矩阵

### Stage 1

- `sgd + 1e-2`
- `sgd + 1e-3`
- `adamw + 1e-3`
- `adamw + 1e-4`

### Stage 2

- `none`
- `random_crop`
- `flip_h`
- `random_crop + flip_h`

### Stage 3

`BN x Dropout x Weight Decay = 8` 组：

- `BN in {off, on}`
- `Dropout in {0.0, 0.5}`
- `Weight Decay in {0, 1e-4}`

### Stage 4

- `shallow + max`
- `shallow + avg`
- `deep + max`
- `deep + avg`

## 输出目录

| 目录 | 内容 |
|------|------|
| `outputs/models/` | 训练好的模型权重 |
| `outputs/logs/` | 单次实验日志 |
| `outputs/eval/` | 验证集/测试集评估结果 |
| `outputs/reports/` | 按阶段聚合的结果汇总 |
| `outputs/figures/` | 所有分析图片输出根目录 |

图片输出规则：

- `outputs/figures/training/<model_name>.png`：训练曲线
- `outputs/figures/confusion/<model_name>.png`：混淆矩阵
- `outputs/figures/tsne/<model_name>.png`：t-SNE 图
- `outputs/figures/pca/<model_name>/`：PCA 分阶段可视化
- `outputs/figures/comparison*`：多实验对比图

## 分析功能

- `python scripts/analysis.py pca --model <exp1> <exp2>`
- `python scripts/analysis.py cm --model <exp1> <exp2>`
- `python scripts/analysis.py train --model <exp>`
- `python scripts/analysis.py tsne --model <exp1> <exp2>`
- `python scripts/analysis.py compare --model <exp1> <exp2>`
- `python scripts/analysis.py eval --stage stage1`
- `python scripts/analysis.py table --model <exp1> <exp2>`
- `python scripts/analysis.py table --table-kind stage-summary --stage stage1`
