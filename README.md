# 基于 STL-10 数据集的图像分类

## 实验设计

项目已重构为 4 个阶段的控制变量实验，所有主实验统一采用：

- 固定 epoch
- 不使用 scheduler
- 不使用 early stopping
- 每组实验默认使用 3 个随机种子：`42, 52, 62`
- 用验证集选择 baseline，用测试集做最终汇总

4 个阶段如下：

1. `stage1`：优化器与学习率
2. `stage2`：数据增强
3. `stage3`：正则化全因子实验
4. `stage4`：模型深度与池化方式

## 项目结构

- `configs/base.py`：统一基础配置
- `configs/experiments/`：四个阶段的实验生成器
- `model/`：CNN 模型定义
- `scripts/train.py`：训练入口，仅支持新实验名
- `scripts/run.sh`：按阶段批量训练
- `scripts/eval.sh`：按阶段批量评估并汇总
- `scripts/summarize_stage.py`：聚合多 seed 结果
- `utils/dataloader.py`：数据加载与组合增强

## 数据准备

先从 `STL10/train` 划分验证集到 `STL10/val`：

```bash
python utils/split.py
```

## 环境配置

```bash
conda create -n cnn python=3.11 -y
conda activate cnn
pip install -r requirements.txt
```

## 运行方式

### 训练某个阶段

```bash
bash scripts/run.sh stage1
```

从上一个阶段最佳实验继续派生 baseline：

```bash
bash scripts/run.sh stage2 s1_optsgd_lr1e-2_seed42
```

### 训练单个实验

```bash
python scripts/train.py --experiment s1_optsgd_lr1e-2_seed42
```

### 评估并汇总某个阶段

```bash
bash scripts/eval.sh stage1
```

或指定基线：

```bash
bash scripts/eval.sh stage3 s2_augcrop_seed42
```

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
| `outputs/figures/` | 训练曲线、对比图、PCA 等可视化 |

## 分析功能

保留的分析入口：

- `python scripts/analysis.py pca --model <exp1> <exp2>`
- `python scripts/analysis.py cm --model <exp1> <exp2>`
- `python scripts/analysis.py train --model <exp>`
- `python scripts/analysis.py compare --model <exp1> <exp2>`
- `python scripts/analysis.py eval --stage stage1`
- `python scripts/analysis.py table --model <exp1> <exp2>`
- `python scripts/analysis.py table --table-kind stage-summary --stage stage1`
