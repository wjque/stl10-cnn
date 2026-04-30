# 基于 STL‑10 数据集的图像分类

## 项目框架

- `./configs` 从以下三个角度设计不同配置
  - 数据增强：无数据增强 | 使用随机裁剪 | 使用mixup
  - 模型结构：是否使用残差网络 / 两种模型深度 / ReLU vs Sigmoid / Max Pooling vs Avg Pooling
  - 正则化：是否使用 Batch Norm
  - 超参数选择：学习率 / 优化器
- `./model` 存放不同的模型框架
- `./scripts` 存放训练脚本和测试推理脚本
  - `split.py` 从训练集中按 80/20 划分验证集
  - `train.py` 训练脚本
  - `infer.py` 推理测试脚本
  - `run.sh` 一键复现所有实验
- `./utils` 存放工具函数
  - `metrics.py` 计算各种不同的指标：准确率、召回率、F1-Score、AUC 等
  - `dataloader.py` 为训练/验证/测试集创建 DataLoader
  - `visulization.py` 用于可视化训练过程的损失曲线和准确率曲线，以及对模型提取的特征可视化以进行可解释性分析

## 使用方法

### 数据集预处理

从原始训练集中按照 80/20 的比例划分出验证集（移动文件到 `STL10/val/`）：

```bash
cd project_2
python scripts/split.py
```

参数说明：
- `--data_dir STL10`：数据集根目录（默认）
- `--val_ratio 0.2`：验证集比例（默认 0.2）
- `--seed 42`：随机种子，确保可复现（默认 42）
- `--dry_run`：预览划分结果，不实际移动文件

### 环境配置

```bash
# 创建虚拟环境并激活
conda create -n cnn python=3.11 -y
conda activate cnn

# 安装依赖
pip install -r requirements.txt
```

### 一键运行所有实验

```bash
cd project_2
bash scripts/run.sh
```

该脚本会自动完成：训练 9 个配置 → 测试集评估 → 可视化对比

### 单独训练某个配置

```bash
python scripts/train.py --config configs.config_01_baseline
```

### 测试集推理

```bash
# 评估所有训练好的模型
python scripts/infer.py --all

# 评估单个模型
python scripts/infer.py --model configs.config_01_baseline
```

### 输出说明

| 目录 | 内容 |
|------|------|
| `outputs/models/` | 训练好的模型权重 (.pth) |
| `outputs/logs/` | 训练日志 (JSON)，含损失/准确率曲线数据和测试集指标 |
| `outputs/figures/` | 训练曲线图、t-SNE 特征可视化、模型对比图 |

### 实验配置列表

| 编号 | 实验目的 | 数据增强 | 残差 | 深度 | 激活 | 池化 | BN | 优化器 |
|------|---------|----------|------|------|------|------|----|--------|
| 01 | 基线 | 无 | 无 | 浅 | ReLU | Max | 无 | SGD |
| 02 | 数据增强 | 随机裁剪 | 无 | 浅 | ReLU | Max | 无 | SGD |
| 03 | MixUp | MixUp | 无 | 浅 | ReLU | Max | 无 | SGD |
| 04 | 残差网络 | 随机裁剪 | 有 | 浅 | ReLU | Max | 无 | SGD |
| 05 | 深层网络 | 随机裁剪 | 无 | 深 | ReLU | Max | 无 | SGD |
| 06 | Sigmoid | 随机裁剪 | 无 | 浅 | Sigmoid | Max | 无 | SGD |
| 07 | 平均池化 | 随机裁剪 | 无 | 浅 | ReLU | Avg | 无 | SGD |
| 08 | BatchNorm | 随机裁剪 | 无 | 浅 | ReLU | Max | 有 | SGD |
| 09 | AdamW | 随机裁剪 | 无 | 浅 | ReLU | Max | 无 | AdamW |
| 10 | 深层残差 | 随机裁剪 | 有 | 深 | ReLU | Max | 无 | SGD |
