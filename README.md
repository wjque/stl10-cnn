# 基于 STL‑10 数据集的图像分类

## 项目框架

- `./configs` 从以下三个角度设计不同配置
  - 数据增强：无数据增强 / 使用随机裁剪 / 使用颜色抖动 / 使用水平翻转
  - 模型结构：两种模型深度 / ReLU vs Sigmoid / Max Pooling vs Avg Pooling
  - 正则化：是否使用 Batch Norm
  - 超参数选择：SGD vs AdamW
- `./model` 模型框架
- `./scripts` 存放训练脚本和测试推理脚本
  - `train.py` 训练脚本
  - `infer.py` 推理测试脚本
  - `run.sh` 批量运行实验并进行评估
- `./utils` 存放工具函数
  - `dataloader.py` 为训练/验证/测试集创建 DataLoader
  - `grad_utils` 用于分析模型各层梯度范数，观察模型梯度回传是否异常
  - `metrics.py` 计算各种不同的指标：准确率、召回率、F1-Score、AUC 等
  - `split.py` 从训练集中按 80/20 划分验证集
  - `visulization.py` 用于可视化训练过程的损失曲线和准确率曲线，以及对模型提取的特征可视化以进行可解释性分析

## 使用方法

### 数据集预处理

从原始训练集中按照 80/20 的比例划分出验证集（移动文件到 `STL10/val/`）：

```bash
cd project_2
python utils/split.py
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

### 批量运行实验

```bash
cd project_2
bash scripts/run.sh
```

该脚本会自动运行三组实验，分别探究数据增强的方式、正则化方式以及模型结构对分类效果的影响

### 单独训练某个配置

```bash
python scripts/train.py --config configs.config_01_baseline
```

### 测试集推理

```bash
# 评估单个模型
python scripts/infer.py --model 01_baseline

# 评估多个模型
python scripts/infer.py --model 01_baseline 03_adamw 04_deep --output-dir outputs/figures
```

### 输出说明

| 目录 | 内容 |
|------|------|
| `outputs/models/` | 训练好的模型权重 (.pth) |
| `outputs/logs/` | 训练日志 (JSON)，含损失/准确率曲线数据和测试集指标 |
| `outputs/figures/` | 训练曲线图、t-SNE 特征可视化、模型对比图 |

### 实验配置列表

| 编号 | 实验目的 | 数据增强 | 深度 | 激活 | 池化 | BN | Dropout | 优化器 |
|------|---------|----------|------|------|------|----|---------|--------|
| 01_baseline | 基线 | 无 | 浅 | ReLU | Max | ✗ | 0.0 | SGD |
| 02_random_crop | 数据增强 | 随机裁剪 | 浅 | ReLU | Max | ✗ | 0.0 | SGD |
| 02_flip_h | 数据增强 | 水平翻转 | 浅 | ReLU | Max | ✗ | 0.0 | SGD |
| 02_color_jitter | 数据增强 | 颜色抖动 | 浅 | ReLU | Max | ✗ | 0.0 | SGD |
| 03_batchnorm | 正则化 | 水平翻转 | 浅 | ReLU | Max | ✓ | 0.0 | SGD |
| 03_dropout | 正则化 | 水平翻转 | 浅 | ReLU | Max | ✓ | 0.5 | SGD |
| 03_adamw | 优化器 | 水平翻转 | 浅 | ReLU | Max | ✓ | 0.0 | AdamW |
| 04_deep | 模型结构 | 水平翻转 | 深 | ReLU | Max | ✓ | 0.0 | AdamW |
| 04_avgpool | 模型结构 | 水平翻转 | 浅 | ReLU | Avg | ✓ | 0.0 | AdamW |
| 04_sigmoid | 模型结构 | 水平翻转 | 浅 | Sigmoid | Max | ✓ | 0.0 | AdamW |
