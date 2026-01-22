# HC3英文数据集评估指南

## 📋 项目说明

本项目用于在HC3英文数据集上评估Binoculars方法的AI文本检测性能。

## 🎯 关于双模型配置

### ❓ 必须使用两个不同的模型

Binoculars使用两个模型计算检测分数：

```
Binoculars Score = Perplexity(Performer) / Cross-Entropy(Observer, Performer)
```

**为什么需要两个不同的模型？**

- **Observer模型**：提供参考视角
- **Performer模型**：评估文本质量
- 两个模型的差异是检测的关键：
  - AI文本：两个模型预测一致 → 分数高（接近1.0）
  - 人类文本：两个模型预测有差异 → 分数低

**推荐配置：**

| 场景 | Observer | Performer | GPU需求 | 说明 |
|------|----------|-----------|---------|------|
| 高性能（默认）✅ | `EleutherAI/gpt-neo-1.3B` | `EleutherAI/gpt-neo-2.7B` | 8GB | RTX 2070S级别 |
| 平衡性能 | `gpt2-medium` | `EleutherAI/gpt-neo-1.3B` | 4-6GB | 中等显卡 |
| 轻量级 | `gpt2` | `gpt2-medium` | 2GB | 资源有限时 |
| 最佳（需大GPU） | `tiiuae/falcon-7b` | `tiiuae/falcon-7b-instruct` | 16GB+ | 7B参数 |

详细原理请参考：[DUAL_MODEL_EXPLANATION.md](DUAL_MODEL_EXPLANATION.md)

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install datasets transformers torch scikit-learn matplotlib pandas
```

### 2. 准备HC3英文数据集

```bash
python experiments/hc3_loader.py
```

这将下载并格式化HC3英文数据集到 `datasets/hc3/hc3_english_qa.jsonl`

### 3. 运行评估

#### 快速测试（100样本，使用默认配置）

```bash
# 使用默认模型：gpt-neo-1.3B + gpt-neo-2.7B（高性能，需要8GB显存）
python experiments/run_hc3_evaluation.py \
    --dataset_path datasets/hc3/hc3_english_qa.jsonl \
    --max_samples 100
```

#### 完整评估（全数据集，使用默认配置）

```bash
# 使用默认配置：gpt-neo-1.3B + gpt-neo-2.7B（适合RTX 2070S等8GB显卡）
python experiments/run_hc3_evaluation.py \
    --dataset_path datasets/hc3/hc3_english_qa.jsonl \
    --output_dir results/hc3_full_evaluation
```

#### 高性能评估（需要大GPU，使用Falcon模型）

```bash
# 仅在有16GB+显存时使用
python experiments/run_hc3_evaluation.py \
    --dataset_path datasets/hc3/hc3_english_qa.jsonl \
    --observer tiiuae/falcon-7b \
    --performer tiiuae/falcon-7b-instruct \
    --batch_size 8 \
    --use_bfloat16 \
    --output_dir results/hc3_falcon_evaluation
```

#### 使用Windows脚本

```bash
run_hc3_eval.bat
```

## 📊 评估指标

评估完成后会输出以下指标：

| 指标 | 说明 |
|------|------|
| **Accuracy** | 整体准确率 |
| **Precision** | 精确率（检测为AI的文本中真正是AI的比例） |
| **Recall** | 召回率（所有AI文本中被正确检测的比例） |
| **F1-Score** | F1分数（精确率和召回率的调和平均） |
| **ROC-AUC** | ROC曲线下面积 |
| **TPR@FPR=0.01%** | 在极低误报率（0.01%）下的真正例率 |
| **FPR@TPR=95%** | 在95%召回率下的误报率 |

## 📁 输出文件

评估结果保存在 `results/` 目录：

```
results/hc3_eval_20260119_120000/
├── metrics_summary.json       # 指标摘要
├── detailed_scores.csv        # 每个样本的详细分数
├── roc_curve.png             # ROC曲线图
└── score_distribution.png    # 分数分布直方图
```

## 🔧 参数说明

```bash
--dataset_path      # HC3数据集路径（默认：datasets/hc3/hc3_english_qa.jsonl）
--human_key         # 人类文本字段名（默认：human_sample）
--chatgpt_key       # ChatGPT文本字段名（默认：chatgpt_generated_text）
--max_samples       # 最大样本数，用于快速测试

--observer          # Observer模型（默认：EleutherAI/gpt-neo-1.3B，1.3B参数）
--performer         # Performer模型（默认：EleutherAI/gpt-neo-2.7B，2.7B参数）
--mode              # 检测模式：accuracy 或 low-fpr（默认：accuracy）
--use_bfloat16      # 使用bfloat16精度（节省内存）

--tokens_seen       # 最大token数（默认：512）
--batch_size        # 批处理大小（默认：16）
--output_dir        # 结果输出目录
```

## 💡 使用示例

### 示例1：使用默认配置（RTX 2070S等8GB显卡）

```bash
# 默认：gpt-neo-1.3B + gpt-neo-2.7B，高性能
python experiments/run_hc3_evaluation.py \
    --max_samples 50
```

### 示例2：中等配置（4-6GB显存）

```bash
# 如果是4-6GB显存，使用中等模型
python experiments/run_hc3_evaluation.py \
    --observer gpt2-medium \
    --performer EleutherAI/gpt-neo-1.3B \
    --batch_size 8
```

### 示例3：使用Falcon模型（原论文配置）

```bash
python experiments/run_hc3_evaluation.py \
    --observer tiiuae/falcon-7b \
    --performer tiiuae/falcon-7b-instruct \
    --batch_size 8 \
    --use_bfloat16
```

### 示例4：自定义GPT-Neo模型

```bash
python experiments/run_hc3_evaluation.py \
    --observer EleutherAI/gpt-neo-1.3B \
    --performer EleutherAI/gpt-neo-2.7B \
    --batch_size 8
```

## 🛠️ 故障排除

### GPU内存不足

```bash
# 减小batch_size
--batch_size 4

# 减少token数
--tokens_seen 256

# 使用更小的模型
--observer gpt2 --performer gpt2-medium

# 启用bfloat16
--use_bfloat16
```

### 模型下载慢

```python
# 设置HuggingFace镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### 数据集加载失败

检查网络连接，或手动下载HC3数据集到本地：

```python
from datasets import load_dataset
dataset = load_dataset("Hello-SimpleAI/HC3")
dataset.to_json("datasets/hc3/hc3_manual.jsonl")
```

## 📈 预期结果

在HC3英文数据集上，Binoculars（使用Falcon-7B配置）的典型性能：

- **Accuracy**: ~85-90%
- **F1-Score**: ~85-88%
- **ROC-AUC**: ~0.92-0.95

使用不同模型配置会有不同的性能表现。

## 🎓 实验流程建议

1. **第一步：快速验证（10分钟）**
   ```bash
   --max_samples 100 --observer gpt2 --performer gpt2-medium
   ```

2. **第二步：中等规模测试（30分钟）**
   ```bash
   --max_samples 1000 --observer gpt2-medium --performer gpt2-large
   ```

3. **第三步：完整评估（1-2小时）**
   ```bash
   # 使用完整数据集和最佳模型
   --observer tiiuae/falcon-7b --performer tiiuae/falcon-7b-instruct
   ```

## 📚 相关文档

- [双模型原理详解](DUAL_MODEL_EXPLANATION.md)
- [完整实现总结](HC3_IMPLEMENTATION_SUMMARY.md)
- [实验指南](HC3_EXPERIMENT_GUIDE.md)

## 🔗 数据集和论文

- **HC3数据集**: [HuggingFace](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
- **HC3论文**: [How Close is ChatGPT to Human Experts?](https://arxiv.org/abs/2301.07597)
- **Binoculars论文**: [Spotting LLMs with Binoculars](https://arxiv.org/abs/2401.12070)

## ✨ 结果示例

评估完成后，控制台输出示例：

```
================================================================================
HC3 Evaluation Results
================================================================================
Dataset: HC3 English
Observer Model: tiiuae/falcon-7b
Performer Model: tiiuae/falcon-7b-instruct
Total Samples: 24322
--------------------------------------------------------------------------------
Accuracy:        0.8765
Precision:       0.8821
Recall:          0.8709
F1-Score:        0.8765
ROC-AUC:         0.9423
TPR@FPR=0.01%:   0.7856
FPR@TPR=95%:     0.0812
================================================================================
```

祝评估顺利！🎉
