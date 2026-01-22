# HC3数据集对比实验指南

## 项目概述

本项目旨在比较Binoculars方法在HC3数据集上的表现，包括：
1. **原始Binoculars方法**：使用Falcon-7B和Falcon-7B-Instruct
2. **自定义方法**：使用你选择的模型对（如不同的RoBERTa变体或GPT系列模型）

## 关于Observer和Performer双模型的问题

### ✅ **是的，你需要使用两个不同的模型**

**为什么需要两个模型？**

Binoculars的核心思想是：
- **Performer模型**：计算文本的困惑度（Perplexity, PPL）- 评估文本流畅度
- **Observer模型**：与Performer一起计算交叉熵（Cross-Entropy）- 评估两模型对文本理解的差异
- **Binoculars Score = PPL / Cross-Entropy**

关键洞察：
- AI生成的文本在两个模型上的表现会**更一致**（因为它们训练数据重叠）
- 人类文本在两个模型上会有**更大差异**

如果使用相同模型，cross-entropy就等于perplexity，检测能力就会丧失！

### 推荐的模型配置

#### 方案1：使用不同大小的GPT模型（推荐）
```python
# Observer: 较小模型
observer = "gpt2"  # 124M参数

# Performer: 较大或指令微调模型  
performer = "gpt2-medium"  # 355M参数
# 或
performer = "gpt2-large"   # 774M参数
```

#### 方案2：使用中文模型
```python
# Observer: 基础中文GPT
observer = "uer/gpt2-chinese-cluecorpussmall"

# Performer: 更大或微调过的中文模型
performer = "uer/gpt2-chinese-cluecorpussmall"  # 可以找对话微调版本
```

#### 方案3：原始Falcon配置
```python
observer = "tiiuae/falcon-7b"
performer = "tiiuae/falcon-7b-instruct"  # 在指令数据上微调过
```

**重要提示**：
- RoBERTa是**Masked Language Model (MLM)**，不是Causal Language Model
- Binoculars需要因果语言模型（能预测下一个词的模型）
- 建议使用GPT-2、GPT-Neo、LLaMA、Falcon等因果模型

## 安装依赖

```bash
pip install datasets transformers torch sklearn matplotlib pandas
```

## 使用步骤

### 第一步：准备HC3数据集

```bash
# 下载并格式化HC3英文数据集
python experiments/hc3_loader.py
```

这将创建：
- `datasets/hc3/hc3_english_qa.jsonl` - 英文问答格式
- `datasets/hc3/hc3_chinese_qa.jsonl` - 中文问答格式

你也可以在Python中手动准备：

```python
from experiments.hc3_loader import prepare_hc3_for_comparison

# 准备英文数据（1000个样本用于快速测试）
en_path = prepare_hc3_for_comparison(
    language="english",
    output_dir="datasets/hc3",
    qa_mode=True,  # True: 问题+答案, False: 仅答案
    max_samples=1000
)

# 准备完整数据集
en_full = prepare_hc3_for_comparison(
    language="english",
    output_dir="datasets/hc3",
    qa_mode=True,
    max_samples=None  # 使用全部数据
)
```

### 第二步：运行对比实验

#### 快速测试（推荐先运行这个）

```bash
# 使用小模型快速测试
python experiments/run_hc3_comparison.py \
    --dataset_path datasets/hc3/hc3_english_qa.jsonl \
    --max_samples 100 \
    --custom_observer gpt2 \
    --custom_performer gpt2-medium \
    --batch_size 8
```

#### 完整实验：对比原始方法和自定义方法

```bash
python experiments/run_hc3_comparison.py \
    --dataset_path datasets/hc3/hc3_english_qa.jsonl \
    --run_original \
    --run_custom \
    --original_observer tiiuae/falcon-7b \
    --original_performer tiiuae/falcon-7b-instruct \
    --custom_observer gpt2 \
    --custom_performer gpt2-medium \
    --batch_size 16 \
    --output_dir results/hc3_gpt2_vs_falcon
```

#### 仅运行自定义模型

```bash
python experiments/run_hc3_comparison.py \
    --dataset_path datasets/hc3/hc3_english_qa.jsonl \
    --run_custom \
    --custom_observer gpt2 \
    --custom_performer gpt2-large \
    --batch_size 16
```

#### 中文实验

```bash
python experiments/run_hc3_comparison.py \
    --dataset_path datasets/hc3/hc3_chinese_qa.jsonl \
    --run_custom \
    --custom_observer uer/gpt2-chinese-cluecorpussmall \
    --custom_performer uer/gpt2-chinese-cluecorpussmall \
    --batch_size 16
```

### 第三步：查看结果

实验完成后，会在输出目录生成：

1. **metrics_comparison.csv** - 指标对比表格
   ```
   Method              Accuracy  Precision  Recall  F1-Score  ROC-AUC
   Original Binoculars  0.8523    0.8621    0.8425   0.8522    0.9234
   Custom Model         0.8612    0.8701    0.8523   0.8611    0.9312
   ```

2. **roc_comparison.png** - ROC曲线对比图

3. **results_summary.json** - 详细结果JSON

## 参数说明

```bash
--dataset_path       # HC3 JSONL文件路径
--human_key          # 人类文本的字段名 (默认: human_sample)
--chatgpt_key        # ChatGPT文本的字段名 (默认: chatgpt_generated_text)
--max_samples        # 最大样本数（用于快速测试）

--run_original       # 运行原始Binoculars
--run_custom         # 运行自定义模型

--original_observer  # 原始方法的Observer模型
--original_performer # 原始方法的Performer模型
--custom_observer    # 自定义方法的Observer模型
--custom_performer   # 自定义方法的Performer模型

--tokens_seen        # 最大token数 (默认: 512)
--batch_size         # 批处理大小 (默认: 16)
--output_dir         # 结果输出目录
```

## 实验建议

### 1. 渐进式实验策略

**第一阶段：验证流程**
```bash
# 使用少量样本和小模型快速验证
python experiments/run_hc3_comparison.py \
    --dataset_path datasets/hc3/hc3_english_qa.jsonl \
    --max_samples 50 \
    --custom_observer gpt2 \
    --custom_performer gpt2-medium \
    --batch_size 4
```

**第二阶段：模型对比**
```bash
# 测试不同模型组合（样本量适中）
python experiments/run_hc3_comparison.py \
    --dataset_path datasets/hc3/hc3_english_qa.jsonl \
    --max_samples 500 \
    --run_custom \
    --custom_observer gpt2 \
    --custom_performer gpt2-large \
    --batch_size 16
```

**第三阶段：完整评估**
```bash
# 使用完整数据集和最佳模型配置
python experiments/run_hc3_comparison.py \
    --dataset_path datasets/hc3/hc3_english_qa.jsonl \
    --run_original \
    --run_custom \
    --original_observer tiiuae/falcon-7b \
    --original_performer tiiuae/falcon-7b-instruct \
    --custom_observer gpt2 \
    --custom_performer gpt2-large \
    --batch_size 32
```

### 2. 模型选择建议

**轻量级（快速实验）**
- Observer: `gpt2` (124M)
- Performer: `gpt2-medium` (355M)

**中等规模（平衡性能）**
- Observer: `gpt2-medium` (355M)
- Performer: `gpt2-large` (774M)

**大规模（最佳性能，需要GPU）**
- Observer: `EleutherAI/gpt-neo-1.3B`
- Performer: `EleutherAI/gpt-neo-2.7B`

**中文模型**
- Observer: `uer/gpt2-chinese-cluecorpussmall`
- Performer: 寻找对话微调版本或使用更大的中文GPT模型

### 3. GPU内存优化

如果遇到GPU内存不足：

```bash
# 减小batch_size
--batch_size 4

# 减少max_token_observed
--tokens_seen 256

# 使用更小的模型
--custom_observer gpt2 \
--custom_performer gpt2-medium
```

## 预期结果

实验完成后，你将得到：

1. **性能指标对比**：
   - Accuracy（准确率）
   - Precision（精确率）
   - Recall（召回率）
   - F1-Score
   - ROC-AUC
   - TPR @ FPR=0.01%（在极低误报率下的真正例率）

2. **可视化对比**：
   - ROC曲线对比图
   - 便于直观比较不同方法的性能

3. **数据集适应性分析**：
   - 评估原始Binoculars在HC3数据集上的表现
   - 评估你的自定义模型配置的效果
   - 为后续优化提供依据

## 常见问题

### Q1: RoBERTa能用于Binoculars吗？

**不能直接使用**。RoBERTa是Masked Language Model，而Binoculars需要Causal Language Model（因果语言模型）。建议使用：
- GPT-2系列
- GPT-Neo系列
- LLaMA系列
- Falcon系列

### Q2: 为什么一定要两个不同的模型？

因为Binoculars的核心是测量两个模型对同一文本的**理解差异**：
- 如果是同一模型，差异为0，失去检测能力
- 两个模型有差异时，AI文本的差异通常小于人类文本

### Q3: Observer和Performer应该选哪个大哪个小？

**推荐**: Observer较小，Performer较大或经过特殊微调

原因：
- Performer负责评估文本质量（困惑度）
- Observer提供参考视角
- 原始论文使用Falcon-7B（observer）和Falcon-7B-Instruct（performer，指令微调）

### Q4: HC3数据集是否需要预处理？

`hc3_loader.py`已经处理了：
- 将问题和答案合并（QA模式）
- 或仅使用答案部分
- 格式化为Binoculars实验格式

你可以根据需要选择不同的模式。

## 下一步优化

1. **阈值优化**：在HC3数据集上重新优化检测阈值
2. **模型微调**：在HC3训练集上微调模型以提升性能
3. **集成方法**：结合多个模型的结果
4. **特征工程**：添加额外的语言学特征

## 联系与支持

如有问题，请查看原始Binoculars论文：
- 论文：https://arxiv.org/abs/2401.12070
- HC3数据集：https://huggingface.co/datasets/Hello-SimpleAI/HC3
