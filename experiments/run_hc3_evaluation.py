"""
在HC3英文数据集上评估Binoculars方法
Evaluate Binoculars on HC3 English Dataset
"""

import os
import sys
import argparse
import datetime
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from datasets import Dataset
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

from binoculars.detector import Binoculars
from experiments.utils import convert_to_pandas


def load_hc3_dataset(dataset_path: str) -> Dataset:
    """加载HC3格式的JSONL数据集"""
    return Dataset.from_json(dataset_path)


def run_binoculars_on_hc3(bino, dataset: Dataset, 
                          human_key: str = "human_sample",
                          chatgpt_key: str = "chatgpt_generated_text",
                          batch_size: int = 32):
    """
    在HC3数据集上运行Binoculars检测
    
    Args:
        bino: Binoculars检测器实例
        dataset: HC3数据集
        human_key: 人类文本的键名
        chatgpt_key: ChatGPT文本的键名
        batch_size: 批处理大小
    
    Returns:
        人类文本分数和ChatGPT文本分数
    """
    print(f"Scoring human text...")
    human_scores = dataset.map(
        lambda batch: {"score": bino.compute_score(batch[human_key])},
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names
    )
    
    print(f"Scoring ChatGPT text...")
    chatgpt_scores = dataset.map(
        lambda batch: {"score": bino.compute_score(batch[chatgpt_key])},
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names
    )
    
    return human_scores, chatgpt_scores


def compute_metrics(human_scores, chatgpt_scores, threshold):
    """
    计算检测指标
    
    Returns:
        字典包含: accuracy, precision, recall, f1, roc_auc等指标
    """
    # 转换为DataFrame
    score_df = convert_to_pandas(human_scores, chatgpt_scores)
    score_df["pred"] = np.where(score_df["score"] < threshold, 1, 0)
    
    # 基础指标
    accuracy = metrics.accuracy_score(score_df["class"], score_df["pred"])
    precision = metrics.precision_score(score_df["class"], score_df["pred"])
    recall = metrics.recall_score(score_df["class"], score_df["pred"])
    f1_score = metrics.f1_score(score_df["class"], score_df["pred"])
    
    # ROC曲线
    score = -1 * score_df["score"]  # 负值使得class 1（AI生成）为正类
    fpr, tpr, thresholds = metrics.roc_curve(y_true=score_df["class"], y_score=score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    
    # TPR @ FPR=0.01%
    tpr_at_fpr_001 = np.interp(0.01 / 100, fpr, tpr)
    
    # FPR @ TPR=95%
    fpr_at_tpr_95 = np.interp(0.95, tpr, fpr)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "roc_auc": roc_auc,
        "tpr_at_fpr_0.01": tpr_at_fpr_001,
        "fpr_at_tpr_95": fpr_at_tpr_95,
        "score_df": score_df,
        "fpr": fpr,
        "tpr": tpr
    }


def save_results(result: dict, output_dir: str, args):
    """保存评估结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存指标摘要
    metrics_summary = {
        "Dataset": "HC3 English",
        "Observer Model": args.observer,
        "Performer Model": args.performer,
        "Total Samples": len(result['score_df']),
        "Threshold": args.threshold if hasattr(args, 'threshold') and args.threshold else "auto",
        "Metrics": {
            "Accuracy": f"{result['accuracy']:.4f}",
            "Precision": f"{result['precision']:.4f}",
            "Recall": f"{result['recall']:.4f}",
            "F1-Score": f"{result['f1']:.4f}",
            "ROC-AUC": f"{result['roc_auc']:.4f}",
            "TPR@FPR=0.01%": f"{result['tpr_at_fpr_0.01']:.4f}",
            "FPR@TPR=95%": f"{result['fpr_at_tpr_95']:.4f}"
        }
    }
    
    # 打印到控制台
    print(f"\n{'='*80}")
    print("HC3 Evaluation Results")
    print(f"{'='*80}")
    print(f"Dataset: HC3 English")
    print(f"Observer Model: {args.observer}")
    print(f"Performer Model: {args.performer}")
    print(f"Total Samples: {len(result['score_df'])}")
    print(f"{'-'*80}")
    print(f"Accuracy:        {result['accuracy']:.4f}")
    print(f"Precision:       {result['precision']:.4f}")
    print(f"Recall:          {result['recall']:.4f}")
    print(f"F1-Score:        {result['f1']:.4f}")
    print(f"ROC-AUC:         {result['roc_auc']:.4f}")
    print(f"TPR@FPR=0.01%:   {result['tpr_at_fpr_0.01']:.4f}")
    print(f"FPR@TPR=95%:     {result['fpr_at_tpr_95']:.4f}")
    print(f"{'='*80}\n")
    
    # 保存JSON
    with open(os.path.join(output_dir, "metrics_summary.json"), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    # 保存详细分数CSV
    result['score_df'].to_csv(os.path.join(output_dir, "detailed_scores.csv"), index=False)
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(result['fpr'], result['tpr'], 
             label=f"Binoculars (AUC = {result['roc_auc']:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve on HC3 English Dataset', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {os.path.join(output_dir, 'roc_curve.png')}")
    
    # 绘制分数分布
    plt.figure(figsize=(10, 6))
    human_scores = result['score_df'][result['score_df']['class'] == 0]['score']
    ai_scores = result['score_df'][result['score_df']['class'] == 1]['score']
    
    plt.hist(human_scores, bins=50, alpha=0.6, label='Human', color='blue')
    plt.hist(ai_scores, bins=50, alpha=0.6, label='AI (ChatGPT)', color='red')
    plt.xlabel('Binoculars Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Score Distribution on HC3 Dataset', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=300, bbox_inches='tight')
    print(f"Score distribution saved to {os.path.join(output_dir, 'score_distribution.png')}")
    
    print(f"\nAll results saved to: {output_dir}")


def main(args):
    """主评估函数"""
    print(f"{'='*80}")
    print(f"Binoculars Evaluation on HC3 English Dataset")
    print(f"{'='*80}\n")
    
    # 加载HC3数据集
    print(f"Loading HC3 dataset from {args.dataset_path}...")
    dataset = load_hc3_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} samples\n")
    
    # 如果指定了样本数限制
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        print(f"Limited to {args.max_samples} samples for evaluation\n")
    
    # 初始化Binoculars
    print(f"Initializing Binoculars...")
    print(f"Observer Model: {args.observer}")
    print(f"Performer Model: {args.performer}")
    print(f"Mode: {args.mode}")
    print(f"Max Tokens: {args.tokens_seen}\n")
    
    bino = Binoculars(
        observer_name_or_path=args.observer,
        performer_name_or_path=args.performer,
        mode=args.mode,
        max_token_observed=args.tokens_seen,
        use_bfloat16=args.use_bfloat16
    )
    
    # 运行评估
    print(f"Running evaluation...")
    human_scores, chatgpt_scores = run_binoculars_on_hc3(
        bino, dataset, 
        human_key=args.human_key,
        chatgpt_key=args.chatgpt_key,
        batch_size=args.batch_size
    )
    
    # 计算指标
    print(f"\nComputing metrics...")
    result = compute_metrics(human_scores, chatgpt_scores, bino.threshold)
    
    # 保存结果
    output_dir = args.output_dir or f"results/hc3_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_results(result, output_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Binoculars on HC3 English dataset"
    )
    
    # 数据集参数
    parser.add_argument("--dataset_path", type=str, 
                       default="datasets/hc3/hc3_english_qa.jsonl",
                       help="Path to HC3 JSONL file")
    parser.add_argument("--human_key", type=str, default="human_sample",
                       help="Key for human-generated text")
    parser.add_argument("--chatgpt_key", type=str, default="chatgpt_generated_text",
                       help="Key for ChatGPT-generated text")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples (for testing)")
    
    # Binoculars模型参数
    parser.add_argument("--observer", type=str, 
                       default="gpt2-medium",
                       help="Observer model (default: gpt2-medium - 355M params)")
    parser.add_argument("--performer", type=str,
                       default="gpt2-large",
                       help="Performer model (default: gpt2-large - 774M params)")
    parser.add_argument("--mode", type=str, default="accuracy",
                       choices=["accuracy", "low-fpr"],
                       help="Detection mode")
    parser.add_argument("--use_bfloat16", action="store_true",
                       help="Use bfloat16 precision")
    
    # 计算参数
    parser.add_argument("--tokens_seen", type=int, default=256,
                       help="Maximum tokens to process (default: 256 for 8GB GPU)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing (default: 4 for 8GB GPU, reduce to 2 if OOM)")
    
    # 输出
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Type: {torch.cuda.get_device_name(0)}\n")
    
    main(args)
