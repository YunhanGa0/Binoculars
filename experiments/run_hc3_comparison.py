"""
HC3数据集上的Binoculars vs RoBERTa方法对比实验
Comparison experiments on HC3 dataset: Original Binoculars vs RoBERTa-based approach
"""

import os
import argparse
import datetime
import json

import torch
import numpy as np
from datasets import Dataset
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

from binoculars.detector import Binoculars
from experiments.utils import convert_to_pandas, save_experiment


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
        字典包含: accuracy, precision, recall, f1, roc_auc, fpr_at_tpr_95
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
    tpr_at_fpr_0_01 = np.interp(0.01 / 100, fpr, tpr)
    
    # FPR @ TPR=95%
    fpr_at_tpr_95 = np.interp(0.95, tpr, fpr)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "roc_auc": roc_auc,
        "tpr_at_fpr_0.01": tpr_at_fpr_0_01,
        "fpr_at_tpr_95": fpr_at_tpr_95,
        "score_df": score_df,
        "fpr": fpr,
        "tpr": tpr
    }


def save_comparison_results(results_dict: dict, output_dir: str):
    """保存对比实验结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存指标对比表
    metrics_comparison = []
    for method_name, result in results_dict.items():
        metrics_comparison.append({
            "Method": method_name,
            "Accuracy": f"{result['accuracy']:.4f}",
            "Precision": f"{result['precision']:.4f}",
            "Recall": f"{result['recall']:.4f}",
            "F1-Score": f"{result['f1']:.4f}",
            "ROC-AUC": f"{result['roc_auc']:.4f}",
            "TPR@FPR=0.01%": f"{result['tpr_at_fpr_0.01']:.4f}",
            "FPR@TPR=95%": f"{result['fpr_at_tpr_95']:.4f}"
        })
    
    df_comparison = pd.DataFrame(metrics_comparison)
    df_comparison.to_csv(os.path.join(output_dir, "metrics_comparison.csv"), index=False)
    print(f"\n{'='*80}")
    print("Metrics Comparison:")
    print(df_comparison.to_string(index=False))
    print(f"{'='*80}\n")
    
    # 绘制ROC曲线对比
    plt.figure(figsize=(10, 8))
    for method_name, result in results_dict.items():
        plt.plot(result['fpr'], result['tpr'], 
                label=f"{method_name} (AUC = {result['roc_auc']:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison on HC3 Dataset')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "roc_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {os.path.join(output_dir, 'roc_comparison.png')}")
    
    # 保存详细JSON结果
    json_results = {}
    for method_name, result in results_dict.items():
        json_results[method_name] = {
            k: v for k, v in result.items() 
            if k not in ['score_df', 'fpr', 'tpr']  # 排除大数组
        }
    
    with open(os.path.join(output_dir, "results_summary.json"), 'w') as f:
        json.dump(json_results, f, indent=2)


def main(args):
    """主实验函数"""
    print(f"{'='*80}")
    print(f"HC3 Dataset Comparison Experiment")
    print(f"{'='*80}\n")
    
    # 加载HC3数据集
    print(f"Loading HC3 dataset from {args.dataset_path}...")
    dataset = load_hc3_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} samples\n")
    
    # 如果指定了样本数限制
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        print(f"Limited to {args.max_samples} samples for quick testing\n")
    
    results = {}
    
    # 1. 运行原始Binoculars（使用Falcon模型）
    if args.run_original:
        print(f"\n{'='*80}")
        print("Running Original Binoculars (Falcon-based)")
        print(f"{'='*80}\n")
        
        bino_original = Binoculars(
            observer_name_or_path=args.original_observer,
            performer_name_or_path=args.original_performer,
            mode="accuracy",
            max_token_observed=args.tokens_seen
        )
        
        human_scores, chatgpt_scores = run_binoculars_on_hc3(
            bino_original, dataset, 
            human_key=args.human_key,
            chatgpt_key=args.chatgpt_key,
            batch_size=args.batch_size
        )
        
        results["Original Binoculars"] = compute_metrics(
            human_scores, chatgpt_scores, 
            bino_original.threshold
        )
    
    # 2. 运行RoBERTa/自定义模型版本
    if args.run_custom:
        print(f"\n{'='*80}")
        print(f"Running Custom Binoculars")
        print(f"Observer: {args.custom_observer}")
        print(f"Performer: {args.custom_performer}")
        print(f"{'='*80}\n")
        
        bino_custom = Binoculars(
            observer_name_or_path=args.custom_observer,
            performer_name_or_path=args.custom_performer,
            mode="accuracy",
            max_token_observed=args.tokens_seen
        )
        
        human_scores, chatgpt_scores = run_binoculars_on_hc3(
            bino_custom, dataset,
            human_key=args.human_key,
            chatgpt_key=args.chatgpt_key,
            batch_size=args.batch_size
        )
        
        results["Custom Model"] = compute_metrics(
            human_scores, chatgpt_scores,
            bino_custom.threshold
        )
    
    # 保存对比结果
    output_dir = args.output_dir or f"results/hc3_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_comparison_results(results, output_dir)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Binoculars methods on HC3 dataset"
    )
    
    # 数据集参数
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to HC3 JSONL file")
    parser.add_argument("--human_key", type=str, default="human_sample",
                       help="Key for human-generated text")
    parser.add_argument("--chatgpt_key", type=str, default="chatgpt_generated_text",
                       help="Key for ChatGPT-generated text")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples (for testing)")
    
    # 实验选项
    parser.add_argument("--run_original", action="store_true",
                       help="Run original Binoculars (Falcon-based)")
    parser.add_argument("--run_custom", action="store_true",
                       help="Run custom model Binoculars")
    
    # 原始Binoculars模型
    parser.add_argument("--original_observer", type=str, 
                       default="tiiuae/falcon-7b",
                       help="Observer model for original Binoculars")
    parser.add_argument("--original_performer", type=str,
                       default="tiiuae/falcon-7b-instruct",
                       help="Performer model for original Binoculars")
    
    # 自定义模型（RoBERTa或其他）
    parser.add_argument("--custom_observer", type=str,
                       default="gpt2",
                       help="Observer model for custom approach")
    parser.add_argument("--custom_performer", type=str,
                       default="gpt2-medium",
                       help="Performer model for custom approach")
    
    # 计算参数
    parser.add_argument("--tokens_seen", type=int, default=512,
                       help="Maximum tokens to process")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for processing")
    
    # 输出
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # 至少运行一个方法
    if not args.run_original and not args.run_custom:
        args.run_original = True
        args.run_custom = True
        print("Running both original and custom methods by default\n")
    
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Type: {torch.cuda.get_device_name(0)}\n")
    
    main(args)
