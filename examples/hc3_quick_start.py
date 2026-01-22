"""
简单示例：如何使用HC3数据集和Binoculars进行实验
Quick Start Example for HC3 + Binoculars Experiment
"""

import os
import sys

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.hc3_loader import HC3DatasetLoader, prepare_hc3_for_comparison
from binoculars import Binoculars


def example_1_load_hc3_dataset():
    """示例1：加载和查看HC3数据集"""
    print("="*80)
    print("示例1：加载HC3数据集")
    print("="*80)
    
    # 创建加载器
    loader = HC3DatasetLoader(language="english")
    
    # 加载数据集
    print("正在加载HC3-English数据集...")
    dataset = loader.load_dataset(split="all")
    
    # 查看统计信息
    stats = loader.get_statistics(dataset)
    print(f"\n数据集统计信息：")
    print(f"  - 总样本数: {stats['total_samples']}")
    print(f"  - 字段名: {stats['sample_keys']}")
    
    # 查看第一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n第一个样本示例：")
        print(f"  - 问题: {sample.get('question', 'N/A')[:100]}...")
        print(f"  - 人类回答数: {len(sample.get('human_answers', []))}")
        print(f"  - ChatGPT回答数: {len(sample.get('chatgpt_answers', []))}")
    
    print("\n✓ 示例1完成\n")


def example_2_prepare_dataset_for_experiment():
    """示例2：准备实验用数据集"""
    print("="*80)
    print("示例2：准备实验数据集（100个样本用于快速测试）")
    print("="*80)
    
    # 准备小规模数据集用于快速测试
    output_path = prepare_hc3_for_comparison(
        language="english",
        output_dir="datasets/hc3",
        qa_mode=True,  # 包含问题和答案
        max_samples=100
    )
    
    print(f"\n✓ 数据集已准备完毕: {output_path}")
    print("\n✓ 示例2完成\n")


def example_3_quick_detection_test():
    """示例3：快速检测测试"""
    print("="*80)
    print("示例3：使用Binoculars进行快速检测测试")
    print("="*80)
    
    # 初始化Binoculars（使用小模型进行快速测试）
    print("正在加载模型...")
    print("  - Observer: gpt2")
    print("  - Performer: gpt2-medium")
    
    bino = Binoculars(
        observer_name_or_path="gpt2",
        performer_name_or_path="gpt2-medium",
        mode="accuracy"
    )
    
    # 测试样本
    human_text = """
    The history of artificial intelligence began in antiquity with myths, stories and rumors 
    of artificial beings endowed with intelligence or consciousness by master craftsmen. 
    The seeds of modern AI were planted by classical philosophers who attempted to describe 
    human thinking as a symbolic system.
    """
    
    ai_text = """
    Artificial Intelligence (AI) is a fascinating field that has revolutionized the way we 
    interact with technology. It encompasses machine learning, neural networks, and deep learning, 
    enabling computers to perform tasks that typically require human intelligence. From virtual 
    assistants to autonomous vehicles, AI continues to shape our modern world in unprecedented ways.
    """
    
    # 进行检测
    print("\n检测人类文本...")
    human_score = bino.compute_score(human_text.strip())
    human_pred = bino.predict(human_text.strip())
    print(f"  - 分数: {human_score:.4f}")
    print(f"  - 预测: {human_pred}")
    
    print("\n检测AI文本...")
    ai_score = bino.compute_score(ai_text.strip())
    ai_pred = bino.predict(ai_text.strip())
    print(f"  - 分数: {ai_score:.4f}")
    print(f"  - 预测: {ai_pred}")
    
    print("\n✓ 示例3完成\n")


def example_4_batch_detection():
    """示例4：批量检测"""
    print("="*80)
    print("示例4：批量检测多个文本")
    print("="*80)
    
    # 准备测试文本
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence that focuses on data analysis.",
        "I went to the park yesterday and saw beautiful flowers blooming everywhere.",
        "The implementation of neural networks requires careful consideration of architecture design."
    ]
    
    print("正在加载模型...")
    bino = Binoculars(
        observer_name_or_path="gpt2",
        performer_name_or_path="gpt2-medium"
    )
    
    print("\n批量检测结果：")
    scores = bino.compute_score(texts)
    predictions = bino.predict(texts)
    
    for i, (text, score, pred) in enumerate(zip(texts, scores, predictions)):
        print(f"\n文本 {i+1}: {text[:60]}...")
        print(f"  分数: {score:.4f}")
        print(f"  预测: {pred}")
    
    print("\n✓ 示例4完成\n")


def main():
    """主函数：运行所有示例"""
    print("\n" + "="*80)
    print("HC3数据集 + Binoculars 快速入门示例")
    print("="*80 + "\n")
    
    try:
        # 示例1：加载HC3数据集
        example_1_load_hc3_dataset()
        
        # 示例2：准备实验数据
        example_2_prepare_dataset_for_experiment()
        
        # 示例3：快速检测测试
        example_3_quick_detection_test()
        
        # 示例4：批量检测
        example_4_batch_detection()
        
        print("="*80)
        print("所有示例运行完成！")
        print("="*80)
        print("\n下一步：")
        print("1. 运行完整对比实验：")
        print("   python experiments/run_hc3_comparison.py --dataset_path datasets/hc3/hc3_english_qa.jsonl")
        print("\n2. 或使用快速启动脚本：")
        print("   Windows: run_hc3_experiment.bat")
        print("   Linux/Mac: bash run_hc3_experiment.sh")
        print("\n3. 查看详细指南：HC3_EXPERIMENT_GUIDE.md")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保已安装所有依赖：")
        print("  pip install datasets transformers torch sklearn matplotlib pandas")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
