"""
HC3数据集加载器
HC3 Dataset Loader for ChatGPT Detection
"""

import os
import sys
import json
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import Dataset, load_dataset
import pandas as pd


class HC3DatasetLoader:
    """
    HC3数据集加载器
    支持从HuggingFace加载HC3-English和HC3-Chinese数据集
    """
    
    def __init__(self, language: str = "english"):
        """
        Args:
            language: "english" 或 "chinese"
        """
        self.language = language.lower()
        if self.language == "english":
            self.dataset_name = "Hello-SimpleAI/HC3"
        elif self.language == "chinese":
            self.dataset_name = "Hello-SimpleAI/HC3-Chinese"
        else:
            raise ValueError(f"Unsupported language: {language}. Choose 'english' or 'chinese'")
    
    def load_dataset(self, split: str = "all", local_path: str = None) -> Dataset:
        """
        加载HC3数据集（支持本地文件）
        
        Args:
            split: "all" 或特定子集
            local_path: 本地parquet文件路径
        
        Returns:
            Dataset对象
        """
        # 优先从本地加载
        if local_path and os.path.exists(local_path):
            print(f"Loading from local file: {local_path}")
            if local_path.endswith('.parquet'):
                dataset = Dataset.from_parquet(local_path)
            elif local_path.endswith('.json') or local_path.endswith('.jsonl'):
                dataset = Dataset.from_json(local_path)
            else:
                raise ValueError(f"Unsupported file format: {local_path}")
            print(f"Loaded {len(dataset)} samples from local file")
            return dataset
        
        # 尝试从本地目录加载
        local_dir = os.path.join("datasets", "hc3")
        local_files = [
            os.path.join(local_dir, "all.jsonl"),  # 完整数据集
            os.path.join(local_dir, "hc3_all.jsonl"),
            os.path.join(local_dir, "open_qa.jsonl"),  # 单个子集
            os.path.join(local_dir, "finance.jsonl"),
        ]
        
        for local_file in local_files:
            if os.path.exists(local_file):
                print(f"Found local file: {local_file}")
                return self.load_dataset(local_path=local_file)
        
        # 如果本地没有，给出下载指引
        print("No local files found.")
        print("\n" + "="*70)
        print("Please download HC3 dataset manually:")
        print("="*70)
        print("\n📥 Download Instructions:")
        print("\n1. Visit: https://huggingface.co/datasets/Hello-SimpleAI/HC3/tree/main")
        print("\n2. Download one of these files (recommended: all.jsonl):")
        print("   - all.jsonl (73.7 MB) - Complete dataset ⭐ RECOMMENDED")
        print("   - open_qa.jsonl (2.91 MB) - Just open QA subset")
        print("   - finance.jsonl (9.89 MB) - Just finance subset")
        print("   - medicine.jsonl (2.68 MB) - Just medicine subset")
        print("   - reddit_eli5.jsonl (55.4 MB) - Just Reddit ELI5 subset")
        print("   - wiki_csai.jsonl (2.2 MB) - Just Wikipedia subset")
        print(f"\n3. Save the file to: {os.path.abspath(local_dir)}/")
        print("\n4. Run this script again")
        print("\n" + "="*70)
        
        raise RuntimeError(
            f"\n❌ HC3 dataset not found locally.\n"
            f"Please download 'all.jsonl' and save to:\n"
            f"{os.path.abspath(local_dir)}/all.jsonl"
        )
    
    def format_for_binoculars(self, dataset: Dataset, qa_mode: bool = True) -> Dataset:
        """
        将HC3数据集格式化为Binoculars实验格式
        
        Args:
            dataset: HC3数据集
            qa_mode: 是否使用问答模式（Question + Answer）
        
        Returns:
            格式化后的Dataset，包含human_text和chatgpt_text字段
        """
        formatted_data = []
        
        for item in dataset:
            # HC3数据集结构：
            # - question: 问题
            # - human_answers: 人类回答列表
            # - chatgpt_answers: ChatGPT回答列表
            
            question = item.get('question', '')
            human_answers = item.get('human_answers', [])
            chatgpt_answers = item.get('chatgpt_answers', [])
            
            # 1:1配对，避免重复
            # 取最小长度，确保每个样本只配对一次
            num_pairs = min(len(human_answers), len(chatgpt_answers))
            
            for i in range(num_pairs):
                human_ans = human_answers[i]
                chatgpt_ans = chatgpt_answers[i]
                
                # 如果使用QA模式，将问题和答案合并
                if qa_mode and question:
                    formatted_data.append({
                        'question': question,
                        'human_text': f"Question: {question}\nAnswer: {human_ans}",
                        'chatgpt_text': f"Question: {question}\nAnswer: {chatgpt_ans}",
                        'human_answer_only': human_ans,
                        'chatgpt_answer_only': chatgpt_ans
                    })
                else:
                    # 仅使用答案部分
                    formatted_data.append({
                        'question': question,
                        'human_text': human_ans,
                        'chatgpt_text': chatgpt_ans,
                        'human_answer_only': human_ans,
                        'chatgpt_answer_only': chatgpt_ans
                    })
        
        return Dataset.from_list(formatted_data)
    
    def create_jsonl_for_experiment(self, 
                                    dataset: Dataset, 
                                    output_path: str,
                                    qa_mode: bool = True,
                                    max_samples: int = None):
        """
        创建用于实验的JSONL文件（类似现有的cc_news-falcon7.jsonl格式）
        
        Args:
            dataset: HC3数据集
            output_path: 输出路径
            qa_mode: 是否使用问答模式
            max_samples: 最大样本数（用于快速测试）
        """
        formatted_ds = self.format_for_binoculars(dataset, qa_mode=qa_mode)
        
        if max_samples:
            formatted_ds = formatted_ds.select(range(min(max_samples, len(formatted_ds))))
        
        # 转换为Binoculars实验格式
        # 每个样本包含human文本和machine（ChatGPT）生成的文本
        output_data = []
        for item in formatted_ds:
            output_data.append({
                "question": item["question"],
                "human_sample": item["human_text"],
                "chatgpt_generated_text": item["chatgpt_text"],
                "human_answer_only": item["human_answer_only"],
                "chatgpt_answer_only": item["chatgpt_answer_only"]
            })
        
        # 写入JSONL文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Created {len(output_data)} samples in {output_path}")
        return output_path
    
    def get_statistics(self, dataset: Dataset) -> Dict:
        """获取数据集统计信息"""
        stats = {
            "total_samples": len(dataset),
            "sample_keys": list(dataset.features.keys()) if dataset else []
        }
        
        if len(dataset) > 0:
            sample = dataset[0]
            if 'human_answers' in sample:
                stats["avg_human_answers"] = sum(len(item.get('human_answers', [])) for item in dataset) / len(dataset)
            if 'chatgpt_answers' in sample:
                stats["avg_chatgpt_answers"] = sum(len(item.get('chatgpt_answers', [])) for item in dataset) / len(dataset)
        
        return stats


def prepare_hc3_for_comparison(language: str = "english", 
                               output_dir: str = "datasets/hc3",
                               qa_mode: bool = True,
                               max_samples: int = None):
    """
    便捷函数：准备HC3数据集用于Binoculars实验
    
    Args:
        language: "english" 或 "chinese"
        output_dir: 输出目录
        qa_mode: 是否使用问答模式
        max_samples: 最大样本数
    """
    import os
    
    loader = HC3DatasetLoader(language=language)
    
    # 加载数据集
    print(f"Loading HC3-{language.capitalize()} dataset...")
    dataset = loader.load_dataset()
    
    # 打印统计信息
    stats = loader.get_statistics(dataset)
    print(f"Dataset statistics: {stats}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建JSONL文件
    mode_suffix = "qa" if qa_mode else "answer_only"
    output_path = os.path.join(output_dir, f"hc3_{language}_{mode_suffix}.jsonl")
    
    loader.create_jsonl_for_experiment(
        dataset=dataset,
        output_path=output_path,
        qa_mode=qa_mode,
        max_samples=max_samples
    )
    
    return output_path


if __name__ == "__main__":
    # 准备HC3英文数据集
    print("Preparing HC3 English dataset...")
    en_path = prepare_hc3_for_comparison(
        language="english",
        output_dir="datasets/hc3",
        qa_mode=True,
        max_samples=None  # 使用全部数据
    )
    
    print(f"\nDataset prepared: {en_path}")
    print("Ready for evaluation!")
