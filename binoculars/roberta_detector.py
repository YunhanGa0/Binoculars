"""
RoBERTa-based Binoculars Detector for HC3 Dataset
使用RoBERTa模型的Binoculars检测器，用于HC3数据集
"""

from typing import Union
import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaForCausalLM, RobertaTokenizer

from .utils import assert_tokenizer_consistency
from .metrics import perplexity, entropy

torch.set_grad_enabled(False)

huggingface_config = {
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

# Thresholds - 这些需要在HC3数据集上重新优化
ROBERTA_ACCURACY_THRESHOLD = 0.9015310749276843  # 需要重新计算
ROBERTA_FPR_THRESHOLD = 0.8536432310785527  # 需要重新计算

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class RobertaBinoculars(object):
    """
    基于RoBERTa的Binoculars检测器
    
    Args:
        observer_name_or_path: Observer模型路径 (例如: "roberta-base" 或 "hfl/chinese-roberta-wwm-ext")
        performer_name_or_path: Performer模型路径 (例如: "roberta-large" 或微调过的RoBERTa变体)
        use_bfloat16: 是否使用bfloat16精度
        max_token_observed: 最大token数量
        mode: 检测模式 ("low-fpr" 或 "accuracy")
    """
    
    def __init__(self,
                 observer_name_or_path: str = "roberta-base",
                 performer_name_or_path: str = "roberta-large",
                 use_bfloat16: bool = False,
                 max_token_observed: int = 512,
                 mode: str = "accuracy",
                 model_type: str = "roberta"  # "roberta" 或 "causal-lm"
                 ) -> None:
        
        self.model_type = model_type
        self.change_mode(mode)
        
        # RoBERTa不是因果语言模型，但我们可以将其适配为类似的用途
        # 或者使用GPT-2/GPT-Neo等因果模型
        if model_type == "roberta":
            # 注意：标准RoBERTa是MLM模型，不是CLM
            # 如果要用于Binoculars，可能需要使用专门的因果版本或GPT系列
            print("Warning: RoBERTa is a masked language model, not causal. Consider using GPT-2 or similar.")
            self.observer_model = AutoModelForCausalLM.from_pretrained(
                observer_name_or_path,
                device_map={"": DEVICE_1},
                torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                token=huggingface_config["TOKEN"],
                trust_remote_code=True
            )
            self.performer_model = AutoModelForCausalLM.from_pretrained(
                performer_name_or_path,
                device_map={"": DEVICE_2},
                torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                token=huggingface_config["TOKEN"],
                trust_remote_code=True
            )
        else:
            # 使用标准的因果语言模型
            self.observer_model = AutoModelForCausalLM.from_pretrained(
                observer_name_or_path,
                device_map={"": DEVICE_1},
                torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                token=huggingface_config["TOKEN"]
            )
            self.performer_model = AutoModelForCausalLM.from_pretrained(
                performer_name_or_path,
                device_map={"": DEVICE_2},
                torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                token=huggingface_config["TOKEN"]
            )
        
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = ROBERTA_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = ROBERTA_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        """计算Binoculars分数"""
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        """预测文本是否为AI生成"""
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated")
        return pred.tolist() if isinstance(input_text, list) else pred.item()
