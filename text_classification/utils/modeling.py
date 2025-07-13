"""
모델 관련 모듈 (utils/modeling.py)
================================

이 모듈은 텍스트 분류를 위한 모델 생성, 설정, 관리 기능을 제공합니다.
학생들이 다양한 모델 아키텍처와 학습 방법을 이해할 수 있도록 구성되어 있습니다.

주요 기능:
- 사전 훈련된 모델 로딩
- LoRA (Low-Rank Adaptation) 적용
- Linear Probing 설정
- 모델 저장 및 로딩
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    모델 설정을 위한 데이터 클래스
    
    이 클래스는 모델의 다양한 설정을 한 곳에서 관리할 수 있게 해줍니다.
    """
    pretrained_model_name: str = "skt/kobert-base-v1"  # 기본 한국어 BERT 모델
    num_labels: int = 3  # 분류할 레이블 개수
    linear_probing: bool = False  # Linear Probing 사용 여부
    use_lora: bool = True  # LoRA 사용 여부
    lora_r: int = 8  # LoRA의 rank
    lora_alpha: int = 16  # LoRA의 alpha 값
    lora_dropout: float = 0.1  # LoRA의 dropout 비율
    max_length: int = 128  # 최대 시퀀스 길이

def get_target_modules(model_type: str) -> List[str]:
    """
    모델 타입에 따라 LoRA를 적용할 타겟 모듈을 결정합니다.
    
    Args:
        model_type: 모델 타입 (예: 'bert', 'gpt2')
        
    Returns:
        target_modules: LoRA를 적용할 모듈 이름 리스트
    """
    if model_type in {"bert", "kobert"}:  # BERT 계열 모델
        return ["query", "key", "value"]
    elif model_type == "gpt2":  # GPT-2 모델
        return ["c_attn"]
    else:  # 기타 모델들
        return ["q_proj", "v_proj"]

def build_model(config: ModelConfig):
    """
    설정에 따라 모델을 생성합니다.
    
    Args:
        config: 모델 설정
        
    Returns:
        model: 생성된 모델
        tokenizer: 토크나이저
    """
    print(f"모델 생성 중: {config.pretrained_model_name}")
    
    # 1. 토크나이저 로딩
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name, use_fast=False)
    
    # 2. GPT-2 모델의 경우 pad 토큰 추가
    added_pad = False
    if tokenizer.pad_token is None and "gpt2" in config.pretrained_model_name.lower():
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        added_pad = True
        print("GPT-2 모델에 pad 토큰 추가됨")
    
    # 3. 모델 설정 및 로딩
    model_config = AutoConfig.from_pretrained(
        config.pretrained_model_name, 
        num_labels=config.num_labels
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model_name, 
        config=model_config
    )
    
    # 4. Linear Probing 설정 (백본 모델 고정)
    if config.linear_probing:
        print("Linear Probing 모드: 백본 모델 파라미터 고정")
        freeze_backbone_parameters(model)
    
    # 5. LoRA 적용
    if config.use_lora:
        print("LoRA 적용 중...")
        target_modules = get_target_modules(model_config.model_type)
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=target_modules,
        )
        
        model = get_peft_model(model, lora_config)
        print(f"LoRA 적용 완료 (target_modules: {target_modules})")
    
    # 6. 토큰 추가로 인한 임베딩 크기 조정
    if added_pad:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        print("임베딩 크기 조정 완료")
    
    print("모델 생성 완료")
    return model, tokenizer

def freeze_backbone_parameters(model):
    """
    백본 모델의 파라미터를 고정합니다 (Linear Probing용).
    
    Args:
        model: 고정할 모델
    """
    # BERT 계열 모델의 경우
    if hasattr(model, 'bert'):
        for param in model.bert.parameters():
            param.requires_grad = False
    # GPT-2 계열 모델의 경우
    elif hasattr(model, 'transformer'):
        for param in model.transformer.parameters():
            param.requires_grad = False
    # 기타 모델의 경우
    else:
        for name, param in model.named_parameters():
            if "classifier" not in name and "score" not in name:
                param.requires_grad = False

def load_model(model_path: str, model_name: str, num_labels: int, use_lora: bool = False):
    """
    저장된 모델을 로딩합니다.
    
    Args:
        model_path: 모델 저장 경로
        model_name: 사전 훈련된 모델 이름
        num_labels: 레이블 개수
        use_lora: LoRA 모델 여부
        
    Returns:
        model: 로딩된 모델
        tokenizer: 토크나이저
    """
    print(f"모델 로딩 중: {model_path}")
    
    # 토크나이저 로딩
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 기본 모델 로딩
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    
    # LoRA 모델인 경우
    if use_lora:
        model = PeftModel.from_pretrained(model, model_path)
        print("LoRA 모델 로딩 완료")
    else:
        # 전체 모델 로딩
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("전체 모델 로딩 완료")
    
    return model, tokenizer

def save_model(model, tokenizer, save_path: str, model_name: str):
    """
    모델을 저장합니다.
    
    Args:
        model: 저장할 모델
        tokenizer: 토크나이저
        save_path: 저장 경로
        model_name: 모델 이름
    """
    print(f"모델 저장 중: {save_path}")
    
    # 모델 저장
    model.save_pretrained(save_path)
    
    # 토크나이저 저장
    tokenizer.save_pretrained(save_path)
    
    print("모델 저장 완료")

def get_model_info(model):
    """
    모델의 정보를 출력합니다 (교육용).
    
    Args:
        model: 정보를 확인할 모델
    """
    print("모델 정보:")
    print(f"  총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  학습 가능한 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # LoRA 모델인 경우 추가 정보
    if hasattr(model, 'peft_config'):
        print("  LoRA 모델: 예")
        for name, config in model.peft_config.items():
            print(f"  LoRA 설정: {config}")
    else:
        print("  LoRA 모델: 아니오")

def count_parameters(model):
    """
    모델의 파라미터 수를 계산합니다.
    
    Args:
        model: 파라미터를 계산할 모델
        
    Returns:
        total_params: 총 파라미터 수
        trainable_params: 학습 가능한 파라미터 수
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params 