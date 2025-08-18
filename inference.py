"""
텍스트 분류 모델 추론 스크립트 (inference.py)
==========================================

이 스크립트는 훈련된 텍스트 분류 모델을 사용하여 새로운 텍스트를 분류합니다.
LoRA, 4-bit 양자화, 동적 양자화 모델을 모두 지원합니다.

사용법:
    python inference.py

주요 기능:
- 훈련된 모델 로딩 (3가지 유형 자동 감지)
- 텍스트 전처리
- 예측 실행
- 결과 출력
"""

import torch
import json
import os
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, BitsAndBytesConfig
from torch import nn
from peft import PeftModel

class TextClassifier:
    """
    텍스트 분류기 클래스
    
    훈련된 모델을 사용하여 텍스트를 분류하는 기능을 제공합니다.
    """
    
    def __init__(self, model_path: str, model_name: str, num_labels: int):
        """
        분류기 초기화
        
        Args:
            model_path: 모델 저장 경로
            model_name: 사전 훈련된 모델 이름
            num_labels: 레이블 개수
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"초기 디바이스 설정: {self.device}")
        
        print(f"모델 로딩 중: {model_path}")
        
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        config_path = os.path.join(model_path, "config.json")
        dynamic_weights_path = os.path.join(model_path, "pytorch_model.bin")

        # Case 1: LoRA Adapter
        if os.path.exists(adapter_config_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)

        # Case 2: Merged Model
        elif os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # 4-bit 양자화 모델인지 확인
            if config_data.get("quantization_config", {}).get("load_in_4bit"):
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                # LoRA가 병합된 일반 모델
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Case 3: Dynamic Quantized Model
        elif os.path.exists(dynamic_weights_path):
            # 동적 양자화는 CPU에 최적화되어 있습니다.
            if self.device.type == 'cuda':
                self.device = torch.device('cpu')

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
            model_to_quantize = AutoModelForSequenceClassification.from_config(config)
            
            # 모델 구조를 재구성하기 위해 양자화를 다시 적용합니다.
            self.model = torch.quantization.quantize_dynamic(
                model_to_quantize, {nn.Linear}, dtype=torch.qint8
            )
            
            # 재구성된 모델에 저장된 가중치를 불러옵니다.
            self.model.load_state_dict(torch.load(dynamic_weights_path, map_location=self.device))
        
        else:
            raise FileNotFoundError(f"모델 경로 '{model_path}'에서 유효한 모델 파일을 찾을 수 없습니다.")

        self.model.to(self.device)
        self.model.eval()
        
        self.label2id, self.id2label = self._load_label_mapping(model_path)
    
    def _load_label_mapping(self, model_path: str) -> tuple:
        """레이블 매핑을 로딩합니다."""
        label_map_path = os.path.join(model_path, "label_map.json")
        
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                label2id = json.load(f)
            id2label = {v: k for k, v in label2id.items()}
            print(f"레이블 매핑 로딩 완료: {label2id}")
        else:
            label2id = {"billing": 0, "delivery": 1, "product": 2}
            id2label = {0: "billing", 1: "delivery", 2: "product"}
            print("기본 레이블 매핑 사용")
        
        return label2id, id2label
    
    def preprocess_text(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """텍스트를 전처리합니다."""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        if 'token_type_ids' in encoding:
            encoding['token_type_ids'].zero_()
        
        return encoding
    
    def predict(self, text: str) -> Dict[str, Any]:
        """텍스트를 분류합니다."""
        inputs = self.preprocess_text(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_id = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
        
        predicted_label = self.id2label[predicted_id]
        
        result = {
            'text': text,
            'predicted_label': predicted_label,
            'predicted_id': predicted_id,
            'confidence': confidence,
            'probabilities': {
                self.id2label[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """여러 텍스트를 배치로 분류합니다."""
        return [self.predict(text) for text in texts]
    
    def print_prediction(self, result: Dict[str, Any]):
        """예측 결과를 예쁘게 출력합니다."""
        print(f"텍스트: {result['text']}")
        print(f"예측 레이블: {result['predicted_label']}")
        print(f"신뢰도: {result['confidence']:.4f}")
        print("모든 레이블 확률:")
        
        for label, prob in result['probabilities'].items():
            print(f"  {label:10}: {prob:.4f}")

def load_test_texts() -> List[str]:
    """테스트용 텍스트를 로딩합니다."""
    return [
        "This isn't the price I was quoted.",
        "The tracking says delivered, but it's not here.",
        "The item won't turn on.",
        "I cancelled this subscription last month.",
        "The box arrived completely crushed.",
        "I ordered a blue one, not a red one.",
        "Why did my monthly bill go up?",
        "It was left at the wrong apartment."
    ]

def main():
    """메인 함수 - 추론 과정을 실행합니다."""
    print("텍스트 분류 추론을 시작합니다!")
    print("=" * 50)
    
    # 옵션 1: LoRA 어댑터 경로
    # model_path = "models/text_classifier" 
    # 옵션 2: 4-bit 양자화 모델 경로
    # model_path = "models/quantized_4bit"
    # 옵션 3: 동적 양자화 모델 경로
    model_path = "models/quantized_dynamic"
    
    if not os.path.exists(model_path):
        print(f"모델을 찾을 수 없습니다: {model_path}")
        print("먼저 train.py를 실행하여 모델을 훈련해주세요.")
        return
    
    classifier = TextClassifier(
        model_path=model_path,
        model_name="bert-base-uncased",
        num_labels=3
    )
    
    test_texts = load_test_texts()
    
    print(f"{len(test_texts)}개의 테스트 텍스트로 예측을 시작합니다...")
    
    print("="*50)
    print("개별 텍스트 예측 결과:")
    print("="*50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"--- 예측 {i} ---")
        result = classifier.predict(text)
        classifier.print_prediction(result)
    
    print("="*50)
    print("사용자 입력 받기 (종료하려면 'quit' 입력):")
    print("="*50)
    
    while True:
        user_text = input("텍스트를 입력하세요: ").strip()
        
        if user_text.lower() in ['quit', 'exit', '종료']:
            break
        
        if not user_text:
            continue
        
        try:
            result = classifier.predict(user_text)
            classifier.print_prediction(result)
        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
    
    print("추론을 종료합니다!")

if __name__ == "__main__":
    main()
