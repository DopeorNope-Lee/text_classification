"""
텍스트 분류 모델 추론 스크립트 (inference.py)
==========================================

이 스크립트는 훈련된 텍스트 분류 모델을 사용하여 새로운 텍스트를 분류합니다.
학생들이 모델을 어떻게 사용하는지 이해할 수 있도록 구성되어 있습니다.

사용법:
    python inference.py

주요 기능:
- 훈련된 모델 로딩
- 텍스트 전처리
- 예측 실행
- 결과 출력
"""

import torch
import json
import os
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 모듈들 임포트
from utils.modeling import load_model
from utils.data import load_sample_data

class TextClassifier:
    """
    텍스트 분류기 클래스
    
    훈련된 모델을 사용하여 텍스트를 분류하는 기능을 제공합니다.
    """
    
    def __init__(self, model_path: str, model_name: str, num_labels: int, use_lora: bool = False):
        """
        분류기 초기화
        
        Args:
            model_path: 모델 저장 경로
            model_name: 사전 훈련된 모델 이름
            num_labels: 레이블 개수
            use_lora: LoRA 모델 여부
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"디바이스 설정: {self.device}")
        
        # 모델 로딩
        self.model, self.tokenizer = load_model(model_path, model_name, num_labels, use_lora)
        self.model.to(self.device)
        self.model.eval()
        
        # 레이블 매핑 로딩
        self.label2id, self.id2label = self._load_label_mapping(model_path)
        
        print("분류기 초기화 완료")
    
    def _load_label_mapping(self, model_path: str) -> tuple:
        """
        레이블 매핑을 로딩합니다.
        
        Args:
            model_path: 모델 경로
            
        Returns:
            label2id: 레이블을 ID로 매핑하는 딕셔너리
            id2label: ID를 레이블로 매핑하는 딕셔너리
        """
        label_map_path = os.path.join(model_path, "label_map.json")
        
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                label2id = json.load(f)
            id2label = {v: k for k, v in label2id.items()}
            print(f"레이블 매핑 로딩 완료: {label2id}")
        else:
            # 기본 레이블 매핑 (샘플 데이터용)
            label2id = {"neutral": 0, "positive": 1, "negative": 2}
            id2label = {0: "neutral", 1: "positive", 2: "negative"}
            print("기본 레이블 매핑 사용")
        
        return label2id, id2label
    
    def preprocess_text(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        텍스트를 전처리합니다.
        
        Args:
            text: 전처리할 텍스트
            max_length: 최대 시퀀스 길이
            
        Returns:
            inputs: 모델 입력 텐서
        """
        # 토크나이징
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # BERT류 모델의 token_type_ids를 0으로 설정
        if 'token_type_ids' in encoding:
            encoding['token_type_ids'].zero_()
        
        return encoding
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        텍스트를 분류합니다.
        
        Args:
            text: 분류할 텍스트
            
        Returns:
            result: 예측 결과 딕셔너리
        """
        # 텍스트 전처리
        inputs = self.preprocess_text(text)
        
        # 디바이스로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_id = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
        
        # 결과 구성
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
        """
        여러 텍스트를 배치로 분류합니다.
        
        Args:
            texts: 분류할 텍스트 리스트
            
        Returns:
            results: 예측 결과 리스트
        """
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results
    
    def print_prediction(self, result: Dict[str, Any]):
        """
        예측 결과를 예쁘게 출력합니다.
        
        Args:
            result: 예측 결과
        """
        print(f"텍스트: {result['text']}")
        print(f"예측 레이블: {result['predicted_label']}")
        print(f"신뢰도: {result['confidence']:.4f}")
        print("모든 레이블 확률:")
        
        for label, prob in result['probabilities'].items():
            print(f"  {label:10}: {prob:.4f}")

def load_test_texts() -> List[str]:
    """
    테스트용 텍스트를 로딩합니다.
    
    Returns:
        texts: 테스트 텍스트 리스트
    """
    # 샘플 테스트 텍스트들
    test_texts = [
        "This isn't the price I was quoted.",
        "The tracking says delivered, but it's not here.",
        "The item won't turn on.",
        "I cancelled this subscription last month.",
        "The box arrived completely crushed.",
        "I ordered a blue one, not a red one.",
        "Why did my monthly bill go up?",
        "It was left at the wrong apartment."
    ]
    
    return test_texts

def main():
    """
    메인 함수 - 추론 과정을 실행합니다.
    """
    print("텍스트 분류 추론을 시작합니다!")
    print("=" * 50)
    
    # 모델 경로 설정
    model_path = "models/text_classifier"
    
    # 모델이 존재하는지 확인
    if not os.path.exists(model_path):
        print(f"모델을 찾을 수 없습니다: {model_path}")
        print("먼저 train.py를 실행하여 모델을 훈련해주세요.")
        return
    
    # 분류기 초기화
    classifier = TextClassifier(
        model_path=model_path,
        model_name="bert-base-uncased",
        num_labels=3,
        use_lora=True  # LoRA 모델 사용
    )
    
    # 테스트 텍스트 로딩
    test_texts = load_test_texts()
    
    print(f"{len(test_texts)}개의 테스트 텍스트로 예측을 시작합니다...")
    
    # 개별 예측
    print("="*50)
    print("개별 텍스트 예측 결과:")
    print("="*50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"--- 예측 {i} ---")
        result = classifier.predict(text)
        classifier.print_prediction(result)
    
    # 배치 예측
    print("="*50)
    print("배치 예측 결과:")
    print("="*50)
    
    batch_results = classifier.predict_batch(test_texts)
    
    for i, result in enumerate(batch_results, 1):
        print(f"{i:2d}. {result['text'][:30]:30} → {result['predicted_label']:10} ({result['confidence']:.3f})")
    
    # 사용자 입력 받기
    print("="*50)
    print("직접 텍스트를 입력해보세요 (종료하려면 'quit' 입력):")
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
