"""
텍스트 분류 모델 훈련 스크립트 (train.py)
========================================

이 스크립트는 텍스트 분류 모델을 훈련하는 메인 파일입니다.
학생들이 쉽게 이해할 수 있도록 단계별로 구성되어 있습니다.

사용법:
    python train.py

주요 기능:
- 데이터 로딩 및 전처리
- 모델 생성 및 설정
- 훈련 실행
- 모델 저장
- 성능 평가
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

from pathlib import Path

# 프로젝트 모듈들 임포트
from utils.data import get_dataset
from utils.modeling import ModelConfig, build_model, get_model_info, save_model
from utils.metrics import evaluate_model, print_metrics, save_metrics

class SmartCollator(DataCollatorWithPadding):
    """
    스마트 데이터 콜레이터
    
    BERT류 모델의 token_type_ids를 0으로 설정하는 기능을 추가한 콜레이터입니다.
    """
    
    def __call__(self, features):
        batch = super().__call__(features)
        # BERT류 모델의 token_type_ids를 0으로 설정
        if "token_type_ids" in batch:
            batch["token_type_ids"].zero_()
        return batch

def setup_device():
    """
    사용 가능한 디바이스를 설정합니다.
    
    Returns:
        device: 사용할 디바이스 ('cuda' 또는 'cpu')
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU 사용: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("CPU 사용")
    
    return device

def train_model(model, train_dataloader, val_dataloader, 
                num_epochs=3, learning_rate=2e-5, device='cpu'):
    """
    모델을 훈련합니다.
    
    Args:
        model: 훈련할 모델
        train_dataloader: 훈련 데이터로더
        val_dataloader: 검증 데이터로더
        num_epochs: 훈련 에포크 수
        learning_rate: 학습률
        device: 사용할 디바이스
        
    Returns:
        trained_model: 훈련된 모델
        training_history: 훈련 히스토리
    """
    print(f"모델 훈련 시작 (에포크: {num_epochs}, 학습률: {learning_rate})")
    
    # 모델을 디바이스로 이동
    model = model.to(device)
    
    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 손실 함수 설정
    criterion = nn.CrossEntropyLoss()
    
    # 훈련 히스토리
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # 훈련 루프
    for epoch in range(num_epochs):
        print(f"에포크 {epoch + 1}/{num_epochs}")
        
        # 훈련 모드
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # 배치를 디바이스로 이동
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            # 순전파
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 진행상황 출력 (10배치마다)
            if (batch_idx + 1) % 10 == 0:
                print(f"  배치 {batch_idx + 1}/{len(train_dataloader)}, 손실: {loss.item():.4f}")
        
        # 평균 훈련 손실 계산
        avg_train_loss = train_loss / len(train_dataloader)
        
        # 검증
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                
                outputs = model(**inputs)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                
                # 정확도 계산
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = correct / total
        
        # 히스토리에 저장
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_accuracy'].append(val_accuracy)
        
        print(f"  훈련 손실: {avg_train_loss:.4f}")
        print(f"  검증 손실: {avg_val_loss:.4f}")
        print(f"  검증 정확도: {val_accuracy:.4f}")
    
    print("훈련 완료!")
    return model, training_history

def main():
    """
    메인 함수 - 전체 훈련 과정을 실행합니다.
    """
    print("텍스트 분류 모델 훈련을 시작합니다!")
    print("=" * 50)
    
    # 1. 디바이스 설정
    device = setup_device()
    
    # 2. 모델 설정
    config = ModelConfig(
        pretrained_model_name="bert-base-uncased", 
        num_labels=3,  # billing, delivery, product
        use_lora=True,  # LoRA 사용
        lora_r=8,
        max_length=128
    )
    
    # 3. 모델 생성
    print("모델 생성 중...")
    model, tokenizer = build_model(config)
    get_model_info(model)
    
    # 4. 데이터 로딩
    print("데이터 로딩 중...")
    
    # Hugging Face datasets에서 데이터 로딩
    tokenized_dataset, label2id = get_dataset(tokenizer, config.max_length)
    
    # 데이터로더 생성
    train_loader = DataLoader(tokenized_dataset["train"], batch_size=16, shuffle=True, collate_fn=SmartCollator(tokenizer))
    val_loader = DataLoader(tokenized_dataset["validation"], batch_size=16, shuffle=False, collate_fn=SmartCollator(tokenizer))
    test_loader = DataLoader(tokenized_dataset["test"], batch_size=16, shuffle=False, collate_fn=SmartCollator(tokenizer))
    
    # 5. 모델 훈련
    trained_model, history = train_model(
        model, train_loader, val_loader, 
        num_epochs=3, learning_rate=2e-5, device=device # 효과적인 학습을 위해 수정 필요!
    )
    
    # 6. 모델 평가
    print("모델 평가 중...")
    predictions, true_labels, metrics = evaluate_model(
        trained_model, test_loader, device
    )
    
    print_metrics(metrics)
    
    # 7. 모델 저장
    print("모델 저장 중...")
    save_dir = "models/text_classifier"
    os.makedirs(save_dir, exist_ok=True)
    
    save_model(trained_model, tokenizer, save_dir, config.pretrained_model_name)
    
    # 레이블 매핑 저장
    import json
    with open(os.path.join(save_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    
    # 평가 지표 저장
    save_metrics(metrics, os.path.join(save_dir, "metrics.json"))
    
    print(f"훈련 완료! 모델이 '{save_dir}'에 저장되었습니다.")
    print("다음 단계:")
    print("1. inference.py를 실행하여 예측을 해보세요")
    print("2. quantization.py를 실행하여 모델을 최적화해보세요")

if __name__ == "__main__":
    main() 
