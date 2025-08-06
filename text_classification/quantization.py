"""
모델 양자화 스크립트 (quantization.py)
====================================

이 스크립트는 훈련된 모델을 양자화하여 크기를 줄이고 추론 속도를 향상시킵니다.
학생들이 모델 최적화 방법을 이해할 수 있도록 구성되어 있습니다.

사용법:
    python quantization.py

주요 기능:
- 동적 양자화 (Dynamic Quantization)
- 4비트 양자화 (4-bit Quantization)
- 모델 크기 비교
- 추론 속도 측정
"""

import torch
import torch.nn as nn
import time
import os
from pathlib import Path
from typing import Dict, Any
# 모델 크기 측정할 때 필요
import sys

# 프로젝트 모듈들 임포트
from utils.modeling import load_model

def measure_model_size(model, model_name: str = "모델") -> Dict[str, Any]:
    """
    모델의 크기를 측정합니다.
    
    Args:
        model: 측정할 모델
        model_name: 모델 이름
        
    Returns:
        size_info: 크기 정보 딕셔너리
    """
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 모델 상태 딕셔너리 크기 계산
    state_dict = model.state_dict()
    total_size_bytes = 0
    # NOTE: 양자화하면 state_dict의 형태가 변해서 다음과 같이 계산해야 됨
    for param_name, param in state_dict.items():
          # 1. 값이 텐서인 경우 (압축된 가중치, bias, LayerNorm 등)
          if isinstance(param, torch.Tensor):
              total_size_bytes += param.numel() * param.element_size()
          # 2. 값이 텐서가 아닌 경우 (scale, zero_point 등 메타데이터)
          else:
              total_size_bytes += sys.getsizeof(param)
  
    # MB로 변환
    size_mb = total_size_bytes / (1024 * 1024)
    
    size_info = {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': size_mb,
        'total_size_bytes': total_size_bytes
    }
    
    print(f"{model_name} 크기 정보:")
    print(f"  총 파라미터 수: {total_params:,}")
    print(f"  학습 가능한 파라미터 수: {trainable_params:,}")
    print(f"  모델 크기: {size_mb:.2f} MB")
    
    return size_info

def measure_inference_speed(model, test_input, num_runs: int = 100, 
                          model_name: str = "모델") -> Dict[str, float]:
    """
    모델의 추론 속도를 측정합니다.
    
    Args:
        model: 측정할 모델
        test_input: 테스트 입력
        num_runs: 측정 횟수
        model_name: 모델 이름
        
    Returns:
        speed_info: 속도 정보 딕셔너리
    """
    model.eval()
    
    # 워밍업
    with torch.no_grad():
        for _ in range(10):
            _ = model(**test_input)
    
    # 속도 측정
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(**test_input)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    throughput = num_runs / total_time
    
    speed_info = {
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': throughput
    }
    
    print(f"{model_name} 속도 정보:")
    print(f"  총 시간: {total_time:.4f}초")
    print(f"  평균 추론 시간: {avg_time*1000:.2f}ms")
    print(f"  처리량: {throughput:.2f} 추론/초")
    
    return speed_info

def apply_dynamic_quantization(model):
    """
    동적 양자화를 적용합니다.
    
    Args:
        model: 양자화할 모델
        
    Returns:
        quantized_model: 양자화된 모델
    """
    print("동적 양자화 적용 중...")
    
    # 동적 양자화 적용
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    print("동적 양자화 완료")
    return quantized_model

def apply_4bit_quantization(model_path: str, model_name: str, num_labels: int):
    """
    4비트 양자화를 적용합니다 (bitsandbytes 사용).
    
    Args:
        model_path: 모델 경로
        model_name: 모델 이름
        num_labels: 레이블 개수
        
    Returns:
        quantized_model: 4비트 양자화된 모델
    """
    try:
        from transformers import AutoModelForSequenceClassification
        import bitsandbytes as bnb
        
        print("4비트 양자화 적용 중...")
        
        # 4비트로 모델 로딩
        quantized_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print("4비트 양자화 완료")
        return quantized_model
        
    except ImportError:
        print("bitsandbytes가 설치되지 않았습니다.")
        print("4비트 양자화를 사용하려면 다음을 설치하세요:")
        print("pip install bitsandbytes")
        return None

def compare_models(original_model, quantized_model, test_input):
    """
    원본 모델과 양자화된 모델을 비교합니다.
    
    Args:
        original_model: 원본 모델
        quantized_model: 양자화된 모델
        test_input: 테스트 입력
    """
    print("모델 비교:")
    print("=" * 60)
    
    # 크기 비교
    print("크기 비교:")
    original_size = measure_model_size(original_model, "원본 모델")
    quantized_size = measure_model_size(quantized_model, "양자화 모델")
    
    size_reduction = (original_size['size_mb'] - quantized_size['size_mb']) / original_size['size_mb'] * 100
    print(f"  크기 감소: {size_reduction:.1f}%")
    
    # 속도 비교
    print("속도 비교:")
    device_orig = next(original_model.parameters()).device
    device_quant = next(quantized_model.parameters()).device

    test_input_orig = {k: v.to(device_orig) for k, v in test_input.items()}
    test_input_quant = {k: v.to(device_quant) for k, v in test_input.items()}
    
    original_speed = measure_inference_speed(original_model, test_input_orig, model_name="원본 모델")
    quantized_speed = measure_inference_speed(quantized_model, test_input_quant, model_name="양자화 모델")
    
    speed_improvement = (quantized_speed['throughput'] - original_speed['throughput']) / original_speed['throughput'] * 100
    print(f"  속도 향상: {speed_improvement:.1f}%")
    
    # 정확도 비교 (간단한 테스트)
    print("정확도 테스트:")
    
    original_model.eval()
    quantized_model.eval()
    
    with torch.no_grad():
        original_output = original_model(**test_input_orig)
        quantized_output = quantized_model(**test_input_quant)
        
        original_probs = torch.softmax(original_output.logits, dim=-1)
        quantized_probs = torch.softmax(quantized_output.logits, dim=-1)
        
        # 확률 차이 계산
        prob_diff = torch.abs(original_probs.cpu() - quantized_probs.cpu()).mean().item()
        print(f"  평균 확률 차이: {prob_diff:.6f}")

def save_quantized_model(model, save_path: str, model_name: str):
    """
    양자화된 모델을 저장합니다.
    
    Args:
        model: 저장할 모델
        save_path: 저장 경로
        model_name: 모델 이름
    """
    print(f"양자화된 모델 저장 중: {save_path}")
    
    # 디렉토리 생성
    os.makedirs(save_path, exist_ok=True)
    
    # 모델 저장
    model.save_pretrained(save_path)
    
    print("양자화된 모델 저장 완료")

def main():
    """
    메인 함수 - 양자화 과정을 실행합니다.
    """
    print("모델 양자화를 시작합니다!")
    print("=" * 50)
    
    # 모델 경로 설정
    model_path = "models/text_classifier"
    
    # 모델이 존재하는지 확인
    if not os.path.exists(model_path):
        print(f"모델을 찾을 수 없습니다: {model_path}")
        print("먼저 train.py를 실행하여 모델을 훈련해주세요.")
        return
    
    # 원본 모델 로딩
    print("원본 모델 로딩 중...")
    original_model, tokenizer = load_model(
        model_path=model_path,
        model_name="skt/kobert-base-v1",
        num_labels=3,
        use_lora=True
    )
    
    # 테스트 입력 생성
    test_text = "이 제품은 정말 좋아요"
    test_input = tokenizer(
        test_text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # BERT류 모델의 token_type_ids를 0으로 설정
    if 'token_type_ids' in test_input:
        test_input['token_type_ids'].zero_()
    
    print("원본 모델 로딩 완료")
    
    # 1. 동적 양자화
    print("="*50)
    print("동적 양자화 실행")
    print("="*50)
    # 동적 양자화전에 추론 시 forward 호출을 위해 `.merge_and_unload()`.
    original_model.merge_and_unload()
    quantized_model = apply_dynamic_quantization(original_model)
    
    # 모델 비교
    compare_models(original_model, quantized_model, test_input)
    
    # 양자화된 모델 저장
    quantized_save_path = "models/quantized_dynamic"
    save_quantized_model(quantized_model, quantized_save_path, "동적 양자화 모델")
    
    # 2. 4비트 양자화 (선택사항)
    print("="*50)
    print("4비트 양자화 실행 (선택사항)")
    print("="*50)
    
    quantized_4bit = apply_4bit_quantization(model_path, "skt/kobert-base-v1", 3)
    
    if quantized_4bit is not None:
        # 4비트 모델과 비교
        compare_models(original_model, quantized_4bit, test_input)
        
        # 4비트 모델 저장
        quantized_4bit_save_path = "models/quantized_4bit"
        save_quantized_model(quantized_4bit, quantized_4bit_save_path, "4비트 양자화 모델")
    
    print("양자화 완료!")
    print("다음 단계:")
    print("1. inference.py를 실행하여 양자화된 모델로 추론해보세요")
    print("2. 성능 차이를 비교해보세요")

if __name__ == "__main__":
    main() 
