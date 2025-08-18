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
import sys
import shutil

# 프로젝트 모듈들 임포트
from utils.modeling import load_model
from transformers import AutoModelForSequenceClassification

def measure_model_size(model, model_name: str = "모델") -> Dict[str, Any]:
    """
    모델의 크기를 측정합니다.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    state_dict = model.state_dict()
    total_size_bytes = 0
    for param_name, param in state_dict.items():
          if isinstance(param, torch.Tensor):
              total_size_bytes += param.numel() * param.element_size()
          else:
              total_size_bytes += sys.getsizeof(param)
  
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
    """
    model.eval()
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(**test_input)
    
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
    """
    print("동적 양자화 적용 중...")
    
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
    """
    try:
        from transformers import AutoModelForSequenceClassification
        
        print("4비트 양자화 적용 중...")
        
        quantized_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print("4비트 양자화 완료")
        return quantized_model
        
    except ImportError:
        print("bitsandbytes가 설치되지 않았습니다.")
        return None

def compare_models(original_model, quantized_model, test_input):
    """
    원본 모델과 양자화된 모델을 비교합니다.
    """
    print("모델 비교:")
    print("=" * 60)
    
    print("크기 비교:")
    original_size = measure_model_size(original_model, "원본 모델")
    quantized_size = measure_model_size(quantized_model, "양자화 모델")
    
    size_reduction = (original_size['size_mb'] - quantized_size['size_mb']) / original_size['size_mb'] * 100
    print(f"  크기 감소: {size_reduction:.1f}%")
    
    print("속도 비교:")
    device_orig = next(original_model.parameters()).device
    is_quantized_dynamic = any(isinstance(m, torch.nn.quantized.dynamic.Linear) for m in quantized_model.modules())
    device_quant = torch.device('cpu') if is_quantized_dynamic else next(quantized_model.parameters()).device

    test_input_orig = {k: v.to(device_orig) for k, v in test_input.items()}
    test_input_quant = {k: v.to(device_quant) for k, v in test_input.items()}
    
    original_speed = measure_inference_speed(original_model, test_input_orig, model_name="원본 모델")
    quantized_speed = measure_inference_speed(quantized_model, test_input_quant, model_name="양자화 모델")
    
    speed_improvement = (quantized_speed['throughput'] - original_speed['throughput']) / original_speed['throughput'] * 100
    print(f"  속도 향상: {speed_improvement:.1f}%")
    
    print("정확도 테스트:")
    
    original_model.eval()
    quantized_model.eval()
    
    with torch.no_grad():
        original_output = original_model(**test_input_orig)
        quantized_output = quantized_model(**test_input_quant)
        
        original_probs = torch.softmax(original_output.logits, dim=-1)
        quantized_probs = torch.softmax(quantized_output.logits, dim=-1)
        
        prob_diff = torch.abs(original_probs.cpu() - quantized_probs.cpu()).mean().item()
        print(f"  평균 확률 차이: {prob_diff:.6f}")

def main():
    """
    메인 함수 - 양자화 과정을 실행합니다.
    """
    print("모델 양자화를 시작합니다!")
    print("=" * 50)
    
    model_path = "models/text_classifier"
    model_name = "bert-base-uncased"
    num_labels = 3
    
    if not os.path.exists(model_path):
        print(f"모델을 찾을 수 없습니다: {model_path}")
        return
    
    print("원본 LoRA 모델 로딩 중...")
    lora_model, tokenizer = load_model(
        model_path=model_path,
        model_name=model_name,
        num_labels=num_labels,
        use_lora=True
    )
    
    test_text = "이 제품은 정말 좋아요"
    test_input = tokenizer(
        test_text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    if 'token_type_ids' in test_input:
        test_input['token_type_ids'].zero_()
    
    print("원본 모델 로딩 완료")
    
    # 1. 동적 양자화
    print("="*50)
    print("동적 양자화 실행")
    print("="*50)
    
    # --- FIX: "Laundering" the model to create a clean state_dict ---
    print("LoRA 어댑터를 병합하여 깨끗한 상태의 모델을 생성합니다...")
    merged_model = lora_model.merge_and_unload()
    merged_state_dict = merged_model.state_dict()
    
    # Create a completely fresh model instance
    clean_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    clean_model.load_state_dict(merged_state_dict)
    print("깨끗한 모델 생성 완료.")
    
    # Now, quantize the clean model
    quantized_dynamic_model = apply_dynamic_quantization(clean_model)
    
    compare_models(clean_model, quantized_dynamic_model, test_input)
    
    # Save the correctly quantized model and its assets
    quantized_dynamic_save_path = "models/quantized_dynamic"
    print(f"동적 양자화 모델 저장 중: {quantized_dynamic_save_path}")
    os.makedirs(quantized_dynamic_save_path, exist_ok=True)
    torch.save(quantized_dynamic_model.state_dict(), os.path.join(quantized_dynamic_save_path, "pytorch_model.bin"))
    tokenizer.save_pretrained(quantized_dynamic_save_path)
    label_map_source = os.path.join(model_path, "label_map.json")
    if os.path.exists(label_map_source):
        shutil.copy(label_map_source, quantized_dynamic_save_path)
    print("동적 양자화 모델 저장 완료")
    
    # 2. 4비트 양자화 (선택사항)
    print("="*50)
    print("4비트 양자화 실행 (선택사항)")
    print("="*50)
    
    quantized_4bit_model = apply_4bit_quantization(model_path, model_name, num_labels)
    
    if quantized_4bit_model is not None:
        compare_models(clean_model, quantized_4bit_model, test_input)
        
        quantized_4bit_save_path = "models/quantized_4bit"
        print(f"4비트 양자화 모델 저장 중: {quantized_4bit_save_path}")
        quantized_4bit_model.save_pretrained(quantized_4bit_save_path)
        tokenizer.save_pretrained(quantized_4bit_save_path)
        if os.path.exists(label_map_source):
            shutil.copy(label_map_source, quantized_4bit_save_path)
        print("4비트 양자화 모델 저장 완료")
    
    print("="*50)
    print("양자화 완료!")
    print("이제 inference.py를 실행하여 양자화된 모델로 추론할 수 있습니다.")

if __name__ == "__main__":
    main()
