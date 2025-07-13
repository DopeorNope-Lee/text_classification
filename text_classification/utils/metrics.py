"""
평가 지표 모듈 (utils/metrics.py)
==============================

이 모듈은 텍스트 분류 모델의 성능을 평가하기 위한 다양한 지표를 제공합니다.
학생들이 모델 성능을 이해하고 비교할 수 있도록 구성되어 있습니다.

주요 기능:
- 정확도 (Accuracy) 계산
- F1 점수 계산
- 혼동 행렬 (Confusion Matrix) 생성
- 분류 리포트 생성
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def compute_accuracy(predictions: List[int], labels: List[int]) -> float:
    """
    정확도를 계산합니다.
    
    Args:
        predictions: 예측값 리스트
        labels: 실제 레이블 리스트
        
    Returns:
        accuracy: 정확도 (0~1 사이 값)
    """
    return accuracy_score(labels, predictions)

def compute_f1_score(predictions: List[int], labels: List[int], 
                    average: str = 'weighted') -> float:
    """
    F1 점수를 계산합니다.
    
    Args:
        predictions: 예측값 리스트
        labels: 실제 레이블 리스트
        average: 평균 방식 ('micro', 'macro', 'weighted')
        
    Returns:
        f1: F1 점수 (0~1 사이 값)
    """
    return f1_score(labels, predictions, average=average)

def compute_metrics(predictions: List[int], labels: List[int], 
                   label_names: List[str] = None) -> Dict[str, float]:
    """
    다양한 평가 지표를 한 번에 계산합니다.
    
    Args:
        predictions: 예측값 리스트
        labels: 실제 레이블 리스트
        label_names: 레이블 이름 리스트 (선택사항)
        
    Returns:
        metrics: 평가 지표 딕셔너리
    """
    metrics = {}
    
    # 기본 지표들
    metrics['accuracy'] = compute_accuracy(predictions, labels)
    metrics['f1_weighted'] = compute_f1_score(predictions, labels, 'weighted')
    metrics['f1_macro'] = compute_f1_score(predictions, labels, 'macro')
    metrics['f1_micro'] = compute_f1_score(predictions, labels, 'micro')
    
    # 레이블별 F1 점수
    if label_names:
        f1_per_class = f1_score(labels, predictions, average=None)
        for i, label_name in enumerate(label_names):
            metrics[f'f1_{label_name}'] = f1_per_class[i]
    
    return metrics

def create_confusion_matrix(predictions: List[int], labels: List[int], 
                          label_names: List[str] = None) -> np.ndarray:
    """
    혼동 행렬을 생성합니다.
    
    Args:
        predictions: 예측값 리스트
        labels: 실제 레이블 리스트
        label_names: 레이블 이름 리스트 (선택사항)
        
    Returns:
        cm: 혼동 행렬
    """
    return confusion_matrix(labels, predictions)

def plot_confusion_matrix(predictions: List[int], labels: List[int], 
                         label_names: List[str] = None, 
                         save_path: str = None):
    """
    혼동 행렬을 시각화합니다.
    
    Args:
        predictions: 예측값 리스트
        labels: 실제 레이블 리스트
        label_names: 레이블 이름 리스트 (선택사항)
        save_path: 저장 경로 (선택사항)
    """
    cm = create_confusion_matrix(predictions, labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"혼동 행렬 저장됨: {save_path}")
    
    plt.show()

def print_classification_report(predictions: List[int], labels: List[int], 
                              label_names: List[str] = None):
    """
    분류 리포트를 출력합니다.
    
    Args:
        predictions: 예측값 리스트
        labels: 실제 레이블 리스트
        label_names: 레이블 이름 리스트 (선택사항)
    """
    report = classification_report(labels, predictions, 
                                 target_names=label_names, 
                                 digits=4)
    print("분류 리포트:")
    print(report)

def evaluate_model(model, dataloader, device: str = 'cpu') -> Tuple[List[int], List[int], Dict[str, float]]:
    """
    모델을 평가하고 결과를 반환합니다.
    
    Args:
        model: 평가할 모델
        dataloader: 평가용 데이터로더
        device: 사용할 디바이스
        
    Returns:
        all_predictions: 모든 예측값
        all_labels: 모든 실제 레이블
        metrics: 평가 지표
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 배치를 디바이스로 이동
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            # 예측
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # 결과 수집
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 지표 계산
    metrics = compute_metrics(all_predictions, all_labels)
    
    return all_predictions, all_labels, metrics

def print_metrics(metrics: Dict[str, float]):
    """
    평가 지표를 예쁘게 출력합니다.
    
    Args:
        metrics: 평가 지표 딕셔너리
    """
    print("모델 성능 지표:")
    print("=" * 40)
    
    # 주요 지표들
    main_metrics = ['accuracy', 'f1_weighted', 'f1_macro', 'f1_micro']
    for metric in main_metrics:
        if metric in metrics:
            print(f"{metric:15}: {metrics[metric]:.4f}")
    
    # 레이블별 F1 점수
    print("레이블별 F1 점수:")
    for key, value in metrics.items():
        if key.startswith('f1_') and key not in main_metrics:
            print(f"  {key:12}: {value:.4f}")

def compare_models(model_results: Dict[str, Dict[str, float]]):
    """
    여러 모델의 성능을 비교합니다.
    
    Args:
        model_results: 모델별 결과 딕셔너리
                      {모델명: {지표명: 값}}
    """
    print("모델 성능 비교:")
    print("=" * 60)
    
    # 지표별로 비교
    metrics_to_compare = ['accuracy', 'f1_weighted', 'f1_macro']
    
    for metric in metrics_to_compare:
        print(f"{metric.upper()}:")
        for model_name, results in model_results.items():
            if metric in results:
                print(f"  {model_name:20}: {results[metric]:.4f}")

def save_metrics(metrics: Dict[str, float], save_path: str):
    """
    평가 지표를 파일로 저장합니다.
    
    Args:
        metrics: 평가 지표 딕셔너리
        save_path: 저장 경로
    """
    import json
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"평가 지표 저장됨: {save_path}")

def load_metrics(load_path: str) -> Dict[str, float]:
    """
    저장된 평가 지표를 로딩합니다.
    
    Args:
        load_path: 로딩 경로
        
    Returns:
        metrics: 평가 지표 딕셔너리
    """
    import json
    
    with open(load_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    return metrics 