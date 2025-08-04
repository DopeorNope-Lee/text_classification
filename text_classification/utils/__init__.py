"""
텍스트 분류 교육용 프로젝트 - 유틸리티 패키지
============================================

이 패키지는 텍스트 분류 프로젝트에 필요한 다양한 유틸리티 모듈들을 포함합니다.

주요 모듈:
- data: 데이터 로딩 및 전처리
- modeling: 모델 생성 및 관리
- metrics: 성능 평가 지표
- prompts: 프롬프트 템플릿 (미래 확장용)
- collator: 데이터 콜레이터 (미래 확장용)
"""

from .data import (
    TextClassificationDataset,
    load_data_from_csv,
    create_label_mapping,
    prepare_dataset,
    create_dataloaders,
    load_sample_data,
    get_dataset,
    load_dataset_from_hf,
    load_dataset_from_local_parquet,
    tokenize_dataset
)

from .modeling import (
    ModelConfig,
    build_model,
    load_model,
    save_model,
    get_model_info,
    count_parameters,
    freeze_backbone_parameters
)

from .metrics import (
    compute_accuracy,
    compute_f1_score,
    compute_metrics,
    create_confusion_matrix,
    plot_confusion_matrix,
    print_classification_report,
    evaluate_model,
    print_metrics,
    compare_models,
    save_metrics,
    load_metrics
)

__version__ = "1.0.0"
__author__ = "교육용 텍스트 분류 프로젝트"
__description__ = "한국어 텍스트 분류를 위한 교육용 유틸리티 패키지"

__all__ = [
    # Data utilities
    "TextClassificationDataset",
    "load_data_from_csv",
    "create_label_mapping",
    "prepare_dataset",
    "create_dataloaders",
    "load_sample_data",
    "get_dataset",
    "load_dataset_from_hf",
    "load_dataset_from_local_parquet"
    "tokenize_dataset",
    
    # Modeling utilities
    "ModelConfig",
    "build_model",
    "load_model",
    "save_model",
    "get_model_info",
    "count_parameters",
    "freeze_backbone_parameters",
    
    # Metrics utilities
    "compute_accuracy",
    "compute_f1_score",
    "compute_metrics",
    "create_confusion_matrix",
    "plot_confusion_matrix",
    "print_classification_report",
    "evaluate_model",
    "print_metrics",
    "compare_models",
    "save_metrics",
    "load_metrics"
] 