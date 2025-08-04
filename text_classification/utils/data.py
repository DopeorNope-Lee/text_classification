"""
데이터 로딩 및 전처리 모듈 (utils/data.py)
========================================

이 모듈은 텍스트 분류를 위한 데이터 로딩, 전처리, 토크나이징 기능을 제공합니다.
학생들이 데이터 처리 과정을 이해할 수 있도록 단계별로 구성되어 있습니다.

주요 기능:
- Hugging Face datasets에서 데이터 로딩
- 텍스트 토크나이징
- 데이터셋 분할 (훈련/검증/테스트)
- 배치 데이터 생성
"""

import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset, DatasetDict
import numpy as np

class TextClassificationDataset(Dataset):
    """
    텍스트 분류를 위한 커스텀 데이터셋 클래스
    
    이 클래스는 텍스트와 레이블을 받아서 모델이 학습할 수 있는 형태로 변환합니다.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        """
        데이터셋 초기화
        
        Args:
            texts: 텍스트 리스트
            labels: 레이블 리스트 (정수)
            tokenizer: 토크나이저 객체
            max_length: 최대 시퀀스 길이
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """데이터셋의 크기 반환"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """인덱스에 해당하는 데이터 반환"""
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 텍스트를 토큰으로 변환
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # BERT류 모델의 경우 token_type_ids를 0으로 설정
        if 'token_type_ids' in encoding:
            encoding['token_type_ids'] = torch.zeros_like(encoding['token_type_ids'])
        
        # 1차원 텐서로 변환 (배치 차원 제거)
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
        
        return item
    
def load_dataset_from_local_parquet(data_dir: str):
    """
    로컬 디렉토리에서 Parquet 파일들을 로드하고, 레이블 정보를 생성합니다.
    
    Args:
        data_dir (str): Parquet 파일들이 있는 데이터 디렉토리 경로
                        (e.g., './customer-complaints-data/data')
                        
    Returns:
        DatasetDict: 'train', 'validation', 'test' 스플릿을 포함하는 데이터셋
        dict: label2id 매핑 딕셔너리
    """
    # 파일 경로 지정
    data_files = {
        "train": f"{data_dir}/train-00000-of-00001.parquet",
        "validation": f"{data_dir}/validation-00000-of-00001.parquet",
        "test": f"{data_dir}/test-00000-of-00001.parquet",
    }
    
    # Parquet 파일을 로드하여 DatasetDict 생성
    dataset = load_dataset("parquet", data_files=data_files)
    print(f"데이터 로딩 중: {data_dir}") 
    # label 컬럼에서 고유한 레이블 목록 추출 및 정렬
    # 'label' 컬럼이 문자열이라고 가정. 만약 정수형이라면 .features['label'].names를 사용.
    labels = dataset["train"].unique("label")
    labels.sort() # 일관된 순서를 위해 정렬
    
    # label2id, id2label 딕셔너리 생성
    label2id = {label: i for i, label in enumerate(labels)}
    # id2label = {i: label for i, label in enumerate(labels)} # 필요하다면 이것도 생성
    

    print(f"데이터 로딩 완료: {len(dataset['train'])}개 훈련 데이터")
    print(f"레이블: {label2id}")
    
    return dataset, label2id

def load_dataset_from_hf(dataset_name: str = "hblim/customer-complaints") -> Tuple[DatasetDict, Dict[str, int]]:
    """
    Hugging Face datasets에서 데이터를 로딩합니다.
    
    Args:
        dataset_name: 데이터셋 이름
        
    Returns:
        dataset: 로딩된 데이터셋
        label2id: 레이블 매핑 딕셔너리
    """
    print(f"데이터 로딩 중: {dataset_name}")
    
    # Hugging Face datasets에서 데이터 로딩
    dataset = load_dataset(dataset_name)
    
    # 레이블 매핑 생성
    id2label = dataset["train"].features["label"].names
    label2id = {lbl: idx for idx, lbl in enumerate(id2label)}
    
    print(f"데이터 로딩 완료: {len(dataset['train'])}개 훈련 데이터")
    print(f"레이블: {label2id}")
    
    return dataset, label2id

def tokenize_dataset(dataset: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    데이터셋을 토크나이징합니다.
    
    Args:
        dataset: 토크나이징할 데이터셋
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이
        
    Returns:
        tokenized_dataset: 토크나이징된 데이터셋
    """
    def _encode(batch):
        """배치 단위로 텍스트를 인코딩합니다."""
        enc = tokenizer(
            batch["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )
        enc["labels"] = batch["label"]  # 이미 정수형
        return enc
    
    # 토크나이징 적용
    tokenized_dataset = dataset.map(
        _encode, 
        batched=True, 
        remove_columns=["text", "label"]
    )
    
    # PyTorch 형식으로 설정
    tokenized_dataset.set_format(type="torch")
    
    print("토크나이징 완료")
    return tokenized_dataset

def get_dataset(tokenizer, max_length: int = 128, data_dir: str = "./data"):
    """
    토크나이징된 데이터셋과 레이블 매핑을 반환합니다.
    
    Args:
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이
        dataset_name: 데이터셋 이름
        
    Returns:
        tokenized_dataset: 토크나이징된 데이터셋
        label2id: 레이블 매핑 딕셔너리
    """
    # 데이터 로딩
    dataset, label2id = load_dataset_from_local_parquet(data_dir)
    # 토크나이징
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length)
    
    return tokenized_dataset, label2id

def load_data_from_csv(file_path: str) -> Tuple[List[str], List[str]]:
    """
    CSV 파일에서 텍스트와 레이블을 로딩합니다.
    
    Args:
        file_path: CSV 파일 경로
        
    Returns:
        texts: 텍스트 리스트
        labels: 레이블 리스트 (문자열)
    """
    import pandas as pd
    
    print(f"데이터 로딩 중: {file_path}")
    
    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    
    # 컬럼명 확인 및 데이터 추출
    if 'text' in df.columns and 'label' in df.columns:
        texts = df['text'].tolist()
        labels = df['label'].tolist()
    else:
        raise ValueError("CSV 파일에 'text'와 'label' 컬럼이 필요합니다.")
    
    print(f"{len(texts)}개의 데이터 로딩 완료")
    return texts, labels

def create_label_mapping(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    레이블을 정수로 매핑하는 딕셔너리를 생성합니다.
    
    Args:
        labels: 레이블 리스트 (문자열)
        
    Returns:
        label2id: 레이블을 ID로 매핑하는 딕셔너리
        id2label: ID를 레이블로 매핑하는 딕셔너리
    """
    unique_labels = sorted(list(set(labels)))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    print(f"레이블 매핑 생성 완료: {label2id}")
    return label2id, id2label

def prepare_dataset(texts: List[str], labels: List[str], tokenizer, 
                   max_length: int = 128, train_ratio: float = 0.8, 
                   val_ratio: float = 0.1) -> Tuple[TextClassificationDataset, 
                                                   TextClassificationDataset, 
                                                   TextClassificationDataset,
                                                   Dict[str, int]]:
    """
    데이터셋을 준비하고 훈련/검증/테스트 세트로 분할합니다.
    
    Args:
        texts: 텍스트 리스트
        labels: 레이블 리스트 (문자열)
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이
        train_ratio: 훈련 데이터 비율
        val_ratio: 검증 데이터 비율
        
    Returns:
        train_dataset: 훈련 데이터셋
        val_dataset: 검증 데이터셋
        test_dataset: 테스트 데이터셋
        label2id: 레이블 매핑 딕셔너리
    """
    # 레이블 매핑 생성
    label2id, _ = create_label_mapping(labels)
    
    # 레이블을 정수로 변환
    numeric_labels = [label2id[label] for label in labels]
    
    # 전체 데이터셋 생성
    full_dataset = TextClassificationDataset(texts, numeric_labels, tokenizer, max_length)
    
    # 데이터셋 분할
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"데이터셋 분할 완료:")
    print(f"  훈련: {len(train_dataset)}개")
    print(f"  검증: {len(val_dataset)}개")
    print(f"  테스트: {len(test_dataset)}개")
    
    return train_dataset, val_dataset, test_dataset, label2id

def create_dataloaders(train_dataset, val_dataset, test_dataset, 
                      batch_size: int = 16) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    데이터로더를 생성합니다.
    
    Args:
        train_dataset: 훈련 데이터셋
        val_dataset: 검증 데이터셋
        test_dataset: 테스트 데이터셋
        batch_size: 배치 크기
        
    Returns:
        train_loader: 훈련 데이터로더
        val_loader: 검증 데이터로더
        test_loader: 테스트 데이터로더
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"데이터로더 생성 완료 (배치 크기: {batch_size})")
    return train_loader, val_loader, test_loader

def load_sample_data() -> Tuple[List[str], List[str]]:
    """
    샘플 데이터를 로딩합니다 (교육용).
    
    Returns:
        texts: 샘플 텍스트 리스트
        labels: 샘플 레이블 리스트
    """
    sample_texts = [
        "안녕하세요 고객센터입니다",
        "이 제품 정말 최고예요!",
        "서비스가 너무 나빠요",
        "배송이 빨라서 좋았어요",
        "품질이 기대에 못 미쳐요",
        "친절한 응대 감사합니다",
        "환불하고 싶어요",
        "추천하고 싶은 제품입니다",
        "불만이 있어요",
        "만족스러운 구매였습니다"
    ]
    
    sample_labels = [
        "neutral", "positive", "negative", "positive", 
        "negative", "positive", "negative", "positive", 
        "negative", "positive"
    ]
    
    return sample_texts, sample_labels 