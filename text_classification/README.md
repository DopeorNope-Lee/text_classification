# 텍스트 분류 교육용 프로젝트

이 프로젝트는 한국어 텍스트 분류를 위한 교육용 코드입니다. 학생들이 자연어 처리와 딥러닝의 기본 개념을 이해하고, 실제로 모델을 훈련하고 사용하는 방법을 배울 수 있도록 구성되어 있습니다.

## 프로젝트 목표

- **텍스트 분류의 기본 개념 이해**
- **사전 훈련된 모델 활용 방법 학습**
- **LoRA (Low-Rank Adaptation) 기법 이해**
- **모델 훈련부터 추론까지 전체 과정 경험**
- **모델 최적화 방법 학습**

## 프로젝트 구조

```
text_classification/
├── setup.py    # 환경 설정 스크립트
├── train.py               # 모델 훈련 스크립트
├── inference.py           # 모델 추론 스크립트
├── quantization.py        # 모델 양자화 스크립트
├── utils/                 # 유틸리티 모듈들
│   ├── data.py           # 데이터 로딩 및 전처리
│   ├── modeling.py       # 모델 생성 및 관리
│   ├── metrics.py        # 평가 지표 계산
├── data/                  # 데이터 디렉토리 (환경 설정 시 생성)
│   ├── sample_dataset.csv # 샘플 데이터셋 (예시용)
│   └── test_texts.txt    # 테스트용 텍스트 (예시용)
├── models/                # 훈련된 모델 저장 디렉토리
├── checkpoints/           # 체크포인트 저장 디렉토리
├── requirements.txt       # 필요한 라이브러리 목록
└── README.md             # 프로젝트 설명서
```

## 시작하기

### 1. 환경 설정

먼저 프로젝트 환경을 설정합니다:
1. `cd 커맨드 사용 시, 본인이 압축을 푼 디렉토리로 이동해야 합니다.`

2. conda 가상환경을 만듭니다.

3. `pip install -r requirements.txt`와 `python setup.py` 스크립트로 기본 환경을 설정합니다.
    - `pip install -r requirements.txt`: 라이브러리 설치
    - `python setup_environment.py`: 폴더 생성 및 환경 셋팅
```bash
cd text_classifiaction
conda create -n text_classification python=3.11.8 -y
conda activate text_classification
python setup_environment.py
pip install -r requirements.txt
```

이 스크립트는 다음을 수행합니다:
- Python 버전 확인
- 필요한 패키지 설치
- 디렉토리 생성
- 샘플 데이터 생성

**실행 후 생성되는 파일들:**
- `data/` 폴더와 샘플 파일들 (예시용)
- `models/`, `checkpoints/`, `logs/` 폴더들


### 2. 모델 훈련

Hugging Face datasets에서 데이터를 로딩하여 모델을 훈련합니다:

```bash
python train.py
```

훈련 과정에서 다음을 확인할 수 있습니다:
- 모델 생성 및 설정
- 데이터 로딩 및 전처리 (Hugging Face datasets에서)
- 훈련 진행 상황
- 성능 평가 결과

**실행 후 생성되는 파일들:**
- `models/text_classifier/` 폴더에 훈련된 모델 파일들
- `models/text_classifier/label_map.json` (레이블 매핑)
- `models/text_classifier/metrics.json` (성능 지표)

### 3. 모델 추론

훈련된 모델을 사용하여 새로운 텍스트를 분류합니다:

```bash
python inference.py
```

추론 과정에서 다음을 경험할 수 있습니다:
- 훈련된 모델 로딩
- 텍스트 전처리
- 예측 실행
- 결과 해석

**실행 시 동작:**
- 미리 정의된 테스트 텍스트들로 예측 수행
- 사용자가 직접 텍스트를 입력하여 예측 가능
- 각 예측의 신뢰도와 모든 레이블별 확률 출력

### 4. 모델 최적화 (선택사항)

훈련된 모델을 양자화하여 크기를 줄이고 속도를 향상시킵니다:

```bash
python quantization.py
```

**실행 후 생성되는 파일들:**
- `models/quantized_dynamic/` 폴더에 동적 양자화된 모델
- `models/quantized_4bit/` 폴더에 4비트 양자화된 모델 (bitsandbytes 설치 시)

## 주요 기능

### 1. 데이터 처리 (`utils/data.py`)
- Hugging Face datasets에서 데이터 로딩 (`hblim/customer-complaints`)
- 텍스트 토크나이징
- 데이터셋 분할 (훈련/검증/테스트)
- 배치 데이터 생성
- CSV 파일 로딩 기능 (자체 데이터 사용 시)

### 2. 모델 관리 (`utils/modeling.py`)
- 사전 훈련된 모델 로딩
- LoRA 적용
- Linear Probing 설정
- 모델 저장 및 로딩

### 3. 성능 평가 (`utils/metrics.py`)
- 정확도, F1 점수 계산
- 혼동 행렬 생성
- 모델 비교 기능

## 사용된 기술

### 1. 사전 훈련된 모델
- **KoBERT**: 한국어에 특화된 BERT 모델
- **모델**: `skt/kobert-base-v1`

### 2. 효율적 파인튜닝
- **LoRA (Low-Rank Adaptation)**: 적은 파라미터로 모델 조정
- **Linear Probing**: 백본 모델 고정 후 분류기만 학습

### 3. 모델 최적화
- **동적 양자화**: 모델 크기 감소
- **4비트 양자화**: 더욱 극적인 크기 감소

## 예상 결과

### 모델 성능
- **정확도**: 80-90% (샘플 데이터 기준)
- **F1 점수**: 0.8-0.9
- **훈련 시간**: 1-3분 (CPU 기준)

### 모델 크기
- **원본 모델**: ~500MB
- **LoRA 모델**: ~10MB
- **양자화 모델**: ~100MB

## 학습 포인트

### 1. 자연어 처리 기초
- 토크나이징의 개념과 중요성
- 텍스트 전처리 과정
- 어텐션 메커니즘 이해

### 2. 딥러닝 모델 이해
- 트랜스포머 아키텍처
- 사전 훈련과 파인튜닝의 차이
- 과적합과 일반화

### 3. 효율적 학습 기법
- LoRA의 원리와 장점
- 적응적 학습률 조정
- 조기 종료 (Early Stopping)

### 4. 모델 최적화
- 양자화의 원리
- 속도와 정확도의 트레이드오프
- 모델 압축 기법

## 커스터마이징

### 1. 다른 모델 사용
`train.py`에서 `ModelConfig`를 수정하여 다른 모델을 사용할 수 있습니다:

```python
config = ModelConfig(
    pretrained_model_name="klue/bert-base",  # 다른 한국어 BERT 모델
    num_labels=3,
    use_lora=True,
    lora_r=16  # LoRA rank 조정
)
```

### 2. 다른 데이터셋 사용
`utils/data.py`에서 `get_dataset` 함수의 `dataset_name` 파라미터를 수정하여 다른 Hugging Face 데이터셋을 사용할 수 있습니다:

```python
# 다른 데이터셋 사용 예시
tokenized_dataset, label2id = get_dataset(
    tokenizer, 
    max_length=128, 
    dataset_name="your-dataset-name"  # 원하는 데이터셋 이름
)
```

현재 사용 중인 데이터셋: `hblim/customer-complaints` (고객 불만 분류)

### 3. 하이퍼파라미터 조정
`train.py`에서 다음 파라미터들을 조정할 수 있습니다:

```python
# 훈련 파라미터
num_epochs=5          # 에포크 수
learning_rate=1e-5    # 학습률
batch_size=32         # 배치 크기
max_length=256        # 최대 시퀀스 길이
```

## 문제 해결

### 1. 메모리 부족
- 배치 크기를 줄이세요: `batch_size=8`
- 시퀀스 길이를 줄이세요: `max_length=64`
- LoRA rank를 줄이세요: `lora_r=4`

### 2. 훈련 속도가 느림
- GPU 사용을 확인하세요
- 배치 크기를 늘리세요
- 데이터셋 크기를 줄이세요

### 3. 모델 성능이 낮음
- 더 많은 데이터를 사용하세요
- 에포크 수를 늘리세요
- 학습률을 조정하세요

## 추가 학습 자료

### 1. 관련 논문
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

### 2. 온라인 자료
- [Hugging Face Transformers 문서](https://huggingface.co/docs/transformers/)
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)

### 3. 한국어 NLP 자료
- [KoBERT GitHub](https://github.com/SKTBrain/KoBERT)
- [KLUE 벤치마크](https://klue-benchmark.com/)

---

**즐거운 학습 되세요!** 
