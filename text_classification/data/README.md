# 고객 불만사항 분류 데이터셋
## **데이터셋 개요**

본 데이터셋은 **ChatGPT 4o**를 사용하여 생성된 텍스트 분류 학습용 데이터셋입니다. 소형 언어 모델(sLM)의 교육을 목적으로 제작되었으며, 'billing', 'delivery', 'product' 세 가지 카테고리의 고객 불만사항 텍스트를 포함합니다.

---

## **라벨링 기준**

### **라벨 정의**

| 라벨 이름 | 의미 |
| :--- | :--- |
| billing | 결제 관련 불만 |
| delivery | 배송 관련 불만 |
| product | 제품 자체에 대한 불만 |

### **카테고리 분류 기준**

각 데이터는 특정 카테고리를 주제로 한 프롬프트를 통해 생성되었습니다. 예를 들어, 배송(delivery) 데이터는 아래와 같은 프롬프트를 기반으로 제작되었습니다.

> Produce a list of 500 unique examples of angry customers with an issue related to delivery (< 20 words each) without using code.

---

## **데이터 출처 및 구성**

-   **출처**: ChatGPT 4o 생성
-   **구성**: 'billing', 'delivery', 'product' 세 개의 텍스트 파일로 구성된 데이터를 통합
