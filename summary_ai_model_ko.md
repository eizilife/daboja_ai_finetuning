# Llama-3.2-1B 요약 AI 모델

## 모델 개요

이 모델은 Meta의 Llama-3.2-1B-Instruct 모델을 기반으로 파인튜닝된 텍스트 요약 전문 AI 모델입니다.

### 기본 정보
- **베이스 모델**: meta-llama/Llama-3.2-1B-Instruct
- **라이브러리**: PEFT (Parameter-Efficient Fine-Tuning)
- **파이프라인**: 텍스트 생성
- **학습 방식**:
  - LoRA (Low-Rank Adaptation)
  - SFT (Supervised Fine-Tuning)
  - TRL (Transformer Reinforcement Learning)

## 모델 상세 설명

### 모델 특징
- **모델 유형**: 대규모 언어 모델 (LLM)
- **주요 용도**: 텍스트 요약 및 생성
- **언어**: 한국어 및 영어 지원
- **파인튜닝 기법**: LoRA를 활용한 효율적인 파라미터 학습

### 기술 스택
- Transformers
- PEFT 0.17.1
- TRL (Transformer Reinforcement Learning)

## 사용 방법

### 직접 사용
이 모델은 텍스트 요약 작업에 최적화되어 있으며, 다음과 같은 용도로 활용할 수 있습니다:
- 긴 문서의 핵심 내용 추출
- 뉴스 기사 요약
- 보고서 요약
- 회의록 정리

### 활용 예시
```python
# 모델 로드 및 사용 예시
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base_model, "path/to/lora/weights")

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# 텍스트 요약 수행
input_text = "요약할 텍스트..."
inputs = tokenizer(input_text, return_tensors="pt")
summary = model.generate(**inputs)
```

## 성능 및 제한사항

### 장점
- 경량화된 1B 파라미터로 빠른 추론 속도
- LoRA를 활용한 효율적인 메모리 사용
- 한국어 텍스트 요약에 특화

### 제한사항
- 매우 긴 문서(8000 토큰 이상)의 경우 성능 저하 가능
- 전문 분야(의학, 법률 등) 텍스트의 경우 추가 파인튜닝 필요
- 창의적인 요약보다는 추출적 요약에 더 적합

## 학습 정보

### 학습 데이터
- 다양한 한국어 및 영어 문서-요약 쌍
- 뉴스, 논문, 보고서 등 다양한 도메인 포함

### 학습 하이퍼파라미터
- **학습 방식**: Mixed Precision Training
- **LoRA 설정**:
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.1

## 환경 영향

이 모델의 학습과 운영에 따른 탄소 배출량은 다음과 같은 방법으로 추정할 수 있습니다:
- [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) 활용
- 효율적인 LoRA 학습으로 전체 파인튜닝 대비 90% 이상의 에너지 절감

## 라이선스 및 인용

### 라이선스
- Meta Llama 3.2 Community License Agreement에 따름
- 상업적 사용 시 제한사항 확인 필요

### 프레임워크 버전
- PEFT 0.17.1
- Transformers 4.36.0+
- PyTorch 2.0+

## 문의 및 기여

모델 개선이나 버그 리포트는 GitHub 저장소를 통해 제출해 주세요.