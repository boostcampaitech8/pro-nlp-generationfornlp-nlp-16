# Korean SAT Solver

한국어 수능형 문제 풀이를 위한 LLM 파인튜닝 프로젝트입니다. `beomi/gemma-ko-2b` 모델을 LoRA를 사용하여 파인튜닝합니다.

## 프로젝트 구조

```
korean_sat_solver/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # 데이터 로드 함수
│   │   └── preprocessing.py    # 프롬프트 생성, 토큰화
│   ├── model/
│   │   ├── __init__.py
│   │   └── model.py            # 모델 로드, LoRA 설정
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # SFTTrainer 설정
│   │   └── metrics.py          # 평가 메트릭
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict.py          # 추론 로직
│   └── utils/
│       ├── __init__.py
│       └── seed.py             # 시드 설정
├── train.py                    # 학습 진입점
├── inference.py                # 추론 진입점
├── requirements.txt
└── README.md
```

## 모듈 설명

### `src/data/`
- **dataset.py**: CSV 파일 로드 및 JSON 파싱, DataFrame 변환
- **preprocessing.py**: 프롬프트 템플릿 정의, 채팅 형식 변환, 토큰화

### `src/model/`
- **model.py**: 모델/토크나이저 로드, LoRA 설정, chat_template 설정

### `src/training/`
- **trainer.py**: SFTTrainer, DataCollator, SFTConfig 설정
- **metrics.py**: 정확도 계산, logits 전처리

### `src/inference/`
- **predict.py**: 추론 루프, 결과 저장

### `src/utils/`
- **seed.py**: 난수 시드 고정

## 설치

```bash
# 가상환경 생성 (권장)
python3.10 -m venv --system-site-packages venv
source venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

## 사용법

### 학습

```bash
# train.py 내 경로 설정 후 실행
python train.py
```

**설정 항목** (train.py 내에서 수정):
- `TRAIN_DATA_PATH`: 학습 데이터 경로
- `MODEL_NAME`: 베이스 모델명
- `OUTPUT_DIR`: 체크포인트 저장 경로
- `MAX_SEQ_LENGTH`: 최대 시퀀스 길이
- `NUM_EPOCHS`: 학습 에폭 수
- `LEARNING_RATE`: 학습률

### 추론

```bash
# inference.py 내 경로 설정 후 실행
python inference.py
```

**설정 항목** (inference.py 내에서 수정):
- `CHECKPOINT_PATH`: 학습된 체크포인트 경로
- `TEST_DATA_PATH`: 테스트 데이터 경로
- `OUTPUT_PATH`: 결과 저장 경로

## 데이터 형식

### 입력 데이터 (train.csv / test.csv)
```
id,paragraph,problems
generation-for-nlp-0,"지문 내용...","{""question"": ""질문"", ""choices"": [""선택지1"", ...], ""answer"": 1, ""question_plus"": ""보기""}"
```

### 출력 데이터 (output.csv)
```
id,answer
generation-for-nlp-0,2
generation-for-nlp-1,4
```

## 모델 구성

- **Base Model**: `beomi/gemma-ko-2b`
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
  - r: 6
  - lora_alpha: 8
  - lora_dropout: 0.05
  - target_modules: ['q_proj', 'k_proj']

## 프롬프트 템플릿

```
지문:
{paragraph}

질문:
{question}

<보기>:  # (있을 경우)
{question_plus}

선택지:
1 - {choice1}
2 - {choice2}
...

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:
```

## 주요 의존성

- transformers==4.40.2
- trl==0.11.4
- peft==0.13.2
- torch
- datasets
- evaluate
- scikit-learn

## 참고사항

- VRAM 제약으로 인해 입력 길이가 1024 토큰을 초과하는 데이터는 학습에서 제외됩니다.
- 더 긴 데이터를 포함하면 성능 향상이 가능합니다.

## 라이선스

[라이선스 정보 추가]
