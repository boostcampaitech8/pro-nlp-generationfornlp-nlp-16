# LLM 기반 데이터셋 증강

OpenAI GPT-4o-mini Fine-tuned 모델을 활용한 한국어 수능형 문제 데이터 증강 파이프라인입니다.

이 브랜치는 신문 기사 및 도서 데이터를 기반으로 OpenAI GPT-4o-mini 모델을 활용해 수능형 객관식 문제를 생성하는 데이터 증강 파이프라인을 제공입니다. OpenAI Batch API와 Fine-tuned GPT-4o-mini 모델을 활용하여 비용 효율적으로 고품질 문제 데이터를 생성합니다.

## 주요 기능

- **다양한 데이터 소스 지원**: 신문 기사, 도서, AIHub 국어 교과 지문형 데이터
- **OpenAI Batch API 활용**: 비용 효율적인 대량 문제 생성 (50% 할인)
- **Fine-tuning 데이터 준비**: 고품질 데이터 선별 및 OpenAI Fine-tuning 형식 변환
- **Fine-tuned 모델 사용**: GPT-4o-mini 기반 커스텀 문제 생성 모델

## 기반 데이터 출처
이 프로젝트는 다음 데이터셋을 활용합니다:

**문제 생성용 원천 데이터**
- **신문 기사**: [모두의 말뭉치 - 신문 말뭉치](https://corpus.korean.go.kr/)
- **도서**: [모두의 말뭉치 - 문어 말뭉치](https://corpus.korean.go.kr/)

**Fine-tuning 학습 데이터**
- **AIHub 국어 교과 지문형 데이터**: [AI Hub - 한국어 교과서 말뭉치](https://aihub.or.kr/)
  - 고품질 수능형 문제 샘플로 활용
  - 상위 2,000개 문제로 GPT-4o-mini 모델 Fine-tuning

## 프로젝트 구조

```
korean_sat_solver/
├── src/
│   └── data_gen/
│       ├── data_augmentation.py      # 메인 데이터 증강 파이프라인
│       ├── data_loader.py             # 신문/도서 데이터 로더
│       ├── datagen_prompt.py          # 문제 생성 프롬프트 템플릿
│       ├── prepare_finetuning_data.py # Fine-tuning 데이터 준비
│       ├── sorting_good_workbook.py   # 문제 품질 평가 시스템
│       └── get_csv_from_workbook.py   # AIHub 교과서 데이터 추출
├── conf/
│   ├── config.yaml                    # Hydra 메인 설정 파일
│   ├── data_gen/
│   │   ├── newspaper.yaml             # 신문 데이터 생성 설정
│   │   ├── book.yaml                  # 도서 데이터 생성 설정
│   │   └── both.yaml                  # 혼합 데이터 생성 설정
│   ├── training/
│   │   └── default.yaml               # 학습 설정
│   ├── model/
│   │   └── gemma.yaml                 # 모델 설정
│   └── inference/
│       └── default.yaml               # 추론 설정
├── notebook/
│   └── combine_data.ipynb             # 데이터셋 통합 노트북
├── data/
│   ├── data4gen/
│   │   ├── newspaper/                 # 신문 기사 JSON 파일
│   │   ├── written/                   # 도서 JSON 파일
│   │   └── workbook/                  # AIHub 교과서 데이터
│   ├── train.csv                      # 원본 학습 데이터
│   ├── train_combined_augmented.csv   # 증강된 학습 데이터
│   ├── aihub_workbook.csv             # AIHub 교과서 데이터
│   ├── train_all_combined.csv         # 최종 통합 데이터
│   └── finetuning/                    # Fine-tuning JSONL 파일
├── run_gpt_api_finetuning.py          # Fine-tuning 실행 스크립트
├── .env                               # OpenAI API 키 설정
└── requirements.txt
```

## 설치

```bash
uv sync
```

## 환경 설정

프로젝트 루트에 `.env` 파일을 생성하고 OpenAI API 키를 설정하세요:

```bash
OPENAI_API_KEY=your-api-key-here
```

## 사용법

### 1. 데이터 증강 (문제 생성)

#### 신문 데이터만 사용

```bash
uv run -m src.data_gen.data_augmentation \
    --data_type newspaper \
    --newspaper_path data/data4gen/newspaper \
    --num_newspaper_problems 1000 \
    --output_csv data/train_newspaper_augmented.csv \
    --problems_per_article 1 \
    --batch_size 200 \
    --seed 42
```

#### 도서 데이터만 사용

```bash
uv run src.data_gen.data_augmentation \
    --data_type book \
    --book_path data/data4gen/written \
    --num_book_problems 500 \
    --output_csv data/train_book_augmented.csv \
    --problems_per_article 1 \
    --batch_size 200 \
    --seed 42
```

#### 신문 + 도서 혼합

```bash
uv run src.data_gen.data_augmentation \
    --data_type both \
    --newspaper_path data/data4gen/newspaper \
    --book_path data/data4gen/written \
    --num_newspaper_problems 800 \
    --num_book_problems 200 \
    --output_csv data/train_combined_augmented.csv \
    --problems_per_article 1 \
    --batch_size 200 \
    --seed 42
```
### 2. GPT-4o-mini fine-tuning 

#### 교과서 데이터 추출

```bash
uv run src.data_gen.get_csv_from_workbook
```

AIHub 국어 교과 지문형 데이터를 CSV 형식으로 변환합니다.

#### 문제 품질 평가

```bash
uv run src.data_gen.sorting_good_workbook
```

교과서 데이터의 품질을 평가하고 상위 품질 데이터를 선별합니다.

#### Fine-tuning 데이터 준비

```bash
uv run src.data_gen.prepare_finetuning_data
```

#### Fine-tuning 실행

```bash
uv run run_gpt_api_finetuning.py
```

준비된 JSONL 데이터를 OpenAI에 업로드하고 Fine-tuning 작업을 시작합니다.

### 3. 데이터셋 통합

모든 데이터 생성이 완료된 후, Jupyter Notebook을 사용하여 최종 학습 데이터셋을 통합합니다.

```bash
jupyter notebook notebook/combine_data.ipynb
```
