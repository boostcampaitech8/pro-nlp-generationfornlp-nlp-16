# CSAT-Solver: Korean SAT Solver

## 📌 프로젝트 개요

> "AI는 과연 수능 문제를 풀 수 있을까요?”

**CSAT-Solver** 프로젝트는 이 질문에서 시작되었습니다. 대한민국 수험생들이 치르는 대학수학능력시험(CSAT), 그중에서도 고도의 문해력과 배경지식을 요구하는 **국어**와 **사회** 영역은 AI에게도 큰 도전입니다.

우리 팀은 **Naver Boostcamp AI Tech NLP 트랙**의 Level 2 대회를 통해, 거대 모델이 아닌 Small Language Model(SLM)을 사용하여 이 난제에 도전했습니다. 

본 리포지토리는 총 869개의 평가 데이터(Public/Private)에 대해 최적의 성능을 달성하기 위해 시도했던 다양한 실험과 방법론, 그리고 최종 결과물인 **CSAT-Solver** 모델을 소개합니다.

<br>

## 🎖️ 최종 결과
![alt text](<./assets/leaderboard_final_result.png>)
*Figure 1. Final leaderboard result (Private Score)*
![alt text](<./assets/leaderboard_intermediate_result.png>)
*Figure 2. Intermediate leaderboard result (Public Score)*

`Qwen2.5-32B` 모델을 활용하여 **Public 기준 0.8073**의 정확도, **Private 기준 0.7370**의 정확도를 달성하였습니다. 데이터 증강 및 추출을 통해 도메인 특화 데이터셋을 구축하여 학습하였으며, 앙상블 및 고난이도 문제 재추론을 위한 Selective Inference, 그리고 MoA 전략을 결합하여 정답 도출의 정확도를 확보하였습니다.

### 주요 구현 사항 요약
최고 성능을 달성한 최종 모델의 구성을 위해 설계된 핵심 아키텍처와 도메인 특화 데이터셋, 그리고 추론 전략의 요약 정보입니다.

#### 1. 모델
| 항목 | 내용 |
| :--- | :--- |
| **Base Model** | `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` |
| **MoA Model** | `skt/A.X-Light`, `EXAONE-4.0-light` |

#### 2. 학습 데이터셋
| 분류 | 데이터셋 명칭 | 주요 포함 내용 |
| :--- | :--- | :--- |
| **기본 데이터** | KMMLU, MMMLU, KLUE MRC | 한국사, 고교 교과(역사·경제·정치·지리·심리), 도메인별 MRC |
| **증강 데이터** | AI-Hub, 모두의 말뭉치 | 교과별 지문 및 문제, 학술논문, 신문·문어체 기반 증강 데이터 |
| **추출 데이터** | 한국사 능력 검정 시험 | 한능검 기출문제 이미지 및 PDF 기반 추출 데이터 |

#### 3. 추론 전략
| 기법 | 설명 |
| :--- | :--- |
| **Ensemble** | 다양한 실험 브랜치 및 모델 결과값을 결합하여 정답률 극대화 |
| **Selective Inference** | 모델 간 예측이 불일치하는 문항을 선별하여 고성능 모델(MoA)로 재추론 |
| **MoA** | **Mixture of Agents** 구성을 통한 다단계 분석 및 정답 도출 (하단 상세 참고) |

<details>
  <summary><h4 style="display: inline;">&nbsp;Mixture of Agents (MoA) 상세 구조</h4></summary>
  

| 계층 | 역할 | 활용 모델 | 주요 기능 |
| :--- | :--- | :--- | :--- |
| **Layer 1** | **Proposers** | `skt/A.X-Light` | 다양한 Temperature 설정으로 문제에 대한 다각도 분석 및 힌트 생성 |
| **Layer 2** | **Aggregator** | `EXAONE-4.0-light` | Layer 1의 분석 정보를 취합하여 최종 정답 결정 |

</details>

<br>

## 🤝 팀 소개 & 역할

| 프로필 | 이름 | 역할 |
| :---: | :---: | :--- |
| <a href="http://github.com/27kanghan"><img src="https://github.com/27kanghan.png" width="80" height="80" style="border-radius: 50%;"/><br /></a> | **강한** | Focal Loss, NEFTune, CoT (In-context-learning),<br>CoT (Reasoning Dataset based FineTuning) |
| <a href="https://github.com/ho44013"><img src="https://github.com/ho44013.png" width="80" height="80" style="border-radius: 50%;"/><br /></a> | **김성호** | Streamlit 기반 데이터 세부 분석, 리더보드 모델 선정 실험,<br>Unsloth 라이브러리 활용 학습 환경 구축 |
| <a href="https://github.com/sooyouki"><img src="https://github.com/sooyouki.png" width="80" height="80" style="border-radius: 50%;"/><br /></a> | **김수영** | PDF 추출 기반 데이터셋 구축, 프롬프트 순서 관련 실험 |
| <a href="https://github.com/chanspar"><img src="https://github.com/chanspar.png" width="80" height="80" style="border-radius: 50%;"/><br /></a> | **박찬서** | Harness 활용 한국어 벤치마크 우수 베이스 모델(7~10B) 선별,<br>30B 모델(Qwen) 환경 구축 |
| <a href="https://github.com/jacejung-dev"><img src="https://github.com/jacejung-dev.png" width="80" height="80" style="border-radius: 50%;"/><br /></a> | **정지웅** | LLM 기반 데이터셋 증강, llama.cpp 서버 환경 구축,<br>MoA(Mixture-of-Agents) 유사 파이프라인 구현 및 실험 |

<br>

## 📄 Wrap-Up Report

### [NLP16_리포트.pdf](./assets/NLP16-generationForNLP-WrapupReport.pdf)
> 데이터 EDA부터 앙상블까지 프로젝트 전반의 회고는 랩업 리포트를 통해 확인할 수 있습니다.

<br>

## 🛠️ 개발 환경 및 설치 (Installation)

### 요구 사항 (Requirements)
본 프로젝트는 **Python 3.10** 환경에서 구동됩니다.
패키지 관리를 위해 **[uv](https://github.com/astral-sh/uv)** 사용을 권장합니다.

### 설치 명령어
```bash
# uv 설치 (없을 경우)
pip install uv

# 프로젝트 의존성 설치 및 가상환경 생성 (uv.lock 기반)
uv sync

# 가상환경 활성화
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

<br>

## 📁 패키지 구조 (Project Structure)
Hydra 기반의 구조화된 프로젝트 구성을 따르고 있습니다.

```
korean_sat/
├── 📂 conf/                  # Hydra 설정 파일 (실험 제어)
│   ├── config.yaml           # 메인 설정
│   ├── model/                # 모델별 파라미터 (Gemma, Qwen 등)
│   ├── training/             # 학습 하이퍼파라미터
│   └── inference/            # 추론 설정
├── 📂 src/                   # 소스 코드 패키지
│   ├── data/                 # 데이터 로딩, 전처리, 토크나이징
│   ├── model/                # 모델 초기화, Lora 설정
│   ├── training/             # Trainer 및 Metrics 정의
│   ├── inference/            # 추론 로직 및 결과 저장
│   └── utils/                # 유틸리티 (Seed 등)
├── train.py                  # 학습 실행 스크립트
├── inference.py              # 추론 실행 스크립트
├── pyproject.toml            # 프로젝트 메타데이터 및 의존성
└── requirements.txt          # 설치 패키지 목록
```

<br>

## 📚 주요 라이브러리 (Main Libraries)
핵심적으로 사용된 라이브러리와 버전 정보입니다. (`pyproject.toml` 기준)

| 라이브러리 | 역할 | 버전(최소) |
|------------|------|------------|
| `transformers` | 모델 로드 및 토크나이징 | >= 4.46.0 |
| `peft` | LoRA 등 경량화 파인튜닝 | 0.13.2 |
| `trl` | SFT(Supervised Fine-tuning) Trainer | 0.11.4 |
| `hydra-core` | 설정(Configuration) 관리 | 1.3.2 |
| `bitsandbytes` | 4-bit/8-bit 양자화 | >= 0.49.0 |
| `accelerate` | 학습 가속화 및 분산 처리 | >= 0.34.0 |
| `datasets` | 데이터셋 관리 | >= 4.4.2 |
| `evaluate` | 성능 평가 메트릭 | 0.4.3 |

<br>

## 🗓️ 세부일정 (Schedule)
**프로젝트 기간:** 2025.12.15(월) ~ 2026.01.06(화) (3주)

| 주차 | 기간 | 주요 활동 |
|:---:|:---:|---|
| **1주차** | 12.15 ~ 12.21 | • 부스트코스 강의 수강 (Generation for NLP)<br>• 베이스라인 코드 분석 및 모듈화 (Hydra 도입, 구조화) |
| **2주차** | 12.22 ~ 12.28 | • 소형 모델 (7B~10B) 실험 및 성능 검증<br>• 프롬프트 엔지니어링 및 데이터 전처리 파이프라인 구축 |
| **3주차** | 12.29 ~ 01.06 | • 대형 모델 (32B) 파인튜닝 (QLoRA) 및 최적화<br>• ICL (In-Context Learning) 및 Reasoning 기법 적용<br>• MoA (Mixture of Agents) 적용 실험<br>• 앙상블 및 최종 추론 |

<br>

## 📖 사용법 (Usage)
본 프로젝트는 **Hydra**를 사용하여 설정을 관리하며, `uv run`을 통해 스크립트를 실행하는 것을 권장합니다.

### 학습 (Training)
`uv run` 명령어를 사용하여 가상환경 활성화 없이도 학습을 실행할 수 있습니다. Hydra 문법(`key=value`)을 사용하여 설정을 손쉽게 변경할 수 있습니다.

```bash
# 기본 설정으로 학습 실행
uv run train.py

# 하이퍼파라미터 변경 예시
uv run train.py training.num_epochs=5 data.max_length=1024 training.per_device_train_batch_size=2
```

### 추론 (Inference)
추론 시에도 `uv run`을 사용합니다. 체크포인트 경로는 자동으로 탐색되거나 직접 지정할 수 있습니다.

```bash
# 최신 체크포인트 자동 로드 및 추론
uv run inference.py 

# 특정 체크포인트 폴더 및 스텝 지정
uv run inference.py inference.checkpoint_dir=./outputs/train/2024-01-01/12-00-00 inference.checkpoint_step="best"

# 출력 파일명 변경
uv run inference.py inference.output_file="my_submission.csv"
```
<br>

## 🧪 시도한 것들
성능 향상을 위해 데이터 엔지니어링, 학습 최적화, 고도화된 추론 전략 등 다각도의 **기술적 기능(Features)** 을 구현하고 브랜치별로 관리하였습니다. 또한 베이스 모델을 선정하기 위해 다양한 아키텍처와 파라미터 규모를 가진 모델들을 대상으로 **비교 실험(Experiments)** 을 수행하며 최적의 조합을 도출했습니다.

**본 섹션에는 최종 모델에 적용된 기법뿐만 아니라 실험 과정에서 시도했던 모든 유의미한 방법론들이 브랜치별로 보존되어 있습니다.** 상세 요약 및 브랜치 링크는 아래 테이블과 토글을 통해 확인하실 수 있습니다.

### Features

| 카테고리 | 브랜치 | 설명 |
|:---:|:---|:---|
| **데이터 증강** | [`feature/data_augmenting`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/feature/data_augmenting) | 외부 데이터 활용 학습 데이터 확장 |
| **데이터 추출** | [`feature/data_extracting`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/feature/data_extracting) | 한능검 등 시험지 PDF → 데이터셋 변환 |
| **파인튜닝** | [`feature/qlora`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/feature/qlora) | QLoRA 기반 효율적 파인튜닝 |
| **프롬프팅** | [`feature/apply_cot`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/feature/apply_cot) | Chain-of-Thought 프롬프트 적용 |
| **최적화** | [`feature/unsloth`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/feature/unsloth) | Unsloth 라이브러리 적용 |
| **앙상블** | [`feature/ensemble`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/feature/ensemble) | 다중 모델 앙상블 |
| **MoA** | [`experiment/MoA`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/experiment/MoA) | Mixture of Agents 적용 |

### Experiments
| 모델 | 브랜치 |
|:---:|:---|
| **Qwen2.5-7B** | [`experiment/model_select_qwen2.5_7b`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/experiment/model_select_qwen2.5_7b) |
| **Qwen2.5-32B** | [`experiment/model_select_qwen2.5_32b`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/experiment/model_select_qwen2.5_32b) |
| **Qwen3-32B** | [`experiment/model_select_qwen3_32b`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/experiment/model_select_qwen3_32b) |
| **Gemma-9B** | [`experiment/model_select_gemma9b`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/experiment/model_select_gemma9b) |
| **EXAONE-32B** | [`experiment/exaone_32b`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/experiment/exaone_32b) |
| **Llama** | [`experiment/model_select_llama`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/experiment/model_select_llama) |
| **A.X** | [`experiment/model_select_AX`](https://github.com/boostcampaitech8/pro-nlp-generationfornlp-nlp-16/tree/experiment/model_select_AX) |

<details>
<summary><h3 style="display: inline;">Branch 상세 설명</h3></summary>

### 1. 데이터셋

- **`feature/data_augmenting`**:
    - GPT-4o-mini API로 신문·도서에서 수능형 객관식 문제를 대량 생성하는 데이터 증강 파이프라인을 추가했습니다.
    - AIHub 교과서 데이터 추출·모델 기반 품질 평가(스코어링) 및 상위 데이터로 Fine-tuning 준비/실행 스크립트를 구현했습니다.

- **`feature/data_extracting`**: 
    - 한국사능력검정시험 PDF를 학습 데이터로 변환하는 파이프라인을 구축했습니다.
    - pypdfium2로 PDF를 이미지 변환 후, GPT-4o Vision API로 OCR 및 구조화 추출을 수행했습니다.
    - 품질 검수를 위해 추출 모델보다 상위 모델로 문제를 풀게 하여, 오답 또는 풀이 불가 판정 시 해당 데이터를 제외했습니다.

### 2. 파인튜닝 (Fine-tuning)
- **`feature/qlora`**: VRAM 효율성을 위해 QLoRA(Quantized LoRA)를 적용했습니다. `bitsandbytes`를 사용하여 4bit 양자화를 수행했습니다.

### 3. 프롬프팅 (Prompting)
- **`feature/apply_cot`**:
    - Chain-of-Thought 적용: 학습/추론 파이프라인에 CoT를 도입해 모델이 단계별 분석(reasoning)을 생성하도록 학습했습니다.
    - Two-stage CoT 및 로컬/배치 추론 구현: Stage1(분석 생성) → Stage2(분석 기반 정답 추출) 방식의 추론 구현 및 로컬 모델/배치용 CoT 추론 스크립트를 추가했습니다.
    - 추론용 reasoning 생성 자동화 및 평가 도구 추가: GPT 기반 reasoning 생성 스크립트와 평가/저장 파이프라인 등을 포함하여 실험/평가를 용이하게 했습니다.

### 4. 최적화
- **`feature/unsloth`**:
    - Unsloth 통합: Unsloth 기반의 학습·추론 최적화를 도입하여 모델 로딩·추론 속도와 메모리 효율을 개선했습니다.
    - 학습·트레이너 지원: Unsloth에 최적화된 LoRA 적용 및 SFTTrainer 지원 추가로 빠른 SFT 학습이 가능해졌습니다.
    - 설정·실행 편의성 개선: conf에 Unsloth 활성화용 모델 설정(qwen_qlora.yaml)과 추론 설정 추가 및 사용 가이드를 제공했습니다.

### 5. 실험 (Experiments)
주요 모델별 상세 실험 설정입니다.

- **`experiment/model_select_qwen2.5_7b`**:
  - `Qwen/Qwen2.5-7B-Instruct` 모델을 기반으로 실험을 진행했습니다.
  - 파라미터는 `r=16`, `lora_alpha=16`, `lora_dropout=0`를 사용했습니다.
  - 템플릿은 `<|im_start|>assistant`를 적용했습니다.

- **`experiment/model_select_qwen2.5_32b`**:
  - `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` 모델을 기반으로 실험을 진행했습니다 (4-bit 양자화, nf4).
  - 파라미터는 `r=8`, `lora_alpha=16`, `lora_dropout=0.05`를 사용했습니다.
  - 템플릿은 `<|im_start|>assistant\n`을 적용했습니다.

- **`experiment/model_select_qwen3_32b`**:
  - `unsloth/Qwen3-32B-bnb-4bit` 모델을 기반으로 실험을 진행했습니다 (4-bit 양자화, nf4).
  - 파라미터는 `r=8`, `lora_alpha=16`, `lora_dropout=0.05`를 사용했습니다.
  - 템플릿은 `<|im_start|>assistant\n`을 적용했습니다.
  - Lost in the Middle 논문 기반 프롬프트 순서 변경 (질문 → 보기 → 지문 → 선택지)을 적용했습니다.

- **`experiment/model_select_gemma9b`**:
  - `rtzr/ko-gemma-2-9b-it` 모델을 기반으로 실험을 진행했습니다.
  - 파라미터는 `r=16`, `lora_alpha=32`, `lora_dropout=0.05`를 사용했습니다.
  - 템플릿은 `<start_of_turn>model`을 적용했습니다.

- **`experiment/exaone_32b`**:
  - `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` 모델을 기반으로 실험을 진행했습니다.
  - 파라미터는 `r=8`, `lora_alpha=16`, `lora_dropout=0.05`를 사용했습니다.
  - 템플릿은 `[|assistant|]`를 적용했습니다.

- **`experiment/model_select_llama`**: 
  - `beomi/Llama-3-Open-Ko-8B` 모델을 기반으로 실험을 진행했습니다.
  - 파라미터는 `r=8`, `lora_alpha=16`, `lora_dropout=0.05`를 사용했습니다.
  - 템플릿은 `<|start_header_id|>assistant<|end_header_id|>`를 적용했습니다.

- **`experiment/model_select_AX`**:
  - `skt/A.X-4.0-Light` 모델을 기반으로 실험을 진행했습니다.
  - 파라미터는 `r=8`, `lora_alpha=16`, `lora_dropout=0.05`를 사용했습니다.
  - 템플릿은 `<|im_start|><|assistant|>`를 적용했습니다.



### 6. 추론 전략 (Inference Strategy)
- **`feature/ensemble`**: 최종 성능 향상을 위해 **Ensemble** 방식을 고안하여 적용했습니다.
    - **사용 모델**: 총 6개의 모델 csv 결과 활용 (최고 성능 모델 `0.7828`, 전문가 모델 `0.7791` 포함)
    - **알고리즘 (Logic)**:
        1. **Majority Vote (>=4)**: 6개 모델 중 4개 이상이 동일한 답을 선택하면 해당 답안 채택
        2. **Model Agreement**: 최상위 성능 모델(Strongest)과 전문가 모델(Expert)의 답이 일치하면 채택
        3. **Fallback**: 위 조건 만족 불가 시, 최상위 성능 모델의 답안 선택

- **`experiment/MoA`**: 
    - MoA:  SKT A.X로 문제별 description(보조정보)을 생성하고, EXAONE으로 3가지 모드(기본 / descriptions+reasoning / descriptions+non-reasoning) 병렬 추론 → 다수결로 최종 답을 결정하는 MoA 파이프라인을 구현했습니다.
    - DAPT(Domain-Adaptive Pretraining): DAPT용 데이터 로드·전처리, 트레이너/학습 스크립트 및 설정 파일을 포함합니다.

</details>

<br>

## 🔗 References
- AI, 수능에 도전하다: KoNET
    - [https://clova.ai/tech-blog/ai-수능에-도전하다-konet](https://clova.ai/tech-blog/ai-%EC%88%98%EB%8A%A5%EC%97%90-%EB%8F%84%EC%A0%84%ED%95%98%EB%8B%A4-konet)
- Mixture-of-agents enhances large language model capabilities (Wang, L., et al., 2024)
    - https://arxiv.org/abs/2406.04692
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2022)
    - https://arxiv.org/abs/2201.11903
- On the Impact of Fine-Tuning on Chain-of-Thought Reasoning (2024)
    - https://arxiv.org/abs/2411.15382
- Fine-Tuning with Divergent Chains of Thought Boosts Reasoning Through Self-Correction in Language Models (2024)
    - https://arxiv.org/abs/2407.03181
- Focal Loss for Dense Object Detection (2017)
    - https://arxiv.org/abs/1708.02002
- NEFTune: Noisy Embeddings Improve Instruction Finetuning (2023)
    - https://arxiv.org/abs/2310.05914
- Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)
    - https://arxiv.org/abs/2307.03172
