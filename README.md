# MoA(Mixture-of-Agents) 기반 수능형 문제 추론 파이프라인

소형 모델이 지문 정보를 정제·요약·필터링하고
대형 추론 특화 모델이 해당 정보만을 활용해 최종 정답을 선택하는
다단계(MoA) 추론 파이프라인을 구현 

## 핵심 아이디어
지문 + 문제 + 선택지만 대형 모델에 입력하는 대신,
문제 해결 및 reasoning 과정에 도움이 되는 관점이나 정보를 같이 제공하면
정답률(Accuracy) 향상을 기대할 수 있을 것이다.

## 전체 파이프라인 구조
```
  [AIHub 국어 문제 데이터]
          │
          ▼
      [DAPT 학습]
  (sktAX 도메인 적응)
          │
          ▼
      [입력 문제]
          │
          ▼
  [sktAX-4.0-light (DAPT)]
(필터링 · 요약 · 근거 정리)
          │ └─ 핵심 근거 문장
          │ └─ 오답 선택지 제거 (1~2개)
          │ └─ 구조화된 설명 2종 생성
          ▼
   [EXAONE-4.0-32B]
      (최종 추론기)
          │
          ▼
      [정답 선택]
```
## 프로젝트 파일 구조
```
korean_sat_solver/
├── conf
│   ├── config.yaml
│   ├── dapt
│   │   ├── config.yaml
│   │   ├── model.yaml
│   │   └── training.yaml
│   ├── inference
│   │   ├── default.yaml
│   │   └── description.yaml
│   ├── model
│   │   ├── gemma.yaml
│   │   └── sktAX.yaml
│   └── training
│       └── default.yaml
├── src
│   ├── dapt
│   │   ├── dataset.py
│   │   └── trainer.py
│   ├── data
│   │   ├── dataset.py
│   │   ├── description.py
│   │   └── preprocessing.py
│   ├── inference
│   │   ├── description_prompt.py
│   │   ├── exaone_prompts.py
│   │   ├── generate_description.py
│   │   └── predict.py
│   ├── model
│   │   └── model.py
│   ├── training
│   │   ├── custom_train.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── trainer.py
│   └── utils
│       └── seed.py
├── models
│   └── EXAONE-4.0-32B-Q5_K_M.gguf
├── outputs
│   ├── dapt
│   │   └── 2026-01-04
│   ├── inference_description
│   └── moa
├── data -> /data/ephemeral/home/shared/data
├── llama.cpp
├── train.py
├── train_dapt.py
├── inference.py
├── inference_description.py
├── inference_exaone.py
├── inference_pipeline.py
├── README.md
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## 모듈 설명

### 학습 모듈

1. **train_dapt.py** - DAPT (Domain-Adaptive Pre-Training) 학습
   - sktAX 모델을 국어 문제 도메인에 적응시키는 사전 학습
   - Causal Language Modeling (CLM) 방식
   - LoRA를 사용한 파라미터 효율적 학습 (기본값)
   - 설정: `conf/dapt/config.yaml`

2. **src/dapt/dataset.py** - DAPT 데이터 처리
   - AIHub 국어 문제 데이터 로드 및 전처리
   - 지문, 선택지, 해설을 연속 텍스트로 변환
   - 토크나이징 및 train/eval 분할

3. **src/dapt/trainer.py** - DAPT 학습 유틸리티
   - 모델/토크나이저 로드
   - LoRA 적용
   - Trainer 생성

### 추론 모듈

1. **inference_pipeline.py** - 통합 파이프라인
   - Step 1: description 생성 (inference_description.py 실행)
   - Step 2: EXAONE 추론 (inference_exaone.py 실행)

2. **inference_description.py** - Description 생성
   - sktAX 모델로 테스트 문제 분석 힌트 생성
   - 체크포인트 자동 탐색 (outputs/dapt, outputs/train)

3. **inference_exaone.py** - 최종 추론
   - 3가지 모드로 추론 후 다수결 투표:
     - EXAONE만 (reasoning, descriptions 없이)
     - EXAONE + descriptions (reasoning)
     - EXAONE + descriptions (non-reasoning)
   - 결과 저장: outputs/moa/submission.csv

4. **src/data/description.py** - Description 유틸리티
   - load_descriptions_json(): descriptions.json 로드
   - save_descriptions(): description 결과 저장

5. **src/inference/** - Inference 모듈
   - generate_description.py: description 생성 로직
   - exaone_prompts.py: EXAONE 프롬프트 템플릿

## 설치 및 환경 설정

### 자동 설정

- `setup.sh` 스크립트로 환경을 자동 설정합니다.
```bash
bash setup.sh
```

### 수동 설정

1. 작업 공간 및 캐시 경로 설정

```bash
# 작업 공간 설정
WORK_DIR="/data/ephemeral/home/workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# 임시 파일 경로 설정
export TMPDIR="/data/ephemeral/tmp"
mkdir -p "$TMPDIR"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

# 캐시 경로 설정
export XDG_CACHE_HOME="/data/ephemeral/home/shared/cache"
export PIP_CACHE_DIR="/data/ephemeral/home/shared/cache/pip"
export UV_CACHE_DIR="/data/ephemeral/home/shared/cache/uv"
export HF_HOME="/data/ephemeral/home/shared/cache/huggingface"

# 캐시 디렉토리 생성
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HOME"
```

2.  CUDA Toolkit 설치

```bash
# CUDA 12.2 다운로드 및 설치
cd workspace
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
chmod +x cuda_12.2.0_535.54.03_linux.run
sh cuda_12.2.0_535.54.03_linux.run --silent --toolkit

# 심볼릭 링크 생성
ln -sf /usr/local/cuda-12.2 /usr/local/cuda

# 환경 변수 설정
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDACXX="/usr/local/cuda/bin/nvcc"
```

3. python 환경

```bash
uv sync
```

4. llama.cpp 빌드

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
rm -rf build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 6
```


## 사용법

### 모델 학습

#### DAPT (Domain-Adaptive Pre-Training)

sktAX 모델을 국어 문제 도메인에 적응시키는 사전 학습:

```bash
uv run train_dapt.py
```

**설정 파일**: `conf/dapt/config.yaml`
- 모델: `skt/A.X-4.0-Light`
- 데이터: `data/aihub_workbook_final.csv`
- 학습 방식: LoRA (기본값)
- 출력: `outputs/dapt/YYYY-MM-DD/HH-MM-SS/`

**주요 설정**:
- `max_length: 1024` (메모리 최적화)
- `gradient_accumulation_steps: 16`
- `gradient_checkpointing: true`
- `fp16: true`

학습된 모델은 `inference_description.py`에서 자동으로 탐색되어 사용됩니다.

### 추론 파이프라인

1. 모델 다운로드

```bash
cd workspace 

uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='LGAI-EXAONE/EXAONE-4.0-32B-GGUF',
    filename='EXAONE-4.0-32B-Q4_K_M.gguf',
    local_dir='./models',
    local_dir_use_symlinks=False
)
"
```

2. 추론 엔진 실행

- EXAONE 4.0 32B 서버 실행 

```bash
./llama.cpp/build/bin/llama-server \
  -m ./models/EXAONE-4.0-32B-Q5_K_M.gguf \
  -c 51000 \
  -np 3 \
  -cb \
  -fa on \
  --port 8000 \
  --host 0.0.0.0
```

3. 추론 진행

```bash
uv run inference_pipeline.py                      # 전체 파이프라인 실행
uv run inference_pipeline.py --skip-description   # description 건너뛰고 EXAONE만 실행
uv run inference_pipeline.py --description-only   # description만 생성
```

## 참고사항
Paper: [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)
