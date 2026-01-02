# Unsloth 통합 가이드

## 개요

이 프로젝트에 Unsloth 라이브러리가 성공적으로 통합되었습니다. Unsloth는 LLM 학습 및 추론 속도를 크게 향상시키는 최적화 라이브러리입니다.

## 주요 변경 사항

### 1. 모델 로딩 (`src/model/model.py`)
- `load_model_and_tokenizer()`: 학습용 모델 로딩에 Unsloth 지원 추가
- `load_model_for_inference()`: 추론용 모델 로딩에 Unsloth 지원 추가
- `apply_lora_to_model()`: Unsloth의 최적화된 LoRA 적용 함수 추가

### 2. 트레이너 (`src/training/trainer.py`)
- `get_trainer()`: Unsloth 최적화 SFTTrainer 지원 추가

### 3. 학습 스크립트 (`train.py`)
- Unsloth 사용 시 LoRA를 모델에 직접 적용하는 로직 추가
- Config에서 `use_unsloth` 옵션 읽기

### 4. 추론 스크립트 (`inference.py`)
- Unsloth 모델 로딩 및 최적화된 추론 지원

### 5. Config 파일
- `conf/model/gemma.yaml`: Unsloth 활성화 (기본)
- `conf/model/gemma_no_unsloth.yaml`: Unsloth 비활성화 (비교용)
- `conf/model/qwen_qlora.yaml`: Unsloth 활성화

## 사용 방법

### 학습

#### Unsloth 사용 (권장)
```bash
python train.py model=qwen_qlora
```

`conf/model/gemma.yaml`에서 다음 설정 확인:
```yaml
use_unsloth: true
torch_dtype: "float16"  # V100에서는 float16 필수
peft:
  target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
```

### 추론

학습과 동일한 설정으로 추론:
```bash
python inference.py model=qwen_qlora
```

## V100 GPU 최적화 설정

V100 GPU는 bfloat16을 지원하지 않으므로 **반드시 float16을 사용**해야 합니다.

모든 config 파일에서:
```yaml
torch_dtype: "float16"  # bfloat16 사용 불가
```

## Unsloth 장점

1. **학습 속도 향상**: 기존 대비 2-5배 빠른 학습
2. **메모리 효율**: 더 적은 VRAM으로 더 큰 모델 학습 가능
3. **자동 최적화**: Gradient checkpointing, Flash Attention 등 자동 적용
4. **추론 최적화**: `FastLanguageModel.for_inference()` 사용 시 더 빠른 추론

## 성능 비교

| 방식 | 학습 속도 | 메모리 사용 |
|------|----------|------------|
| 기존 (transformers + PEFT) | 1x | 1x |
| Unsloth | 2-5x | 0.7-0.9x |

## 주의사항

1. **torch 버전**: torch 2.6.0과 호환됩니다
2. **CUDA 버전**: cu121 (CUDA 12.1) 사용
3. **target_modules**: Unsloth 사용 시 더 많은 모듈에 LoRA 적용 권장
   - 기존: `['q_proj', 'k_proj']`
   - Unsloth: `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']`

## 문제 해결

### Unsloth를 사용할 수 없는 경우
코드는 자동으로 기존 방식으로 fallback됩니다:
```
⚠️ Unsloth requested but not available. Using standard transformers.
```

### bfloat16 에러
V100에서 bfloat16 사용 시 에러 발생:
```yaml
# 수정 전 (에러)
torch_dtype: "bfloat16"

# 수정 후 (정상)
torch_dtype: "float16"
```

## 추가 리소스

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth 문서](https://docs.unsloth.ai/)

