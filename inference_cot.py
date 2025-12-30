"""
CoT (Chain of Thought) 추론 스크립트

test.csv로 CoT 방식 추론 (제출용)

사용법:
    uv run python inference_cot.py
"""
import os
import ast
import re
import torch
import hydra
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig

from src.model import load_model_for_inference
from src.inference import save_predictions


# =============================================================================
# 데이터 전처리 함수
# =============================================================================
def process_test_data_for_cot(df: pd.DataFrame) -> list:
    """
    test.csv 형식 데이터 전처리
    
    test.csv 구조:
        id, paragraph, problems, question_plus
        
    problems 컬럼:
        "{'question': '...', 'choices': [...]}"  (answer 없음)
    """
    processed_data = []
    
    for _, row in df.iterrows():
        # problems 컬럼 파싱
        problems = row["problems"]
        if isinstance(problems, str):
            problems = ast.literal_eval(problems)
        
        # question_plus 처리 (NaN 체크)
        question_plus = row.get("question_plus", "")
        if pd.isna(question_plus):
            question_plus = ""
        
        processed_data.append({
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": problems["question"],
            "choices": problems["choices"],
            "question_plus": question_plus,
        })
    
    return processed_data


# =============================================================================
# 체크포인트 찾기
# =============================================================================
def find_latest_checkpoint(original_cwd: str) -> str:
    """가장 최근 체크포인트 자동 탐색"""
    outputs_roots = [
        os.path.join(original_cwd, "outputs"),
        os.path.join(original_cwd, "outputs", "train"),
    ]

    for outputs_root in outputs_roots:
        if not os.path.exists(outputs_root):
            continue

        dates = sorted([d for d in os.listdir(outputs_root)
                        if os.path.isdir(os.path.join(outputs_root, d))], reverse=True)

        for date in dates:
            date_dir = os.path.join(outputs_root, date)
            times = sorted([t for t in os.listdir(date_dir)
                            if os.path.isdir(os.path.join(date_dir, t))], reverse=True)

            for time in times:
                run_dir = os.path.join(date_dir, time)
                if any(d.startswith("checkpoint-") for d in os.listdir(run_dir)):
                    return run_dir

    raise ValueError("No checkpoint found in outputs/")


def get_best_checkpoint(checkpoint_path: str, checkpoint_step: str) -> str:
    """best 체크포인트 경로 반환"""
    if "checkpoint-" in os.path.basename(checkpoint_path):
        return checkpoint_path

    if not os.path.isdir(checkpoint_path):
        return checkpoint_path

    checkpoints = [d for d in os.listdir(checkpoint_path) if d.startswith("checkpoint-")]

    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_path}")

    if checkpoint_step == "best":
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        return os.path.join(checkpoint_path, checkpoints[-1])
    else:
        target = f"checkpoint-{checkpoint_step}"
        if target in checkpoints:
            return os.path.join(checkpoint_path, target)
        raise ValueError(f"Checkpoint {target} not found")


# =============================================================================
# CoT 관련 함수
# =============================================================================
COT_PROMPT_TEMPLATE = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

위 문제를 단계별로 분석하고 정답을 고르세요.

[분석]
1. 지문의 핵심 내용을 파악하세요.
2. 각 선택지를 지문과 비교하세요.

[중요] 분석 후 반드시 아래 형식으로 결론을 작성하세요:
"따라서 정답은 N번이다." (N은 1, 2, 3, 4, 5 중 하나, 4번까지만 있을 수 도 있음)

분석:"""


def extract_answer_improved(text: str) -> str:

    patterns_p1 = [
        r'정답[은는이가]?\s*(\d)\s*번',
        r'정답\s*[:\-]\s*(\d)',
        r'정답\s*(\d)\s*번',
    ]
    for pattern in patterns_p1:
        match = re.search(pattern, text)
        if match and match.group(1) in '12345':
            return match.group(1)
    
    patterns_p2 = [
        r'(\d)번이다',
        r'(\d)번입니다',
        r'(\d)번이\s*정답',
        r'(\d)번이\s*맞',
        r'(\d)번이\s*적절',
        r'(\d)번이\s*옳',
    ]
    for pattern in patterns_p2:
        match = re.search(pattern, text)
        if match and match.group(1) in '12345':
            return match.group(1)
    
    match = re.search(r'따라서[^.]*?(\d)\s*번', text)
    if match and match.group(1) in '12345':
        return match.group(1)
    
    sentences = text.strip().split('.')
    for sent in reversed(sentences):
        if sent.strip():
            match = re.search(r'(\d)\s*번', sent)
            if match and match.group(1) in '12345':
                return match.group(1)
    
    lines = text.strip().split('\n')
    for line in reversed(lines):
        if line.strip():
            match = re.search(r'(\d)\s*번', line)
            if match and match.group(1) in '12345':
                return match.group(1)
    
    return "1"


def run_inference_cot(
    model, 
    tokenizer, 
    data_list: list,
    max_new_tokens: int = 300,
    temperature: float = 0.3,
    verbose: bool = True,
) -> tuple:
    """CoT 방식 추론"""
    infer_results = []
    reasoning_results = []
    
    model.eval()
    
    for idx, data in enumerate(tqdm(data_list, desc="CoT Inference")):
        _id = data["id"]
        paragraph = data["paragraph"]
        question = data["question"]
        choices = data["choices"]
        
        if isinstance(choices, list):
            choices_str = "\n".join([f"{i+1} - {c}" for i, c in enumerate(choices)])
        else:
            choices_str = choices
        
        prompt = COT_PROMPT_TEMPLATE.format(
            paragraph=paragraph,
            question=question,
            choices=choices_str,
        )
        
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        answer = extract_answer(generated)
        
        infer_results.append({"id": _id, "answer": answer})
        reasoning_results.append({"id": _id, "answer": answer, "reasoning": generated})
        
        # 처음 3개 샘플 출력
        if verbose and idx < 3:
            print(f"\n[샘플 {idx+1}] 생성: {generated[:200]}...")
            print(f"추출된 정답: {answer}")
    
    return infer_results, reasoning_results


def save_predictions_cot(infer_results: list, reasoning_results: list, output_path: str):
    """CoT 결과 저장"""
    # 제출용 (id, answer)
    pd.DataFrame(infer_results).to_csv(output_path, index=False)
    print(f"제출용 저장: {output_path}")
    
    # 풀이 포함 (분석용)
    reasoning_path = output_path.replace(".csv", "_reasoning.csv")
    pd.DataFrame(reasoning_results).to_csv(reasoning_path, index=False)
    print(f"풀이 포함 저장: {reasoning_path}")


# =============================================================================
# 메인
# =============================================================================
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=" * 60)
    print("CoT (Chain of Thought) Prompting Inference")
    print("=" * 60)
    
    # 체크포인트 찾기
    original_cwd = hydra.utils.get_original_cwd()
    
    checkpoint_path = cfg.inference.checkpoint_dir
    checkpoint_step = cfg.inference.checkpoint_step
    
    if not checkpoint_path:
        print("Checkpoint path empty. Auto-detecting...")
        checkpoint_path = find_latest_checkpoint(original_cwd)
        print(f"Found: {checkpoint_path}")
    
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = hydra.utils.to_absolute_path(checkpoint_path)
    
    checkpoint_path = get_best_checkpoint(checkpoint_path, checkpoint_step)
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # 모델 로드
    print("\nLoading model...")
    torch_dtype = cfg.inference.torch_dtype
    model, tokenizer = load_model_for_inference(checkpoint_path, torch_dtype=torch_dtype)
    
    # 테스트 데이터 로드
    test_path = hydra.utils.to_absolute_path(cfg.data.test_path)
    print(f"\nLoading test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    print(f"Test size: {len(test_df)}")
    
    # 데이터 전처리 (problems 컬럼 파싱)
    test_dataset = process_test_data_for_cot(test_df)
    
    # CoT 추론
    print("\nRunning CoT inference...")
    infer_results, reasoning_results = run_inference_cot(
        model=model,
        tokenizer=tokenizer,
        data_list=test_dataset,
        max_new_tokens=150,
        temperature=0.3,
        verbose=True,
    )
    
    # 저장
    output_path = cfg.inference.output_file.replace(".csv", "_cot.csv")
    print(f"\nSaving results to {output_path}...")
    save_predictions_cot(infer_results, reasoning_results, output_path)
    
    print(f"\n{'='*60}")
    print("CoT Inference completed!")
    print(f"Results saved to: {os.getcwd()}/{output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
