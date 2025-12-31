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


def process_test_data_for_cot(df: pd.DataFrame) -> list:
    """test.csv 형식 데이터 전처리"""
    processed_data = []
    
    for _, row in df.iterrows():
        problems = row["problems"]
        if isinstance(problems, str):
            problems = ast.literal_eval(problems)
        
        # question_plus는 row 레벨에 있음! (problems 안이 아님)
        question_plus = row.get("question_plus", "")
        if question_plus is None or (isinstance(question_plus, float) and pd.isna(question_plus)):
            question_plus = ""
        
        processed_data.append({
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": problems["question"],
            "choices": problems["choices"],
            "question_plus": question_plus,
        })
    
    return processed_data


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


# ============================================
# 학습 때와 동일한 프롬프트!
# ============================================

PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""


def extract_answer(text: str) -> str:
    """생성된 텍스트에서 정답 추출"""
    text = text.strip()
    
    # "정답은 X번" 패턴
    match = re.search(r'정답은?\s*(\d)\s*번', text)
    if match:
        return match.group(1)
    
    # "X번이다" 패턴
    match = re.search(r'(\d)\s*번이다', text)
    if match:
        return match.group(1)
    
    # "따라서 X" 패턴
    match = re.search(r'따라서\s*(\d)', text)
    if match:
        return match.group(1)
    
    # 마지막에 나오는 1-5 숫자
    matches = re.findall(r'[1-5]', text)
    if matches:
        return matches[-1]
    
    return "1"


def run_inference_cot(
    model, 
    tokenizer, 
    data_list: list,
    max_new_tokens: int = 400,  # reasoning 길이 고려해서 넉넉하게
    verbose: bool = True,
) -> tuple:
    """Single-Stage CoT 추론 (학습 패턴과 동일)"""
    infer_results = []
    reasoning_results = []
    
    # Negative 문제 패턴
    negative_patterns = ["않는 것", "않은 것", "아닌 것", "적절하지 않은", "옳지 않은", "일치하지 않는", "잘못된 것", "틀린 것", "부적절한"]
    
    model.eval()
    
    for idx, data in enumerate(tqdm(data_list, desc="CoT Inference")):
        _id = data["id"]
        paragraph = data["paragraph"]
        question = data["question"]
        choices = data["choices"]
        question_plus = data.get("question_plus", "")
        
        if isinstance(choices, list):
            choices_str = "\n".join([f"{i+1} - {c}" for i, c in enumerate(choices)])
        else:
            choices_str = choices
        
        # 학습 때와 동일한 프롬프트!
        if question_plus:
            prompt = PROMPT_QUESTION_PLUS.format(
                paragraph=paragraph,
                question=question,
                question_plus=question_plus,
                choices=choices_str,
            )
        else:
            prompt = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=paragraph,
                question=question,
                choices=choices_str,
            )
        
        # Negative 문제 감지
        is_negative = any(p in question for p in negative_patterns)
        
        if is_negative:
            system_content = "지문을 읽고 질문의 답을 구하세요. 주의: 이 문제는 '틀린 것' 또는 '적절하지 않은 것'을 찾는 문제입니다. 지문과 일치하지 않는 선택지를 고르세요."
        else:
            system_content = "지문을 읽고 질문의 답을 구하세요."
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
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
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        answer = extract_answer(generated)
        
        infer_results.append({"id": _id, "answer": answer})
        reasoning_results.append({
            "id": _id, 
            "answer": answer, 
            "reasoning": generated,
        })
        
        # 처음 3개 샘플 출력
        if verbose and idx < 3:
            print(f"\n{'='*60}")
            print(f"[샘플 {idx+1}]")
            print(f"생성된 응답:\n{generated[:500]}...")
            print(f"추출된 정답: {answer}")
            print(f"{'='*60}")
    
    return infer_results, reasoning_results


def save_predictions_cot(infer_results: list, reasoning_results: list, output_path: str):
    """CoT 결과 저장"""
    pd.DataFrame(infer_results).to_csv(output_path, index=False)
    print(f"제출용 저장: {output_path}")
    
    reasoning_path = output_path.replace(".csv", "_reasoning.csv")
    pd.DataFrame(reasoning_results).to_csv(reasoning_path, index=False)
    print(f"풀이 포함 저장: {reasoning_path}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=" * 60)
    print("CoT Inference (학습 프롬프트와 동일)")
    print("=" * 60)
    
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
    
    # 데이터 전처리
    test_dataset = process_test_data_for_cot(test_df)
    
    # 추론
    print("\nRunning CoT inference...")
    infer_results, reasoning_results = run_inference_cot(
        model=model,
        tokenizer=tokenizer,
        data_list=test_dataset,
        max_new_tokens=400,
        verbose=True,
    )
    
    # 저장
    output_path = cfg.inference.output_file.replace(".csv", "_cot.csv")
    print(f"\nSaving results to {output_path}...")
    save_predictions_cot(infer_results, reasoning_results, output_path)
    
    print(f"\n{'='*60}")
    print("CoT Inference completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
