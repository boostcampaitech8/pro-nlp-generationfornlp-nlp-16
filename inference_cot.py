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


# Stage 1: 분석용 (간결하게)
STAGE1_PROMPT = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

각 선택지가 맞는지 틀린지 간단히 분석하세요.

분석:"""

# Stage 2: 정답 추출용
STAGE2_PROMPT = """{analysis}

위 분석을 바탕으로 정답 번호만 말하세요.
정답:"""


def extract_answer_stage2(text: str) -> str:
    """Stage 2 출력에서 정답 추출 (숫자만 나옴)"""
    # 첫 번째 1-5 숫자 찾기
    match = re.search(r'([1-5])', text)
    if match:
        return match.group(1)
    return "1"

def run_inference_cot_twostage(
    model, 
    tokenizer, 
    data_list: list,
    stage1_max_tokens: int = 200,  # 분석 (300→200 줄임)
    stage2_max_tokens: int = 10,   # 정답만
    verbose: bool = True,
) -> tuple:
    """Two-Stage CoT 추론"""
    infer_results = []
    reasoning_results = []
    
    model.eval()
    
    for idx, data in enumerate(tqdm(data_list, desc="Two-Stage CoT")):
        _id = data["id"]
        paragraph = data["paragraph"]
        question = data["question"]
        choices = data["choices"]
        
        if isinstance(choices, list):
            choices_str = "\n".join([f"{i+1} - {c}" for i, c in enumerate(choices)])
        else:
            choices_str = choices
        
        # ===== Stage 1: 분석 생성 =====
        stage1_prompt = STAGE1_PROMPT.format(
            paragraph=paragraph,
            question=question,
            choices=choices_str,
        )
        
        messages1 = [{"role": "user", "content": stage1_prompt}]
        inputs1 = tokenizer.apply_chat_template(
            messages1,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        
        with torch.no_grad():
            outputs1 = model.generate(
                inputs1,
                max_new_tokens=stage1_max_tokens,
                do_sample=False,  # greedy (빠르고 일관됨)
                pad_token_id=tokenizer.pad_token_id,
            )
        
        analysis = tokenizer.decode(outputs1[0][inputs1.shape[1]:], skip_special_tokens=True)
        
        # ===== Stage 2: 정답만 추출 =====
        stage2_prompt = STAGE2_PROMPT.format(analysis=analysis)
        
        messages2 = [{"role": "user", "content": stage2_prompt}]
        inputs2 = tokenizer.apply_chat_template(
            messages2,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        
        with torch.no_grad():
            outputs2 = model.generate(
                inputs2,
                max_new_tokens=stage2_max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        answer_text = tokenizer.decode(outputs2[0][inputs2.shape[1]:], skip_special_tokens=True)
        answer = extract_answer_stage2(answer_text)
        
        infer_results.append({"id": _id, "answer": answer})
        reasoning_results.append({
            "id": _id, 
            "answer": answer, 
            "analysis": analysis,
            "answer_raw": answer_text,
        })
        
        # 처음 3개 샘플 출력
        if verbose and idx < 3:
            print(f"\n{'='*60}")
            print(f"[샘플 {idx+1}]")
            print(f"Stage1 분석 (마지막 150자): ...{analysis[-150:]}")
            print(f"Stage2 출력: '{answer_text}'")
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
    print("Two-Stage CoT Inference")
    print("Stage 1: 분석 생성 (200 tokens)")
    print("Stage 2: 정답 추출 (10 tokens)")
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
    
    # Two-Stage 추론
    print("\nRunning Two-Stage CoT inference...")
    infer_results, reasoning_results = run_inference_cot_twostage(
        model=model,
        tokenizer=tokenizer,
        data_list=test_dataset,
        stage1_max_tokens=200,
        stage2_max_tokens=10,
        verbose=True,
    )
    
    # 저장
    output_path = cfg.inference.output_file.replace(".csv", "_cot.csv")
    print(f"\nSaving results to {output_path}...")
    save_predictions_cot(infer_results, reasoning_results, output_path)
    
    print(f"\n{'='*60}")
    print("Two-Stage CoT Inference completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

