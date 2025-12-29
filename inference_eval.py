"""
Inference 평가 스크립트 (기존 inference.py 수정 없이)

기존 inference 로직을 그대로 사용하면서 validation set으로 평가만 추가

사용법:
    python inference_eval.py                    # 기존 방식 평가
    python inference_eval.py inference.cot=true # CoT 방식 평가
"""
import os
import ast
import re
import torch
import hydra
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

from src.model import load_model_for_inference
from src.inference import run_inference, save_predictions


# =============================================================================
# 데이터 전처리 함수
# =============================================================================
def process_for_eval(df: pd.DataFrame) -> list:
    """
    train.csv 형식 데이터 전처리
    
    train.csv 구조:
        id, paragraph, problems, question_plus
        
    problems 컬럼:
        "{'question': '...', 'choices': [...], 'answer': 1}"
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
            "answer": problems["answer"],
            "question_plus": question_plus,
        })
    
    return processed_data


def process_test_dataset_from_list(data_list: list) -> list:
    """
    기존 inference용 데이터셋 생성
    run_inference()가 기대하는 형식으로 변환
    """
    test_dataset = []
    
    for data in data_list:
        _id = data["id"]
        paragraph = data["paragraph"]
        question = data["question"]
        choices = data["choices"]
        question_plus = data.get("question_plus", "")
        
        # 선택지 문자열 생성
        choices_str = "\n".join([f"{idx+1} - {choice}" for idx, choice in enumerate(choices)])
        
        # 프롬프트 생성
        if question_plus:
            user_content = f"지문을 읽고 질문의 답을 구하세요.\n\n지문:\n{paragraph}\n\n질문:\n{question}\n\n{question_plus}\n\n선택지:\n{choices_str}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:"
        else:
            user_content = f"지문을 읽고 질문의 답을 구하세요.\n\n지문:\n{paragraph}\n\n질문:\n{question}\n\n선택지:\n{choices_str}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:"
        
        messages = [{"role": "user", "content": user_content}]
        
        test_dataset.append({
            "id": _id,
            "messages": messages,
            "len_choices": len(choices),
        })
    
    return test_dataset


# =============================================================================
# 평가 함수
# =============================================================================
def evaluate_results(infer_results: list, data_list: list) -> dict:
    """예측 결과 평가 (Accuracy, Macro F1)"""
    # id → 정답 매핑
    id_to_answer = {d["id"]: int(d["answer"]) for d in data_list}
    
    y_true = []
    y_pred = []
    
    for result in infer_results:
        _id = result["id"]
        pred = int(result["answer"])
        true = id_to_answer.get(_id)
        
        if true is not None:
            y_true.append(true)
            y_pred.append(pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    
    true_dist = Counter(y_true)
    pred_dist = Counter(y_pred)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "total": len(y_true),
        "correct": sum(1 for t, p in zip(y_true, y_pred) if t == p),
        "true_distribution": dict(sorted(true_dist.items())),
        "pred_distribution": dict(sorted(pred_dist.items())),
    }


def print_evaluation(metrics: dict, mode: str = "기존"):
    """평가 결과 출력"""
    print("\n" + "=" * 60)
    print(f"📊 [{mode}] EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro F1:  {metrics['f1_macro']:.4f}")
    print(f"  Correct:   {metrics['correct']} / {metrics['total']}")
    print("-" * 60)
    print(f"  정답 분포: {metrics['true_distribution']}")
    print(f"  예측 분포: {metrics['pred_distribution']}")
    print("=" * 60)


# =============================================================================
# 체크포인트 찾기
# =============================================================================
def find_checkpoint(cfg: DictConfig, original_cwd: str) -> str:
    """가장 최근 체크포인트 찾기"""
    checkpoint_path = cfg.inference.checkpoint_dir
    checkpoint_step = cfg.inference.checkpoint_step
    
    if not checkpoint_path:
        outputs_roots = [
            os.path.join(original_cwd, "outputs"),
            os.path.join(original_cwd, "outputs", "train"),
        ]
        
        found_path = None
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
                        found_path = run_dir
                        break
                if found_path:
                    break
            if found_path:
                break
        
        if not found_path:
            raise ValueError("No checkpoint found")
        checkpoint_path = found_path
    
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = hydra.utils.to_absolute_path(checkpoint_path)
    
    if "checkpoint-" not in os.path.basename(checkpoint_path) and os.path.isdir(checkpoint_path):
        if checkpoint_step == "best":
            checkpoints = [d for d in os.listdir(checkpoint_path) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                checkpoint_path = os.path.join(checkpoint_path, checkpoints[-1])
        else:
            target = f"checkpoint-{checkpoint_step}"
            possible = os.path.join(checkpoint_path, target)
            if os.path.exists(possible):
                checkpoint_path = possible
    
    return checkpoint_path


# =============================================================================
# CoT 관련 함수
# =============================================================================
COT_PROMPT_TEMPLATE = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

위 문제를 단계별로 분석하세요.
1. 지문에서 관련 정보를 찾으세요.
2. 각 선택지를 검토하세요.
3. 마지막에 "따라서 정답은 N번이다."로 끝내세요.

분석:"""


def extract_answer(text: str) -> str:
    """생성된 텍스트에서 정답 추출"""
    match = re.search(r'정답[은는이가]?\s*(\d)\s*번?', text)
    if match:
        return match.group(1)
    
    match = re.search(r'(\d)번이다', text)
    if match:
        return match.group(1)
    
    numbers = re.findall(r'[1-5]', text)
    if numbers:
        return numbers[-1]
    
    return "1"


def run_inference_cot(model, tokenizer, data_list: list) -> list:
    """CoT 방식 추론"""
    results = []
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
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        answer = extract_answer(generated)
        
        results.append({"id": _id, "answer": answer})
        
        # 처음 3개 샘플 출력
        if idx < 3:
            print(f"\n[샘플 {idx+1}] 생성: {generated[:200]}...")
            print(f"추출된 정답: {answer}")
    
    return results


# =============================================================================
# 메인
# =============================================================================
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # CoT 모드 확인
    cot_mode = cfg.inference.get("cot", False)
    mode_name = "CoT" if cot_mode else "기존"
    
    print("=" * 60)
    print(f"📌 EVALUATION MODE: {mode_name} 방식으로 validation set 평가")
    print("=" * 60)
    
    # 체크포인트 찾기
    original_cwd = hydra.utils.get_original_cwd()
    checkpoint_path = find_checkpoint(cfg, original_cwd)
    print(f"Checkpoint: {checkpoint_path}")
    
    # 모델 로드
    print("\nLoading model...")
    model, tokenizer = load_model_for_inference(
        checkpoint_path, 
        torch_dtype=cfg.inference.torch_dtype
    )
    
    # Validation 데이터 로드 (train에서 split)
    train_path = hydra.utils.to_absolute_path(cfg.data.train_path)
    print(f"\nLoading train data from {train_path}...")
    full_df = pd.read_csv(train_path)
    
    _, val_df = train_test_split(
        full_df,
        test_size=cfg.data.get("test_size", 0.1),
        random_state=cfg.seed
    )
    print(f"Validation size: {len(val_df)}")
    
    # 데이터 전처리 (problems 컬럼 파싱)
    print("Processing data...")
    data_list = process_for_eval(val_df)
    
    # 추론
    print(f"\nRunning {mode_name} inference...")
    
    if cot_mode:
        # CoT 방식
        infer_results = run_inference_cot(model, tokenizer, data_list)
    else:
        # 기존 방식
        test_dataset = process_test_dataset_from_list(data_list)
        infer_results = run_inference(model, tokenizer, test_dataset)
    
    # 평가
    metrics = evaluate_results(infer_results, data_list)
    print_evaluation(metrics, mode=mode_name)
    
    # 저장
    output_path = f"output_eval_{mode_name}.csv"
    save_predictions(infer_results, output_path)
    print(f"\nSaved to: {os.getcwd()}/{output_path}")


if __name__ == "__main__":
    main()