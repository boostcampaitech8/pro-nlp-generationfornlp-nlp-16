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


def process_for_eval(df: pd.DataFrame) -> list:
    """train.csv 형식 데이터 전처리"""
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
            "answer": problems["answer"],
            "question_plus": question_plus,
        })
    
    return processed_data


def process_test_dataset_from_list(data_list: list) -> list:
    """기존 inference용 데이터셋 생성"""
    test_dataset = []
    
    for data in data_list:
        _id = data["id"]
        paragraph = data["paragraph"]
        question = data["question"]
        choices = data["choices"]
        question_plus = data.get("question_plus", "")
        
        choices_str = "\n".join([f"{idx+1} - {choice}" for idx, choice in enumerate(choices)])
        
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

def evaluate_results(infer_results: list, data_list: list) -> dict:
    """예측 결과 평가"""
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


# [수정] Stage 1: 분석용 (question_plus 추가)
STAGE1_PROMPT = """지문:
{paragraph}

질문:
{question}
{question_plus}

선택지:
{choices}

각 선택지가 맞는지 틀린지 간단히 분석하세요.

분석:"""

# Stage 2: 정답 추출용
STAGE2_PROMPT = """{analysis}

위 분석을 바탕으로 정답 번호만 말하세요.
정답:"""

def extract_answer_stage2(text: str) -> str:
    """Stage 2 출력에서 정답 추출"""
    match = re.search(r'([1-5])', text)
    if match:
        return match.group(1)
    return "1"


def run_inference_cot_twostage(
    model, 
    tokenizer, 
    data_list: list,
    stage1_max_tokens: int = 200,
    stage2_max_tokens: int = 10,
) -> list:
    """Two-Stage CoT 추론"""
    results = []
    model.eval()
    
    for idx, data in enumerate(tqdm(data_list, desc="Two-Stage CoT")):
        _id = data["id"]
        paragraph = data["paragraph"]
        question = data["question"]
        choices = data["choices"]
        question_plus = data.get("question_plus", "")
        
        if isinstance(choices, list):
            choices_str = "\n".join([f"{i+1} - {c}" for i, c in enumerate(choices)])
        else:
            choices_str = choices
        
        # ===== Stage 1: 분석 생성 =====
        # [수정] question_plus 추가
        stage1_prompt = STAGE1_PROMPT.format(
            paragraph=paragraph,
            question=question,
            question_plus=question_plus,
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
                do_sample=False,
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
        
        results.append({"id": _id, "answer": answer})
        
        # 처음 3개 샘플 출력
        if idx < 3:
            print(f"\n{'='*60}")
            print(f"[샘플 {idx+1}]")
            print(f"Stage1 분석 (마지막 150자): ...{analysis[-150:]}")
            print(f"Stage2 출력: '{answer_text}'")
            print(f"추출된 정답: {answer}")
            print(f"{'='*60}")
    
    return results

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cot_mode = cfg.inference.get("cot", False)
    mode_name = "Two-Stage CoT" if cot_mode else "기존"
    
    print("=" * 60)
    print(f"📌 EVALUATION MODE: {mode_name} 방식으로 validation set 평가")
    if cot_mode:
        print("Stage 1: 분석 생성 (200 tokens)")
        print("Stage 2: 정답 추출 (10 tokens)")
    print("=" * 60)
    
    original_cwd = hydra.utils.get_original_cwd()
    checkpoint_path = find_checkpoint(cfg, original_cwd)
    print(f"Checkpoint: {checkpoint_path}")
    
    print("\nLoading model...")
    model, tokenizer = load_model_for_inference(
        checkpoint_path, 
        torch_dtype=cfg.inference.torch_dtype
    )
    
    train_path = hydra.utils.to_absolute_path(cfg.data.train_path)
    print(f"\nLoading train data from {train_path}...")
    full_df = pd.read_csv(train_path)
    
    _, val_df = train_test_split(
        full_df,
        test_size=cfg.data.get("test_size", 0.1),
        random_state=cfg.seed
    )
    print(f"Validation size: {len(val_df)}")
    
    print("Processing data...")
    data_list = process_for_eval(val_df)
    
    print(f"\nRunning {mode_name} inference...")
    
    if cot_mode:
        infer_results = run_inference_cot_twostage(model, tokenizer, data_list)
    else:
        test_dataset = process_test_dataset_from_list(data_list)
        infer_results = run_inference(model, tokenizer, test_dataset)
    
    metrics = evaluate_results(infer_results, data_list)
    print_evaluation(metrics, mode=mode_name)
    
    output_path = f"output_eval_{mode_name.replace(' ', '_')}.csv"
    save_predictions(infer_results, output_path)
    print(f"\nSaved to: {os.getcwd()}/{output_path}")


if __name__ == "__main__":
    main()
