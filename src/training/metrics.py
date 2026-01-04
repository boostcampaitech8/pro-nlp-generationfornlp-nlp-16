import re
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}


def extract_answer_from_text(text: str) -> str:
    """
    reasoning 텍스트에서 정답 번호 추출
    """
    text = text.strip()
    
    if text in ["1", "2", "3", "4", "5"]:
        return text
    
    match = re.search(r'정답은\s*(\d)\s*번', text)
    if match:
        return match.group(1)
    
    match = re.search(r'(\d)\s*번이다', text)
    if match:
        return match.group(1)
    
    match = re.search(r'(\d)\s*번입니다', text)
    if match:
        return match.group(1)
    
    matches = re.findall(r'[1-5]', text)
    if matches:
        return matches[-1]
    
    return "1"


def preprocess_logits_for_metrics(logits, labels, tokenizer):
    """
    모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
    """
    logits = logits if not isinstance(logits, tuple) else logits[0]
    logit_idx = [tokenizer.vocab["1"], tokenizer.vocab["2"], tokenizer.vocab["3"], tokenizer.vocab["4"], tokenizer.vocab["5"]]
    logits = logits[:, -2, logit_idx]
    return logits


def compute_metrics(evaluation_result, tokenizer):
    """
    metric 계산 함수

    """
    logits, labels = evaluation_result

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 범용 처리
    def clean_label(x):
        x = x.strip()
        if "<end_of_turn>" in x:
            x = x.split("<end_of_turn>")[0].strip()
        return x
    
    labels = list(map(clean_label, labels))
    labels = list(map(extract_answer_from_text, labels))
    
    def safe_map(x):
        if x in int_output_map:
            return int_output_map[x]
        return 0
    
    labels = list(map(safe_map, labels))

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs, axis=-1)

    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)

    return {"f1": f1, "accuracy": acc}


def get_metrics_functions(tokenizer):
    """
    Get metric functions with tokenizer bound
    """
    def _preprocess_logits_for_metrics(logits, labels):
        return preprocess_logits_for_metrics(logits, labels, tokenizer)

    def _compute_metrics(evaluation_result):
        return compute_metrics(evaluation_result, tokenizer)

    return _preprocess_logits_for_metrics, _compute_metrics

