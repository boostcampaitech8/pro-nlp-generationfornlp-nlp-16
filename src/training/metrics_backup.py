import torch
import numpy as np
# import evaluate
# 변경 : 대회 기준에 맞는 평가지표인 f1으로, accuracy도 확인할 수 있게 추가
from sklearn.metrics import f1_score, accuracy_score


# 정답 토큰 매핑
int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}


def preprocess_logits_for_metrics(logits, labels, tokenizer):
    """
    모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
    """
    logits = logits if not isinstance(logits, tuple) else logits[0]
    logit_idx = [tokenizer.vocab["1"], tokenizer.vocab["2"], tokenizer.vocab["3"], tokenizer.vocab["4"], tokenizer.vocab["5"]]
    logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
    return logits


def compute_metrics(evaluation_result, tokenizer):
    """
    metric 계산 함수
    """
    logits, labels = evaluation_result

    # 토큰화된 레이블 디코딩
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
    labels = list(map(lambda x: int_output_map[x], labels))

    # 소프트맥스 함수를 사용하여 로그트 변환
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs, axis=-1)

    # 정확도 계산
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)

    # acc = acc_metric.compute(predictions=predictions, references=labels)
    return {"f1" : f1, "accuracy": acc}


def get_metrics_functions(tokenizer):
    """
    Get metric functions with tokenizer bound
    """
    def _preprocess_logits_for_metrics(logits, labels):
        return preprocess_logits_for_metrics(logits, labels, tokenizer)

    def _compute_metrics(evaluation_result):
        return compute_metrics(evaluation_result, tokenizer)

    return _preprocess_logits_for_metrics, _compute_metrics
