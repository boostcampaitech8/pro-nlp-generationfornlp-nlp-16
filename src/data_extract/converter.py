import json
from pathlib import Path

import pandas as pd


def merge_questions_and_answers(
    questions: list[dict], answers: dict[str, int], exam_id: str
) -> list[dict]:
    """
    추출된 문제와 정답을 병합

    Args:
        questions: 추출된 문제 리스트
        answers: 문제번호 -> 정답 매핑
        exam_id: 시험 식별자 (id 접두사용)

    Returns:
        병합된 데이터 리스트
    """
    merged = []

    for q in questions:
        q_id = str(q.get("id", ""))
        answer = answers.get(q_id)

        if answer is not None:
            try:
                answer = int(answer)
            except (ValueError, TypeError):
                answer = None

        merged.append(
            {
                "id": f"{exam_id}_{q_id}",
                "paragraph": q.get("paragraph", ""),
                "question": q.get("question", ""),
                "choices": q.get("choices", []),
                "answer": answer,
                "question_plus": q.get("image_description", ""),
            }
        )

    return merged


def convert_to_train_format(data: list[dict]) -> pd.DataFrame:
    """
    train.csv 형식으로 변환

    Args:
        data: merge_questions_and_answers의 출력

    Returns:
        DataFrame with columns: id, paragraph, problems
        - problems: JSON string containing question, choices, answer, question_plus
    """
    rows = []

    for item in data:
        problems = {
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],
        }

        if item.get("question_plus"):
            problems["question_plus"] = item["question_plus"]

        rows.append(
            {
                "id": item["id"],
                "paragraph": item["paragraph"],
                "problems": json.dumps(problems, ensure_ascii=False),
            }
        )

    return pd.DataFrame(rows)


def save_dataset(df: pd.DataFrame, output_path: Path, encoding: str = "utf-8-sig"):
    """데이터셋 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding=encoding)
    print(f"Saved: {output_path} ({len(df)} questions)")


def validate_dataset(df: pd.DataFrame, choices_count: int = 5) -> list[str]:
    """
    데이터셋 검증

    Args:
        df: 검증할 DataFrame
        choices_count: 예상 선택지 수

    Returns:
        발견된 이슈 리스트
    """
    issues = []

    for _, row in df.iterrows():
        try:
            problems = json.loads(row["problems"])
        except json.JSONDecodeError:
            issues.append(f"{row['id']}: problems JSON parsing failed")
            continue

        choices = problems.get("choices", [])
        if len(choices) not in [choices_count - 1, choices_count]:
            issues.append(
                f"{row['id']}: {len(choices)} choices (expected: {choices_count})"
            )

        answer = problems.get("answer")
        if answer is None or answer not in range(1, choices_count + 1):
            issues.append(f"{row['id']}: invalid answer ({answer})")

        question = problems.get("question", "")
        if not question or not str(question).strip():
            issues.append(f"{row['id']}: empty question")

    return issues
