"""
데이터셋 품질 검증 스크립트 (OpenAI API 사용)

사용법: 
    uv run python scripts/validate_extraction.py \
        --input data/processed/historyexam.csv \
        --output data/processed/historyexam_validated.csv
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


SOLVE_PROMPT = """다음 문제를 풀고 분석해주세요.

[지문]
{paragraph}

[추가 정보]
{question_plus}

[질문]
{question}

[선택지]
{choices}

아래 JSON 형식으로만 출력하세요:
{{
  "answer": 정답 번호(1~5),
  "solvable": 문제를 풀 수 있는지 (true/false),
  "issue": "문제가 있다면 이유 (없으면 빈 문자열)"
}}

issue 유형 예시:
- "missing_paragraph": 지문이 비어있거나 불완전
- "incomplete_choices": 선택지 내용이 누락되거나 깨짐
- "missing_context": 이미지/도표 설명이 필요하지만 없음
- "extraction_error": 글자 깨짐, 이상한 문자
- "": 정상적으로 풀 수 있음

JSON만 출력:"""


def solve_question(client: OpenAI, row: pd.Series) -> dict:
    """GPT-4o로 문제 풀기 + 이슈 분석"""
    problems = json.loads(row["problems"])
    choices = problems.get("choices", [])
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])

    prompt = SOLVE_PROMPT.format(
        paragraph=row["paragraph"] or "(none)",
        question_plus=problems.get("question_plus", "") or "(none)",
        question=problems.get("question", ""),
        choices=choices_str,
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    return json.loads(content.strip())


def main():
    parser = argparse.ArgumentParser(description="Dataset quality validation")
    parser.add_argument("--input", "-i", required=True, help="Input CSV path")
    parser.add_argument(
        "--output", "-o", help="Output CSV path (default: adds _validated suffix)"
    )
    parser.add_argument(
        "--incorrect-only", help="Path to save incorrect questions only"
    )
    parser.add_argument(
        "--delay", type=float, default=0.3, help="Delay between requests"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_stem(input_path.stem + "_validated")
    )

    client = OpenAI()

    df = pd.read_csv(input_path)
    print(f"Loaded: {len(df)} questions")

    gpt_answers = []
    solvable_list = []
    issue_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        try:
            result = solve_question(client, row)
            gpt_answers.append(result.get("answer", 0))
            solvable_list.append(result.get("solvable", True))
            issue_list.append(result.get("issue", ""))
            time.sleep(args.delay)
        except Exception as e:
            print(f"  [WARNING] {row['id']}: {e}")
            gpt_answers.append(0)
            solvable_list.append(False)
            issue_list.append(f"api_error: {str(e)}")

    original_answers = []
    for _, row in df.iterrows():
        problems = json.loads(row["problems"])
        original_answers.append(problems.get("answer"))

    df["gpt_answer"] = gpt_answers
    df["original_answer"] = original_answers
    df["is_correct"] = df["original_answer"] == df["gpt_answer"]
    df["solvable"] = solvable_list
    df["issue"] = issue_list

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    if args.incorrect_only:
        incorrect_df = df[~df["is_correct"]]
        incorrect_df.to_csv(args.incorrect_only, index=False, encoding="utf-8-sig")

    correct_count = df["is_correct"].sum()
    incorrect_count = len(df) - correct_count
    unsolvable_count = (~df["solvable"]).sum()

    print("")
    print("=" * 50)
    print("Validation Summary")
    print("=" * 50)
    print(f"  Total: {len(df)}")
    print(f"  GPT correct: {correct_count} ({correct_count/len(df)*100:.1f}%)")
    print(f"  GPT incorrect: {incorrect_count} ({incorrect_count/len(df)*100:.1f}%)")
    print(f"  Unsolvable: {unsolvable_count} ({unsolvable_count/len(df)*100:.1f}%)")

    issue_counts = df[df["issue"] != ""]["issue"].value_counts()
    if len(issue_counts) > 0:
        print("")
        print("Issues by type:")
        for issue, count in issue_counts.head(10).items():
            print(f"  - {issue}: {count}")

    print("")
    print(f"Saved: {output_path}")
    if args.incorrect_only:
        print(f"Saved (incorrect only): {args.incorrect_only}")


if __name__ == "__main__":
    main()
