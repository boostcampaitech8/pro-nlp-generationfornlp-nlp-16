import os
import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from openai import OpenAI
from rich.console import Console
from rich.progress import track, Progress
from dotenv import load_dotenv
import time
from tqdm import tqdm

load_dotenv()
console = Console()

# 품질 평가 프롬프트
QUALITY_EVALUATION_PROMPT = """당신은 교육 문제 품질 평가 전문가입니다. 주어진 독해 문제를 다음 기준으로 평가해주세요:

평가 기준:
1. **지문-문제 연관성** (1-5점): 문제가 지문의 내용을 적절히 다루고 있는가?
2. **문제 명확성** (1-5점): 문제가 명확하고 모호하지 않은가?
3. **선택지 품질** (1-5점): 선택지가 적절하고 오답이 그럴듯한가? 함정 선택지나 오개념을 활용한 정교한 오답이 있는가?
4. **정답 타당성** (1-5점): 정답이 명확하고 논리적으로 타당한가? 정답을 지문에서 근거하여 확실히 도출할 수 있는가?
5. **수능 문제와 유사성** (1-5점): 이 문제가 수능형 문제와 유사한가?
6. **문제의 변별력과 사고력 요구도** (1-5점): 단순 암기나 키워드 찾기가 아닌, 깊은 이해와 추론을 요구하는가? 여러 정보를 종합하거나 비판적 사고를 필요로 하는가?

각 항목에 대해 1점(매우 낮음)에서 5점(매우 높음)까지 점수를 매기고, 총점을 계산해주세요 (최대 30점). 또한, 전체 평가에 대한 간단한 이유를 2-3문장으로 설명해주세요.

**입력:**
- 지문: {paragraph}
- 문제: {question}
- 선택지: {choices}
- 정답: {answer}

**출력 형식 (JSON):**
{{
    "relevance_score": <1-5>,
    "clarity_score": <1-5>,
    "choices_quality_score": <1-5>,
    "answer_validity_score": <1-5>,
    "similarity_to_csat_score": <1-5>,
    "discrimination_thinking_score": <1-5>,
    "total_score": <6-30>,
    "reasoning": "<전체 평가 이유를 2-3문장으로 설명>"
}}

반드시 JSON 형식으로만 응답해주세요."""


def create_evaluation_prompt(row: Dict[str, Any]) -> str:
    """
    데이터 행을 받아 품질 평가 프롬프트를 생성

    Args:
        row: CSV의 한 행 데이터 (dict 형태)

    Returns:
        평가용 프롬프트 문자열
    """
    # problems 컬럼 파싱 (JSON 문자열 → dict)
    try:
        # Python dict 형식(작은따옴표)을 JSON 형식(큰따옴표)으로 변환
        import ast

        problems_str = row["problems"]

        # ast.literal_eval로 Python dict 파싱
        problems = ast.literal_eval(problems_str)

        question = problems.get("question", "")
        choices = problems.get("choices", [])
        answer = problems.get("answer", 0)

        # 선택지를 문자열로 변환
        choices_str = "\n".join(
            [f"{i+1}. {choice}" for i, choice in enumerate(choices)]
        )

        # 정답 번호에 해당하는 선택지 텍스트 가져오기
        answer_text = (
            choices[answer - 1] if 1 <= answer <= len(choices) else "알 수 없음"
        )

        return QUALITY_EVALUATION_PROMPT.format(
            paragraph=row["paragraph"][:500],  # 지문이 너무 길면 500자로 제한
            question=question,
            choices=choices_str,
            answer=f"{answer}번. {answer_text}",
        )
    except Exception as e:
        console.print(f"[red]프롬프트 생성 오류: {e}[/red]")
        return ""


def evaluate_single_question(
    client: OpenAI, row: Dict[str, Any], model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    단일 문제에 대한 품질 평가를 수행

    Args:
        client: OpenAI 클라이언트
        row: 평가할 데이터 행
        model: 사용할 GPT 모델 (기본값: gpt-4o)

    Returns:
        평가 결과 딕셔너리
    """
    prompt = create_evaluation_prompt(row)

    if not prompt:
        return {
            "id": row.get("id", "unknown"),
            "error": "프롬프트 생성 실패",
            "total_score": 0,
        }

    try:
        # GPT-5 모델은 temperature 파라미터를 지원하지 않음
        api_params = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 교육 문제 품질 평가 전문가입니다.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        # GPT-5 계열이 아닌 경우에만 temperature 추가
        if not model.startswith("gpt-5"):
            api_params["temperature"] = 0.3

        response = client.chat.completions.create(**api_params)

        # 응답 파싱
        result = json.loads(response.choices[0].message.content)
        result["id"] = row.get("id", "unknown")

        # logprobs 정보 추가 (가능한 경우)
        if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
            result["logprobs"] = response.choices[0].logprobs

        return result

    except Exception as e:
        console.print(f"[red]평가 오류 (ID: {row.get('id', 'unknown')}): {e}[/red]")
        return {"id": row.get("id", "unknown"), "error": str(e), "total_score": 0}


def evaluate_dataset(
    input_csv: str,
    output_csv: str,
    top_n: int = 2000,
    model: str = "gpt-4o",
    batch_size: int = 10,
    max_samples: int = None,
) -> pd.DataFrame:
    """
    전체 데이터셋에 대한 품질 평가를 수행하고 상위 N개를 선별

    Args:
        input_csv: 원본 CSV 파일 경로
        output_csv: 결과 CSV 파일 저장 경로
        top_n: 선별할 상위 문제 개수
        model: 사용할 GPT 모델
        batch_size: API 호출 간 대기 시간을 위한 배치 크기
        max_samples: 테스트용 최대 샘플 수 (None이면 전체)

    Returns:
        평가 결과가 포함된 DataFrame
    """
    console.print("[bold green]데이터 품질 평가 시작[/bold green]")

    # OpenAI 클라이언트 초기화
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY가 환경 변수에 없습니다.\n"
            ".env 파일을 생성하고 다음과 같이 설정하세요:\n"
            "OPENAI_API_KEY=your-api-key-here"
        )

    client = OpenAI(api_key=api_key)

    # 데이터 로드
    console.print(f"[cyan]데이터 로딩: {input_csv}[/cyan]")
    df = pd.read_csv(input_csv)

    # 테스트 모드인 경우 샘플 제한
    if max_samples:
        df = df.head(max_samples)
        console.print(f"[yellow]테스트 모드: {max_samples}개 샘플만 평가[/yellow]")

    console.print(f"[green]총 {len(df)}개 문제 로딩 완료[/green]")

    # 평가 결과 저장 리스트
    evaluation_results = []

    # 진행률 표시
    console.print(f"[cyan]모델: {model}[/cyan]")
    console.print("[cyan]평가 진행 중...[/cyan]")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="평가 중"):
        result = evaluate_single_question(client, row.to_dict(), model=model)
        evaluation_results.append(result)

        # Rate limit 방지를 위한 대기
        if (idx + 1) % batch_size == 0:
            time.sleep(1)

    # 평가 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(evaluation_results)

    # 원본 데이터와 평가 결과 병합
    df_with_scores = df.merge(results_df, on="id", how="left")

    # total_score 기준으로 정렬
    df_sorted = df_with_scores.sort_values("total_score", ascending=False)

    # 상위 N개 선별
    df_top = df_sorted.head(top_n)

    console.print(f"[green]평가 완료! 상위 {top_n}개 문제 선별[/green]")
    console.print(f"[cyan]평균 점수: {df_sorted['total_score'].mean():.2f}/25[/cyan]")
    console.print(f"[cyan]최고 점수: {df_sorted['total_score'].max():.2f}/25[/cyan]")
    console.print(f"[cyan]최저 점수: {df_sorted['total_score'].min():.2f}/25[/cyan]")

    # 결과 저장
    df_top.to_csv(output_csv, index=False, encoding="utf-8")
    console.print(f"[green]결과 저장 완료: {output_csv}[/green]")

    # 통계 정보 저장
    stats_file = output_csv.replace(".csv", "_stats.json")
    stats = {
        "total_evaluated": len(df),
        "top_n_selected": top_n,
        "average_score": float(df_sorted["total_score"].mean()),
        "max_score": float(df_sorted["total_score"].max()),
        "min_score": float(df_sorted["total_score"].min()),
        "model_used": model,
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    console.print(f"[green]통계 정보 저장: {stats_file}[/green]")

    return df_top


def main():
    """
    메인 실행 함수
    """
    import argparse

    parser = argparse.ArgumentParser(description="데이터 품질 평가 및 우수 문제 선별")

    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/aihub_workbook.csv",
        help="원본 CSV 파일 경로",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/aihub_workbook_top_quality.csv",
        help="결과 CSV 파일 저장 경로",
    )

    parser.add_argument(
        "--top_n", type=int, default=4000, help="선별할 상위 문제 개수 (기본값: 2000)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="사용할 GPT 모델 (기본값: gpt-5-mini, 옵션: gpt-5.2, gpt-4o-mini, o1-mini)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="API 호출 간 대기를 위한 배치 크기 (기본값: 10)",
    )

    parser.add_argument(
        "--test", action="store_true", help="테스트 모드: 100개 샘플만 평가"
    )

    args = parser.parse_args()

    # 테스트 모드 설정
    max_samples = 100 if args.test else None

    # 평가 실행
    evaluate_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        top_n=args.top_n,
        model=args.model,
        batch_size=args.batch_size,
        max_samples=max_samples,
    )

    console.print("[bold green]모든 작업 완료![/bold green]")


if __name__ == "__main__":
    main()
