import os
import argparse
import json
import time
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from rich.console import Console
from rich.progress import track
from dotenv import load_dotenv
from src.utils import set_seed
from src.data_gen.datagen_prompt import SYSTEM_PROMPT, create_prompt_for_article
from src.data_gen.data_loader import load_newspaper_data, load_book_data

load_dotenv()

console = Console()


def create_batch_request_file(
    articles: List[Dict[str, Any]],
    output_file: str,
    problems_per_article: int = 1,
    seed: int = 42,
) -> str:
    """
    OpenAI Batch API에 제출할 요청들을 JSONL 파일 형태로 생성.

    각 신문 기사(article)를 기반으로 하나 이상의 문제 생성 요청을 만들고,
    이를 OpenAI Batch API 규격에 맞는 JSONL 파일로 저장.

    Args:
        articles: 신문 기사 딕셔너리 리스트 (이미 샘플링됨)
        output_file: 생성된 JSONL 파일을 저장할 경로
        problems_per_article: 기사 1개당 생성할 문제 수
        seed: 무작위 샘플링 재현성을 위한 시드 값

    Returns:
        생성된 JSONL 배치 요청 파일의 경로
    """
    console.print(f"[cyan]Creating batch request file...[/cyan]")

    # 재현성을 위한 시드 설정
    set_seed(seed)
    console.print(f"[green]Random seed set to: {seed}[/green]")

    # 전달받은 모든 articles 사용
    selected_articles = articles
    console.print(f"[green]Using {len(articles)} articles[/green]")

    # Batch API 요청 객체 생성
    requests = []
    for article in track(selected_articles, description="Creating requests"):
        for problem_idx in range(problems_per_article):
            # custom_id는 이후 결과 매핑을 위한 고유 식별자
            custom_id = f"{article['article_id']}-q{problem_idx + 1}"

            # Fine-tuned 모델 사용 시 프롬프트 단순화
            # 이전 (base model 사용 시):
            # "model": "gpt-4o-mini",
            # "messages": [
            #     {"role": "system", "content": SYSTEM_PROMPT},
            #     {"role": "user", "content": create_prompt_for_article(article)},
            # ],
            #
            # 현재 (fine-tuned model 사용 시): 학습 데이터와 동일한 간단한 프롬프트 사용
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "ft:gpt-4o-mini-2024-07-18:ainfo:ksat-qa-finetuned:Cs0hEg5r",
                    "messages": [
                        {
                            "role": "system",
                            "content": "당신은 지문으로 수능형 객관식 문제를 생성하는 전문가입니다.",
                        },
                        {
                            "role": "user",
                            "content": f"""다음 지문을 읽고 JSON 형식으로 문제를 생성하세요.

**출력 형식:**
반드시 아래의 JSON 형식으로만 출력하세요. 다른 설명이나 주석은 포함하지 마세요.

{{
  "paragraph": "지문 (제공된 지문 그대로)",
  "question": "문제",
  "choices": ["선택지1", "선택지2", "선택지3", "선택지4", "선택지5"],
  "answer": 1,
  "question_plus": "내용..." 또는 null
}}

**중요**: 가능하면 question_plus 필드에 문제 해결에 도움이 되는 추가 정보를 포함하세요.
question_plus는 문제를 푸는 데 필요한 배경지식이나 판단 기준을 제공해야 합니다.
예: 핵심 개념 정의, 판단 기준, 이론/원리 설명, 배경지식 등

지문:
{article['content']}""",
                        },
                    ],
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"},
                },
            }

            requests.append(request)

    # JSONL 파일로 저장
    # Batch API는 요청을 JSONL (한 줄에 하나의 JSON) 형식으로 받음
    with open(output_file, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    console.print(
        f"[green]Created batch request file with {len(requests)} requests: {output_file}[/green]"
    )
    return output_file


def submit_batch(client: OpenAI, batch_file_path: str) -> str:
    """
    OpenAI Batch API에 배치 요청을 제출.

    사전에 생성된 JSONL 형식의 배치 요청 파일을 OpenAI에 업로드한 뒤,
    해당 파일을 입력으로 하는 배치 작업(batch job)을 생성.

    Args:
        client: OpenAI API 클라이언트 객체
        batch_file_path: Batch API용 JSONL 요청 파일 경로

    Returns:
        생성된 배치 작업의 배치 ID
    """
    console.print(f"[cyan]Uploading batch file...[/cyan]")

    # Batch 요청 파일 업로드 (Batch API는 먼저 JSONL 파일을 파일 리소스로 업로드해야 함)
    with open(batch_file_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    console.print(f"[green]File uploaded: {batch_input_file.id}[/green]")

    # Batch 작업 생성
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Korean SAT problem generation from newspaper articles"
        },
    )

    console.print(f"[green]Batch created: {batch.id}[/green]")
    console.print(f"[yellow]Status: {batch.status}[/yellow]")

    return batch.id


def monitor_batch(
    client: OpenAI, batch_id: str, poll_interval: int = 60
) -> Dict[str, Any]:
    """
    OpenAI Batch 작업의 상태를 주기적으로 확인하여 완료될 때까지 모니터링.

    배치 상태를 일정 간격(poll_interval)으로 조회하며,
    작업이 성공적으로 완료되면 배치 객체를 반환하고,
    실패/만료/취소 상태일 경우 예외를 발생시킵니다.

    Args:
        client: OpenAI API 클라이언트 객체
        batch_id: 모니터링할 배치 작업의 ID
        poll_interval: 상태 조회 간격(초 단위)

    Returns:
        완료된 배치 작업 객체
    """
    console.print(f"[cyan]Monitoring batch {batch_id}...[/cyan]")

    # 배치가 완료되거나 실패할 때까지 주기적으로 확인
    while True:
        batch = client.batches.retrieve(batch_id)

        console.print(f"[yellow]Status: {batch.status}[/yellow]")
        console.print(f"[yellow]Request counts: {batch.request_counts}[/yellow]")

        # 정상 완료
        if batch.status == "completed":
            console.print(f"[green]Batch completed![/green]")
            return batch
        # 실패, 만료, 취소 시 예외 상황
        elif batch.status in ["failed", "expired", "cancelled"]:
            console.print(f"[red]Batch {batch.status}![/red]")
            raise Exception(f"Batch {batch.status}: {batch}")

        # 진행 중인 경우 대기 후 재조회
        console.print(f"[cyan]Waiting {poll_interval} seconds...[/cyan]")
        time.sleep(poll_interval)


def download_results(client: OpenAI, batch: Dict[str, Any], output_file: str) -> str:
    """
    완료된 Batch 작업의 결과 파일을 다운로드하여 로컬에 저장.

    Batch 작업이 완료되면 OpenAI는 결과를 JSONL 파일 형태로 제공하며,
    이 함수는 해당 결과 파일을 다운로드하여 지정된 경로에 저장.

    Args:
        client: OpenAI API 클라이언트 객체
        batch: 완료된 배치 작업 객체
        output_file: 결과 파일을 저장할 경로

    Returns:
        저장된 결과 파일의 경로
    """
    console.print(f"[cyan]Downloading results...[/cyan]")

    # 에러 파일이 있으면 먼저 다운로드 및 확인
    if batch.error_file_id:
        console.print(f"[red]⚠️  Errors detected! Downloading error file...[/red]")
        error_file = output_file.replace(".jsonl", "_errors.jsonl")
        error_response = client.files.content(batch.error_file_id)

        with open(error_file, "wb") as f:
            f.write(error_response.content)

        console.print(f"[yellow]Error file saved to: {error_file}[/yellow]")

        # 첫 번째 에러 출력
        with open(error_file, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line:
                first_error = json.loads(first_line)
                console.print("[red]First error details:[/red]")
                console.print(json.dumps(first_error, indent=2, ensure_ascii=False))

    # output_file_id가 없으면 모든 요청이 실패한 것
    if not batch.output_file_id:
        error_msg = (
            f"All {batch.request_counts.failed} requests failed! "
            f"No successful results to download. "
        )
        if batch.error_file_id:
            error_msg += f"Check error file: {error_file}"
        else:
            error_msg += "No error file available."

        console.print(f"[red]{error_msg}[/red]")
        raise ValueError(error_msg)

    # 성공한 결과 파일 다운로드 및 저장
    file_response = client.files.content(batch.output_file_id)

    with open(output_file, "wb") as f:
        f.write(file_response.content)

    console.print(f"[green]Results saved to {output_file}[/green]")

    # 성공/실패 통계 출력
    console.print(f"[cyan]Summary:[/cyan]")
    console.print(f"  - Completed: {batch.request_counts.completed}")
    console.print(f"  - Failed: {batch.request_counts.failed}")
    console.print(f"  - Total: {batch.request_counts.total}")

    return output_file


def validate_and_parse_problems(results_file: str) -> List[Dict[str, Any]]:
    """
    Batch API 결과 파일(JSONL)을 검증하고 문제 데이터만 추출하여 파싱.

    각 줄(JSON 객체)마다 API 응답 상태를 확인하고,
    문제(JSON)가 요구되는 필수 필드와 형식을 만족하는지 검증한 뒤
    유효한 문제만 리스트로 반환.

    Args:
        results_file: Batch API 결과 JSONL 파일 경로

    Returns:
        검증을 통과한 문제 딕셔너리 리스트
    """
    console.print(f"[cyan]Validating and parsing results...[/cyan]")

    # 정상적으로 파싱된 문제
    problems = []
    # 오류 로그 모음
    errors = []

    # 결과 파일은 JSONL 형식 (한 줄 = 하나의 응답)
    with open(results_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                result = json.loads(line)

                # API 응답 상태 코드 확인
                if result.get("response", {}).get("status_code") != 200:
                    errors.append(f"Line {line_num}: API error")
                    continue

                # 응답 출력
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                problem = json.loads(content)

                # 필수 필드 검증
                required_fields = ["paragraph", "question", "choices", "answer"]
                if not all(field in problem for field in required_fields):
                    errors.append(f"Line {line_num}: Missing required fields")
                    continue

                # 선택지 개수 검증
                if not (4 <= len(problem["choices"]) <= 5):
                    errors.append(f"Line {line_num}: Invalid choices count")
                    continue

                # 정답 인덱스 검증 (1부터 4 또는 5까지)
                if not (1 <= problem["answer"] <= len(problem["choices"])):
                    errors.append(f"Line {line_num}: Answer out of range")
                    continue

                # Batch 요청 시 설정한 custom_id를 문제 ID로 사용
                problem["id"] = result.get("custom_id", f"unknown-{line_num}")

                problems.append(problem)

            except Exception as e:
                # JSON 파싱 오류 등 기타 예외 처리
                errors.append(f"Line {line_num}: {str(e)}")

    # 검증 결과 출력
    console.print(f"[green]Validated {len(problems)} problems[/green]")
    if errors:
        console.print(f"[red]Errors: {len(errors)}[/red]")
        for error in errors[:10]:  # Show first 10 errors
            console.print(f"[red]  - {error}[/red]")

    return problems


def save_to_csv(problems: List[Dict[str, Any]], output_csv: str):
    """
    검증된 문제 데이터를 학습용 train.csv 형식으로 저장합니다.

    문제(JSON)를 평탄화하여,
    지문(paragraph) + 문제 정보(problems) + question_plus 구조로 변환.

    Args:
        problems: 검증을 통과한 문제 딕셔너리 리스트
        output_csv: 저장할 CSV 파일 경로
    """
    console.print(f"[cyan]Saving to CSV...[/cyan]")

    rows = []
    for i, problem in enumerate(problems):
        # 증강된 문제 고유 ID 생성
        problem_id = f"augmented-gen-{i:04d}"

        # Problems 컬럼에 들어갈 JSON 구성
        problems_dict = {
            "question": problem["question"],
            "choices": problem["choices"],
            "answer": problem["answer"],
        }

        # csv의 row 구성
        row = {
            "id": problem_id,
            "paragraph": problem["paragraph"],
            "problems": json.dumps(problems_dict, ensure_ascii=False),
            "question_plus": problem.get("question_plus") or "",
        }
        rows.append(row)

    # DataFrame 변환 후 CSV 저장
    df = pd.DataFrame(rows)

    # 부모 디렉토리가 없으면 생성
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False, encoding="utf-8")

    console.print(f"[green]Saved {len(rows)} problems to {output_csv}[/green]")


def main(
    newspaper_path: str = "data/data4gen/newspaper",
    book_path: str = "data/data4gen/book",
    output_csv: str = "data/train_augmented.csv",
    problems_per_article: int = 1,
    seed: int = 42,
    data_type: str = "newspaper",
    num_newspaper_problems: int = None,
    num_book_problems: int = None,
    batch_size: int = 200,
):
    """
    데이터 기반 수능형 문제 데이터 증강 파이프라인의 메인 함수.

    신문 또는 도서 JSON 데이터를 불러와 OpenAI Batch API를 통해 문제를 대량 생성하고,
    생성 결과를 검증한 뒤 학습용 CSV 형식으로 저장합니다.

    Args:
        file_path: 데이터 JSON 파일들이 저장된 디렉터리 경로 (단일 타입 사용 시)
        output_csv: 최종 생성된 학습 데이터 CSV 저장 경로
        num_problems: 생성하고자 하는 전체 문제 수 목표값 (단일 타입 사용 시)
        problems_per_article: 기사/passage 1개당 생성할 문제 수
        seed: 무작위 샘플링 및 재현성을 위한 랜덤 시드 값
        data_type: 데이터 타입 ("newspaper", "book", 또는 "both")
        newspaper_path: 신문 데이터 경로 (both 모드에서만 사용)
        book_path: 도서 데이터 경로 (both 모드에서만 사용)
        num_newspaper_problems: 신문 데이터로 생성할 문제 수 (both 모드에서만 사용)
        num_book_problems: 도서 데이터로 생성할 문제 수 (both 모드에서만 사용)
    """
    console.print("[bold green]Starting Data Augmentation Pipeline[/bold green]")

    # openai 클라이언트 초기화
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY가 환경 변수에 없습니다.\n"
            ".env 파일을 생성하고 다음과 같이 설정하세요:\n"
            "OPENAI_API_KEY=your-api-key-here"
        )

    client = OpenAI(api_key=api_key)

    # Step 1: 데이터 로드 및 샘플링
    articles = []

    if data_type == "newspaper":
        console.print(f"[cyan]📰 신문 데이터 로딩 모드[/cyan]")
        all_articles = load_newspaper_data(newspaper_path)

        # 필요한 개수만큼 샘플링
        set_seed(seed)
        num_articles_needed = num_newspaper_problems // problems_per_article

        if len(all_articles) >= num_articles_needed:
            articles = random.sample(all_articles, num_articles_needed)
            console.print(
                f"[green]✓ 신문 기사 {len(articles)}개 샘플링 (전체 {len(all_articles)}개 중)[/green]"
            )
        else:
            articles = all_articles
            console.print(
                f"[yellow]⚠️  신문 기사 부족: {len(articles)}개 사용 → {len(articles) * problems_per_article}개 문제 생성 예정[/yellow]"
            )

        total_num_problems = len(articles) * problems_per_article

    elif data_type == "book":
        console.print(f"[cyan]📚 도서 데이터 로딩 모드[/cyan]")
        all_articles = load_book_data(book_path=book_path, seed=seed)

        # 필요한 개수만큼 샘플링
        set_seed(seed)
        num_articles_needed = num_book_problems // problems_per_article

        if len(all_articles) >= num_articles_needed:
            articles = random.sample(all_articles, num_articles_needed)
            console.print(
                f"[green]✓ 도서 passage {len(articles)}개 샘플링 (전체 {len(all_articles)}개 중)[/green]"
            )
        else:
            articles = all_articles
            console.print(
                f"[yellow]⚠️  도서 passage 부족: {len(articles)}개 사용 → {len(articles) * problems_per_article}개 문제 생성 예정[/yellow]"
            )

        total_num_problems = len(articles) * problems_per_article

    elif data_type == "both":
        console.print(f"[cyan]📰📚 신문+도서 혼합 데이터 로딩 모드[/cyan]")

        set_seed(seed)
        newspaper_articles = []
        book_articles = []

        # 신문 데이터 로드 및 샘플링
        if newspaper_path and num_newspaper_problems and num_newspaper_problems > 0:
            console.print(
                f"[cyan]📰 신문 데이터에서 {num_newspaper_problems}개 문제 생성 시작[/cyan]"
            )
            all_newspaper_articles = load_newspaper_data(newspaper_path)

            # 필요한 개수만큼 샘플링 (problems_per_article 고려)
            num_newspaper_articles_needed = (
                num_newspaper_problems // problems_per_article
            )
            if len(all_newspaper_articles) >= num_newspaper_articles_needed:
                newspaper_articles = random.sample(
                    all_newspaper_articles, num_newspaper_articles_needed
                )
                console.print(
                    f"[green]✓ 신문 기사 {len(newspaper_articles)}개 샘플링 (전체 {len(all_newspaper_articles)}개 중)[/green]"
                )
            else:
                newspaper_articles = all_newspaper_articles
                actual_problems = len(newspaper_articles) * problems_per_article
                console.print(
                    f"[yellow]⚠️  신문 기사 부족: {len(newspaper_articles)}개 사용 → {actual_problems}개 문제 생성 예정[/yellow]"
                )
        else:
            num_newspaper_problems = 0

        # 도서 데이터 로드 및 샘플링
        if book_path and num_book_problems and num_book_problems > 0:
            console.print(
                f"[cyan]📚 도서 데이터에서 {num_book_problems}개 문제 생성 시작[/cyan]"
            )
            all_book_articles = load_book_data(book_path=book_path, seed=seed)

            # 필요한 개수만큼 샘플링 (problems_per_article 고려)
            num_book_articles_needed = num_book_problems // problems_per_article
            if len(all_book_articles) >= num_book_articles_needed:
                book_articles = random.sample(
                    all_book_articles, num_book_articles_needed
                )
                console.print(
                    f"[green]✓ 도서 passage {len(book_articles)}개 샘플링 (전체 {len(all_book_articles)}개 중)[/green]"
                )
            else:
                book_articles = all_book_articles
                actual_problems = len(book_articles) * problems_per_article
                console.print(
                    f"[yellow]⚠️  도서 passage 부족: {len(book_articles)}개 사용 → {actual_problems}개 문제 생성 예정[/yellow]"
                )
        else:
            num_book_problems = 0

        if not newspaper_articles and not book_articles:
            raise ValueError(
                "both 모드에서는 num_newspaper_problems 또는 num_book_problems 중 "
                "최소 하나는 0보다 커야 합니다."
            )

        # 신문 + 도서 데이터 결합
        articles = newspaper_articles + book_articles
        total_num_problems = (
            len(newspaper_articles) * problems_per_article
            + len(book_articles) * problems_per_article
        )

        console.print(f"[green]총 {len(articles)}개 데이터 준비 완료[/green]")
        console.print(
            f"[cyan]생성 예정: 신문 {len(newspaper_articles) * problems_per_article}개 + "
            f"도서 {len(book_articles) * problems_per_article}개 = "
            f"총 {total_num_problems}개 문제[/cyan]"
        )

    else:
        raise ValueError(
            f"Unknown data_type: {data_type}. Use 'newspaper', 'book', or 'both'."
        )

    # Step 2-6: 배치를 나눠서 순차 처리
    # 배치 크기: 한 번에 처리할 기사 수 (요청 수 = batch_size * problems_per_article)
    all_problems = []
    total_articles = len(articles)
    total_batches = (total_articles + batch_size - 1) // batch_size

    console.print(
        f"[cyan]총 {total_articles}개 기사를 {total_batches}개 배치로 나눠 처리합니다[/cyan]"
    )
    console.print(
        f"[yellow]배치 크기: {batch_size}개 기사 (예상 요청 수: {batch_size * problems_per_article}개)[/yellow]"
    )

    for batch_idx in range(0, total_articles, batch_size):
        batch_num = batch_idx // batch_size + 1
        batch_articles = articles[batch_idx : batch_idx + batch_size]

        console.print(
            f"[bold cyan]배치 {batch_num}/{total_batches} 처리 중... ({len(batch_articles)}개 기사)[/bold cyan]"
        )

        # Step 2: 배치 요청 파일 생성
        batch_file = f"batch_request_{batch_idx}.jsonl"
        create_batch_request_file(
            batch_articles,
            batch_file,
            problems_per_article=problems_per_article,
            seed=seed,
        )

        # Step 3: 배치 제출
        batch_id = submit_batch(client, batch_file)

        # Step 4: 배치 모니터링
        completed_batch = monitor_batch(client, batch_id)

        # Step 5: 결과 다운로드
        results_file = f"batch_results_{batch_idx}.jsonl"
        download_results(client, completed_batch, results_file)

        # Step 6: 결과 파싱
        problems = validate_and_parse_problems(results_file)
        all_problems.extend(problems)

        console.print(
            f"[green]✓ 배치 {batch_num} 완료: {len(problems)}개 문제 생성 (누적: {len(all_problems)}개)[/green]"
        )

    # Step 7: 모든 문제를 한 번에 저장
    console.print(f"[cyan]총 {len(all_problems)}개 문제를 저장합니다...[/cyan]")
    save_to_csv(all_problems, output_csv)

    console.print("[bold green]Data Augmentation Pipeline Completed![/bold green]")
    console.print(f"[green]Generated {len(all_problems)} problems[/green]")
    console.print(f"[green]Output saved to {output_csv}[/green]")


if __name__ == "__main__":

    # 커맨드라인 인자 파서 설정
    parser = argparse.ArgumentParser(
        description="신문 기사 기반 수능형 문제 데이터를 생성."
    )

    parser.add_argument(
        "--newspaper_path",
        type=str,
        default=None,
        help="신문 데이터 경로 (both 모드에서만 사용)",
    )
    parser.add_argument(
        "--book_path",
        type=str,
        default=None,
        help="도서 데이터 경로 (both 모드에서만 사용)",
    )
    parser.add_argument(
        "--output_csv",
        default="data/train_newspaper_augmented.csv",
        help="생성된 학습 데이터 CSV 저장 경로",
    )
    parser.add_argument(
        "--problems_per_article",
        type=int,
        default=1,
        help="기사 1개당 생성할 문제 수",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성을 위한 랜덤 시드 값",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="newspaper",
        choices=["newspaper", "book", "both"],
        help="데이터 타입: 'newspaper', 'book', 또는 'both'",
    )
    parser.add_argument(
        "--num_newspaper_problems",
        type=int,
        default=None,
        help="신문 데이터로 생성할 문제 수 (both 모드에서만 사용)",
    )
    parser.add_argument(
        "--num_book_problems",
        type=int,
        default=None,
        help="도서 데이터로 생성할 문제 수 (both 모드에서만 사용)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="한 번에 처리할 기사 수 (기본값: 200, 토큰 제한 회피를 위해 조정 가능)",
    )

    args = parser.parse_args()

    main(
        file_path=args.file_path,
        output_csv=args.output_csv,
        problems_per_article=args.problems_per_article,
        seed=args.seed,
        data_type=args.data_type,
        newspaper_path=args.newspaper_path,
        book_path=args.book_path,
        num_newspaper_problems=args.num_newspaper_problems,
        num_book_problems=args.num_book_problems,
        batch_size=args.batch_size,
    )
