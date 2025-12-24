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
from src.utils.datagen_prompt import SYSTEM_PROMPT, create_prompt_for_article

load_dotenv()

console = Console()

def load_newspaper_data(file_path: str) -> List[Dict[str, Any]]:
    """
    신문 JSON 파일을 로드하고 파싱합니다.

    Args:
        file_path: 신문 JSON 파일이 저장된 파일 경로

    Returns:
        메타데이터를 포함한 기사 목록
    """
    
    # 로딩 시작
    console.print(f"[cyan]Loading newspaper data from {file_path}...[/cyan]")

    articles = []

    # 해당 경로의 모든 JSON 파일 찾기
    json_files = list(Path(file_path).glob("*.json"))

    # 각 JSON 파일 파싱
    for json_file in track(json_files, description="Parsing JSON files"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # JSON 내부의 document에서 기사 데이터 추출
            for doc in data.get("document", []):
                # 흩어져있는 paragraph들을 합쳐서 하나의 content로 만들기
                paragraphs = doc.get("paragraph", [])
                content = "\n".join(
                    [
                        p.get("form", "").replace("<p>", "").replace("</p>", "").strip()
                        for p in paragraphs
                    ]
                )

                # 너무 짧은 본문은 제외
                if len(content) < 200:
                    continue

                # 기사 단위로 데이터 재구성
                article = {
                    "article_id": doc.get("id"),
                    "title": doc.get("metadata", {}).get("title", ""),
                    "content": content,
                    "date": doc.get("metadata", {}).get("date", ""),
                    "topic": doc.get("metadata", {}).get("topic", ""),
                }

                articles.append(article)

        except Exception as e:
            console.print(f"[red]Error parsing {json_file}: {e}[/red]")
            continue

    console.print(f"[green]Loaded {len(articles)} articles[/green]")
    return articles

def create_batch_request_file(
    articles: List[Dict[str, Any]],
    output_file: str,
    num_problems: int = 2000,
    problems_per_article: int = 1,
    seed: int = 42,
) -> str:
    """
    OpenAI Batch API에 제출할 요청들을 JSONL 파일 형태로 생성.

    각 신문 기사(article)를 기반으로 하나 이상의 문제 생성 요청을 만들고,
    이를 OpenAI Batch API 규격에 맞는 JSONL 파일로 저장.

    Args:
        articles: 신문 기사 딕셔너리 리스트
        output_file: 생성된 JSONL 파일을 저장할 경로
        num_problems: 전체 생성하고자 하는 문제 수 목표값
        problems_per_article: 기사 1개당 생성할 문제 수
        seed: 무작위 샘플링 재현성을 위한 시드 값

    Returns:
        생성된 JSONL 배치 요청 파일의 경로
    """
    console.print(f"[cyan]Creating batch request file...[/cyan]")

    # 재현성위 위한 시드 설정
    set_seed(seed)
    console.print(f"[green]Random seed set to: {seed}[/green]")

    # 목표 문제 수에 맞게 필요한 기사 수 계산
    # 기사 수가 부족할 경우를 대비해서 min 사용
    num_articles_needed = min(len(articles), num_problems // problems_per_article)

    # 전체 기사 수가 충분하면 랜덤 샘플링
    if len(articles) > num_articles_needed:
        selected_articles = random.sample(articles, num_articles_needed)
        console.print(
            f"[green]Randomly sampled {num_articles_needed} articles from {len(articles)} total[/green]"
        )
    else:
        selected_articles = articles
        console.print(f"[yellow]Using all {len(articles)} articles[/yellow]")

    # Batch API 요청 객체 생성
    requests = []
    for article in track(selected_articles, description="Creating requests"):
        for problem_idx in range(problems_per_article):
            # custom_id는 이후 결과 매핑을 위한 고유 식별자
            custom_id = f"{article['article_id']}-q{problem_idx + 1}"

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": create_prompt_for_article(article)},
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

    # 배치 결과 파일 다운로드 및 저장
    file_response = client.files.content(batch.output_file_id)

    with open(output_file, "wb") as f:
        f.write(file_response.content)

    console.print(f"[green]Results saved to {output_file}[/green]")
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
    df.to_csv(output_csv, index=False, encoding="utf-8")

    console.print(f"[green]Saved {len(rows)} problems to {output_csv}[/green]")


def main(
    file_path: str = "data/data4gen/newspaper",
    output_csv: str = "data/train_newspaper_augmented.csv",
    num_problems: int = 2000,
    problems_per_article: int = 1,
    seed: int = 42,
):
    """
    신문 기사 기반 수능형 문제 데이터 증강 파이프라인의 메인 함수.

    신문 JSON 데이터를 불러와 OpenAI Batch API를 통해 문제를 대량 생성하고,
    생성 결과를 검증한 뒤 학습용 CSV 형식으로 저장합니다.

    Args:
        file_path: 신문 기사 JSON 파일들이 저장된 디렉터리 경로
        output_csv: 최종 생성된 학습 데이터 CSV 저장 경로
        num_problems: 생성하고자 하는 전체 문제 수 목표값
        problems_per_article: 기사 1개당 생성할 문제 수
        seed: 무작위 샘플링 및 재현성을 위한 랜덤 시드 값
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

    # Step 1: 신문 기사 데이터 로드
    articles = load_newspaper_data(file_path)

    # Step 2: Batch API 요청 파일 생성
    batch_file = "batch_request.jsonl"
    create_batch_request_file(
        articles,
        batch_file,
        num_problems=num_problems,
        problems_per_article=problems_per_article,
        seed=seed,
    )

    # Step 3: 생성된 JSONL 파일을 OpenAI Batch API에 제출
    batch_id = submit_batch(client, batch_file)

    # Step 4: Batch 작업 상태 모니터링
    completed_batch = monitor_batch(client, batch_id)

    # Step 5: 완료된 배치의 결과(JSONL)를 로컬 파일로 저장
    results_file = "batch_results.jsonl"
    download_results(client, completed_batch, results_file)

    # Step 6: 모델 출력 JSON을 검증하고 형식상 유효한 문제만 추출
    problems = validate_and_parse_problems(results_file)

    # Step 7: 검증된 문제를 train.csv 호환 형식으로 저장
    save_to_csv(problems, output_csv)

    console.print("[bold green]Data Augmentation Pipeline Completed![/bold green]")
    console.print(f"[green]Generated {len(problems)} problems[/green]")
    console.print(f"[green]Output saved to {output_csv}[/green]")


if __name__ == "__main__":

    # 커맨드라인 인자 파서 설정
    parser = argparse.ArgumentParser(
        description="신문 기사 기반 수능형 문제 데이터를 생성."
    )

    parser.add_argument(
        "--file_path",
        default="data/data4gen/newspaper",
        help="신문 기사 JSON 파일들이 저장된 디렉터리 경로",
    )
    parser.add_argument(
        "--output_csv",
        default="data/train_newspaper_augmented.csv",
        help="생성된 학습 데이터 CSV 저장 경로",
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=2000,
        help="생성할 전체 문제 수",
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

    args = parser.parse_args()

    main(
        file_path=args.file_path,
        output_csv=args.output_csv,
        num_problems=args.num_problems,
        problems_per_article=args.problems_per_article,
        seed=args.seed,
    )
