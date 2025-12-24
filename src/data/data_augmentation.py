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

def main(
    file_path: str = "data/data4gen/newspaper",
):
    # Step 1: 신문 기사 데이터 로드
    articles = load_newspaper_data(file_path)

    # Step 2: Batch API 요청 파일 생성

    # Step 3: 생성된 JSONL 파일을 OpenAI Batch API에 제출

    # Step 4: Batch 작업 상태 모니터링

    # Step 5: 완료된 배치의 결과(JSONL)를 로컬 파일로 저장

    # Step 6: 모델 출력 JSON을 검증하고 형식상 유효한 문제만 추출

    # Step 7: 검증된 문제를 train.csv 호환 형식으로 저장


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

    args = parser.parse_args()

    main(
        file_path=args.file_path,
    )
