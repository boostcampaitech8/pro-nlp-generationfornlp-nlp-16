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
