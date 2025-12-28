import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.progress import track

console = Console()

MIN_ARTICLE_LENGTH = 200  # 너무 짧은 기사 제외 기준
MIN_PARAGRAPH_LENGTH = 20  # 제목, 캡션 등으로 추정되는 짧은 문단 제거 기준

CATEGORY_DISTRIBUTION = {
    # 주요 카테고리: 300개씩 (총 1800개)
    "문어 > 책-정보 > 사회과학": 300,
    "문어 > 책-정보 > 철학": 300,
    "문어 > 책-정보 > 기술과학": 300,
    "문어 > 책-정보 > 역사": 300,
    "문어 > 책-정보 > 예술": 300,
    "문어 > 책-정보 > 자연과학": 300,
    # 문학: 20개씩 (총 40개)
    "문어 > 책-상상 > 문학": 20,
    "문어 > 책-정보 > 문학": 20,
    # 기타: 40개씩 (총 160개)
    "문어 > 책-정보 > 종교": 40,
    "문어 > 책-정보 > 총류": 40,
    "문어 > 책-정보 > 언어": 40,
    "문어 > 잡지": 40,
}


def load_newspaper_data(newspaper_path: str) -> List[Dict[str, Any]]:
    """
    신문 JSON 파일을 로드하고 파싱합니다.

    Args:
        newspaper_path: 신문 JSON 파일이 저장된 파일 경로

    Returns:
        메타데이터를 포함한 기사 목록
    """

    # 로딩 시작
    console.print(f"[cyan]Loading newspaper data from {newspaper_path}...[/cyan]")

    articles = []

    # 해당 경로의 모든 JSON 파일 찾기
    json_files = list(Path(newspaper_path).glob("*.json"))

    # 각 JSON 파일 파싱
    for json_file in track(json_files, description="Parsing JSON files"):
        data = _load_json_file(json_file)
        if data is None:
            continue

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
            if len(content) < MIN_ARTICLE_LENGTH:
                continue

            # 기사 단위로 데이터 재구성
            article = {
                "article_id": doc.get("id"),
                "title": doc.get("metadata", {}).get("title", ""),
                "content": content,
            }

            articles.append(article)

    console.print(f"[green]Loaded {len(articles)} articles[/green]")
    return articles


def load_book_data(
    book_path: str,
    min_paragraphs: int = 5,
    max_paragraphs: int = 20,
    min_length: int = 500,
    max_length: int = 2000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    도서(JSON) 데이터를 카테고리 분포에 맞추어 샘플링한 뒤,
    여러 문단(paragraph)을 결합하여 passage 단위 데이터로 변환한다.

    처리 흐름:
    1. 전체 JSON 파일을 카테고리별로 분류
    2. 사전 정의된 카테고리 분포에 따라 파일을 샘플링
    3. 선택된 파일들에 대해 문단을 순차적으로 결합하여 passage 생성

    Args:
        book_path (str):
            도서 JSON 파일들이 저장된 디렉토리 경로
        min_paragraphs (int):
            하나의 passage를 구성하기 위한 최소 문단 수
        max_paragraphs (int):
            하나의 passage에 포함될 수 있는 최대 문단 수
        min_length (int):
            passage 전체 텍스트의 최소 길이 (문자 수 기준)
        max_length (int):
            passage 전체 텍스트의 최대 길이 (문자 수 기준)
        seed (int):
            카테고리별 파일 샘플링 시 사용할 랜덤 시드

    Returns:
        List[Dict[str, Any]]:
            passage 리스트.
            각 passage는 article_id, title, content를 포함한다.
    """

    console.print(f"[cyan]도서 데이터 로딩 중: {book_path}[/cyan]")

    # JSON 파일 경로 수집
    written_path = Path(book_path)
    json_files = list(written_path.glob("*.json"))

    # 1. 전체 JSON 파일을 카테고리별로 분류
    category_files = defaultdict(list)

    for json_file in track(json_files, description="카테고리 분류 중"):
        data = _load_json_file(json_file)
        if data is None:
            continue

        # 파일 단위 메타데이터에서 카테고리 추출
        category = data.get("metadata", {}).get("category", "Unknown")
        category_files[category].append(json_file)

    # 2. 카테고리 분포에 따라 파일 샘플링
    random.seed(seed)
    selected_files = []
    shortage_info = []

    for category, count in CATEGORY_DISTRIBUTION.items():
        available = category_files.get(category, [])

        # 해당 카테고리에서 필요한 수만큼 무작위 선택
        selected = random.sample(available, min(count, len(available)))
        selected_files.extend(selected)

        if len(selected) < count:
            shortage = count - len(selected)
            console.print(
                f"[yellow]⚠️  {category}: {len(selected)}개 선택 (요청: {count}개, 부족: {shortage}개)[/yellow]"
            )
            shortage_info.append(
                {
                    "category": category,
                    "requested": count,
                    "selected": len(selected),
                    "shortage": shortage,
                }
            )
        else:
            console.print(f"[green]✓ {category}: {len(selected)}개 선택[/green]")

    console.print(f"[cyan]최종 선택: {len(selected_files)}개 파일[/cyan]")

    # 3. 부족분 요약 출력
    _print_shortage_summary(shortage_info)

    # 4. 선택된 파일들로부터 passage 생성
    passages = []
    console.print(f"[yellow]Passage 생성 시작[/yellow]")

    for json_file in track(selected_files, description="JSON 파일 파싱 중"):
        data = _load_json_file(json_file)
        if data is None:
            continue

        # 하나의 JSON 파일에는 여러 document (책)가 존재할 수 있음
        for doc in data.get("document", []):
            # 문단 리스트 추출
            paragraphs = doc.get("paragraph", [])

            # 문단 텍스트 정제
            paragraph_texts = []
            for p in paragraphs:
                text = p.get("form", "").strip()

                # 제목, 캡션 등으로 추정되는 짧은 문단 제거
                if len(text) > MIN_PARAGRAPH_LENGTH:
                    paragraph_texts.append(text)

            # 문단 수가 너무 적은 문서는 제외
            if len(paragraph_texts) < min_paragraphs:
                continue

            # 문단을 순차적으로 결합하여 passage 생성
            current_passage = []  # 현재 누적 중인 문단들
            current_length = 0  # 누적 텍스트 길이

            for para_text in paragraph_texts:
                para_length = len(para_text)

                # 다음 문단을 추가하면 최대 길이를 초과하는 경우
                if (
                    current_length + para_length > max_length
                    and len(current_passage) >= min_paragraphs
                ):
                    combined_text = "\n".join(current_passage)

                    # 길이 조건을 만족하는 경우만 passage로 저장
                    if min_length <= len(combined_text) <= max_length:
                        passages.append(
                            _create_passage(doc, combined_text, len(passages))
                        )

                    # 새로운 passage 시작
                    current_passage = [para_text]
                    current_length = para_length

                else:
                    # 아직 여유가 있으면 문단 추가
                    current_passage.append(para_text)
                    current_length += para_length

                    # 문단 수가 최대치에 도달한 경우 즉시 passage 생성
                    if len(current_passage) >= max_paragraphs:
                        combined_text = "\n".join(current_passage)

                        if min_length <= len(combined_text) <= max_length:
                            passages.append(
                                _create_passage(doc, combined_text, len(passages))
                            )
                        current_passage = []
                        current_length = 0

            # 루프 종료 후 남아 있는 문단 처리
            if len(current_passage) >= min_paragraphs:
                combined_text = "\n".join(current_passage)

                if min_length <= len(combined_text) <= max_length:
                    passages.append(_create_passage(doc, combined_text, len(passages)))

    console.print(f"[green]총 {len(passages)}개의 passage 로드 완료[/green]")
    return passages


def _load_json_file(json_file: Path) -> Optional[Dict[str, Any]]:
    """
    JSON 파일을 로드합니다.

    Args:
        json_file: JSON 파일 경로

    Returns:
        파싱된 JSON 데이터. 실패 시 None
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[red]{json_file} 오류: {e}[/red]")
        return None


def _print_shortage_summary(shortage_info: List[Dict[str, Any]]) -> None:
    """카테고리별 부족분 요약을 출력합니다."""
    if not shortage_info:
        return

    total_shortage = sum(item["shortage"] for item in shortage_info)
    console.print(f"\n[yellow]📊 부족분 요약 (총 {total_shortage}개 부족)[/yellow]")

    for item in shortage_info:
        console.print(
            f"[yellow]  - {item['category']}: "
            f"{item.get('selected', 0)}/{item['requested']} "
            f"(부족: {item['shortage']}개)[/yellow]"
        )
    console.print()


def _create_passage(
    doc: Dict[str, Any],
    combined_text: str,
    passage_count: int,
) -> Dict[str, Any]:
    """
    passage 딕셔너리를 생성.

    Args:
        doc: 문서 딕셔너리
        combined_text: 결합된 문단 텍스트
        passage_count: 현재까지 생성된 passage 수

    Returns:
        passage 딕셔너리
    """
    return {
        "article_id": f"{doc.get('id', 'unknown')}-{passage_count}",
        "title": doc.get("metadata", {}).get("title", ""),
        "content": combined_text,
    }
