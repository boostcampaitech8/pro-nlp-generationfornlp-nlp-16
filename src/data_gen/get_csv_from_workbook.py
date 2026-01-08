import json
import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html import unescape
from bs4 import BeautifulSoup

# 동그라미 숫자 매핑 (①~⑤ → 1~5)
CIRCLE_NUM_MAP = {
    "①": 1,
    "②": 2,
    "③": 3,
    "④": 4,
    "⑤": 5,
    "⑴": 1,
    "⑵": 2,
    "⑶": 3,
    "⑷": 4,
    "⑸": 5,
    "⓵": 1,
    "⓶": 2,
    "⓷": 3,
    "⓸": 4,
    "⓹": 5,
}


def remove_html_tags(text: str) -> str:
    """
    HTML 태그를 제거하되 내용은 보존하는 함수

    Args:
        text: HTML 태그가 포함된 텍스트

    Returns:
        HTML 태그가 제거되고 정리된 순수 텍스트
        - <br> 태그는 줄바꿈으로 변환
        - HTML 엔티티(&nbsp; 등)는 디코딩
        - 연속된 공백과 줄바꿈 정리
    """
    if not text or pd.isna(text):
        return text

    # BeautifulSoup을 사용하여 HTML 태그 제거 (내용은 보존)
    soup = BeautifulSoup(str(text), "html.parser")

    # <br> 태그를 줄바꿈으로 변환
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # 모든 태그 제거하고 텍스트만 추출
    text = soup.get_text()

    # HTML 엔티티 디코딩 (예: &nbsp; -> 공백)
    text = unescape(text)

    # 연속된 공백을 하나로 줄이기 (단, 줄바꿈은 보존)
    text = re.sub(r" +", " ", text)

    # 연속된 줄바꿈을 최대 2개로 제한
    text = re.sub(r"\n\n+", "\n\n", text)

    return text.strip()


def extract_choice_number(text: str) -> Optional[int]:
    """
    텍스트에서 동그라미 숫자 또는 일반 숫자를 찾아서 1-5 숫자로 변환

    Args:
        text: 선택지 텍스트 (예: "① 선택지 내용" 또는 "(1) 선택지 내용")

    Returns:
        1-5 사이의 숫자 또는 None (숫자를 찾지 못한 경우)
    """
    # 동그라미 숫자 먼저 확인
    for circle_num, num in CIRCLE_NUM_MAP.items():
        if circle_num in text:
            return num

    # 일반 숫자 패턴 확인: (1), (2), (3), (4), (5) 또는 1), 2), 3), 4), 5)
    pattern = r"[\(（]?([1-5])[\)）]"
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))

    return None


def extract_choice_text(text: str) -> str:
    """
    선택지 텍스트에서 동그라미 숫자/일반 숫자와 앞의 불필요한 부분 제거

    Args:
        text: 선택지 텍스트 (예: "① 선택지 내용")

    Returns:
        숫자가 제거된 순수 선택지 텍스트 (예: "선택지 내용")
    """
    # 동그라미 숫자로 시작하는 패턴 찾기
    for circle_num in CIRCLE_NUM_MAP.keys():
        if circle_num in text:
            # 동그라미 숫자 이후의 텍스트 추출
            idx = text.find(circle_num)
            # 동그라미 숫자와 그 뒤의 공백 제거
            result = text[idx + len(circle_num) :].strip()
            # 줄바꿈이 있으면 첫 줄만 가져오기 (일부 선택지가 여러 줄일 수 있음)
            if "\n" in result:
                result = result.split("\n")[0].strip()
            return result

    # 일반 숫자 패턴 처리: (1) 텍스트 또는 1) 텍스트
    pattern = r"[\(（]?[1-5][\)）]\s*(.+)"
    match = re.search(pattern, text)
    if match:
        result = match.group(1).strip()
        # 줄바꿈이 있으면 첫 줄만 가져오기
        if "\n" in result:
            result = result.split("\n")[0].strip()
        return result

    return text.strip()


def parse_choices(
    correct_text: str, incorrect_texts: List[str]
) -> Tuple[List[str], int]:
    """
    정답 텍스트와 오답 텍스트들을 파싱하여 선택지 리스트와 정답 인덱스 반환

    Args:
        correct_text: class_num:3의 text_description
        incorrect_texts: class_num:4의 모든 text_description 리스트

    Returns:
        (choices_list, answer_index): 선택지 리스트와 정답 인덱스 (1-5)
    """
    choices_dict = {}

    # 정답에서 선택지 추출
    correct_num = extract_choice_number(correct_text)
    if correct_num:
        correct_choice = extract_choice_text(correct_text)
        choices_dict[correct_num] = correct_choice

    # 오답들에서 선택지 추출
    for incorrect_text in incorrect_texts:
        # 여러 줄로 나뉘어 있을 수 있으므로 줄별로 처리
        lines = incorrect_text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            choice_num = extract_choice_number(line)
            if choice_num:
                choice_text = extract_choice_text(line)
                choices_dict[choice_num] = choice_text

    # 1-5 순서대로 정렬
    choices = []
    answer = None
    for i in range(1, 6):
        if i in choices_dict:
            choices.append(choices_dict[i])
            if i == correct_num:
                answer = len(choices)  # 1-based index

    return choices, answer


def process_json_file(json_path: Path) -> Optional[Dict]:
    """
    JSON 파일을 처리하여 하나의 row 데이터 반환

    Returns:
        {
            'id': str,
            'paragraph': str,
            'problems': dict (question, choices, answer),
            'description': str
        } 또는 None (파싱 실패 시)
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

    learning_data = data.get("learning_data_info", [])

    # 각 class_num별로 데이터 추출
    question = None
    paragraph = None
    correct_text = None
    incorrect_texts = []
    description = None

    for item in learning_data:
        class_num = item.get("class_num")
        class_info_list = item.get("class_info_list", [])

        if class_num == 1:  # 문항
            # 문항이 여러 개의 Bounding Box로 나뉘어 있을 수 있으므로 모두 합치기
            question_parts = []
            for class_info in class_info_list:
                question_text = class_info.get("text_description", "").strip()
                if question_text:
                    question_parts.append(remove_html_tags(question_text))
            if question_parts:
                question = " ".join(question_parts)
        elif class_num == 2:  # 지문
            # 지문이 여러 개의 Bounding Box로 나뉘어 있을 수 있으므로 모두 합치기
            paragraph_parts = []
            for class_info in class_info_list:
                paragraph_text = class_info.get("text_description", "").strip()
                if paragraph_text:
                    paragraph_parts.append(remove_html_tags(paragraph_text))
            if paragraph_parts:
                paragraph = "\n".join(paragraph_parts)
        elif class_num == 3:  # 정답
            # 정답이 여러 개의 Bounding Box로 나뉘어 있을 수 있으므로 모두 합치기
            correct_parts = []
            for class_info in class_info_list:
                correct_part = class_info.get("text_description", "").strip()
                if correct_part:
                    correct_parts.append(remove_html_tags(correct_part))
            if correct_parts:
                correct_text = " ".join(correct_parts)
        elif class_num == 4:  # 오답
            # 오답이 여러 개의 Bounding Box로 나뉘어 있을 수 있으므로 모두 합치기
            incorrect_parts = []
            for class_info in class_info_list:
                incorrect_text = class_info.get("text_description", "").strip()
                if incorrect_text:
                    incorrect_parts.append(remove_html_tags(incorrect_text))
            # 모든 오답 텍스트를 합쳐서 하나의 문자열로 만들기
            if incorrect_parts:
                # 줄바꿈으로 구분하여 합치기 (parse_choices에서 줄바꿈으로 split함)
                incorrect_texts = ["\n".join(incorrect_parts)]
        elif class_num == 5:  # 해설
            # 해설은 여러 개일 수 있으므로 모두 합치기
            description_parts = []
            for class_info in class_info_list:
                desc_text = class_info.get("text_description", "").strip()
                if desc_text:
                    description_parts.append(remove_html_tags(desc_text))
            if description_parts:
                description = "\n".join(description_parts)

    # 필수 데이터 확인
    if not question or not paragraph or not correct_text:
        print(f"Missing required data in {json_path}")
        return None

    # 선택지 파싱
    choices, answer = parse_choices(correct_text, incorrect_texts)

    if not choices or answer is None:
        print(f"Failed to parse choices in {json_path}")
        return None

    # ID 생성 (파일명 기반)
    file_stem = json_path.stem
    row_id = f"generation-for-nlp-{file_stem}"

    # problems 딕셔너리 생성
    problems = {"question": question, "choices": choices, "answer": answer}

    return {
        "id": row_id,
        "paragraph": paragraph,
        "problems": problems,
        "description": description if description else "",
    }


def process_workbook_directory(workbook_dir: Path) -> List[Dict]:
    """
    workbook 디렉토리의 모든 하위 디렉토리에서 JSON 파일들을 처리

    Returns:
        모든 문제의 row 데이터 리스트
    """
    all_rows = []

    # workbook 디렉토리의 모든 하위 디렉토리 탐색
    for subdir in workbook_dir.iterdir():
        if not subdir.is_dir():
            continue

        print(f"Processing directory: {subdir.name}")

        # 하위 디렉토리의 모든 JSON 파일 처리
        json_files = list(subdir.glob("*.json"))
        print(f"  Found {len(json_files)} JSON files")

        for json_file in json_files:
            row_data = process_json_file(json_file)
            if row_data:
                all_rows.append(row_data)

        print(
            f"  Processed {len([r for r in all_rows if r['id'].endswith(subdir.name.split('/')[-1])])} valid rows"
        )

    return all_rows


def convert_to_csv_format(rows: List[Dict]) -> pd.DataFrame:
    """
    row 데이터를 CSV 형식으로 변환

    Args:
        rows: 문제 데이터 딕셔너리 리스트

    Returns:
        CSV 형식의 DataFrame (problems 딕셔너리는 문자열로 변환됨)
    """
    csv_rows = []

    for row in rows:
        # problems 딕셔너리를 문자열로 변환
        problems_str = str(row["problems"]).replace("'", '"')

        csv_row = {
            "id": row["id"],
            "paragraph": row["paragraph"],
            "problems": problems_str,
            "description": row.get("description", ""),
        }
        csv_rows.append(csv_row)

    return pd.DataFrame(csv_rows)


def main():
    """
    메인 실행 함수: workbook 디렉토리의 JSON 파일들을 처리하여 CSV로 변환
    HTML 태그를 제거하고 문제, 지문, 선택지, 정답, 해설을 추출하여 저장
    """
    # 경로 설정
    workbook_dir = Path("data/data4gen/workbook")
    output_file = Path("data/aihub_workbook.csv")

    if not workbook_dir.exists():
        print(f"Error: {workbook_dir} does not exist")
        return

    print(f"Processing workbook directory: {workbook_dir}")

    # 모든 JSON 파일 처리
    all_rows = process_workbook_directory(workbook_dir)

    print(f"\nTotal rows processed: {len(all_rows)}")

    if not all_rows:
        print("No valid rows found. Exiting.")
        return

    # CSV 형식으로 변환
    df = convert_to_csv_format(all_rows)

    # CSV 저장
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nSaved to {output_file}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
