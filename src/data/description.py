import json
import os
from typing import Dict, List
import pandas as pd
from rich.console import Console

console = Console()


def load_descriptions_json(json_path: str) -> Dict[str, Dict]:
    """
    descriptions.json 파일을 로드하여 id를 키로 하는 딕셔너리로 변환
    
    Args:
        json_path: descriptions.json 파일 경로
        
    Returns:
        dict: {sample_id: description_data} 형태의 딕셔너리
        
    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
    """
    console.print(f"[green]✓[/green] Description 파일을 찾았습니다: [italic]{json_path}[/italic]")
    with open(json_path, "r", encoding="utf-8") as f:
        descriptions_list = json.load(f)
    
    # id를 키로 하는 딕셔너리로 변환
    descriptions_dict = {}
    for item in descriptions_list:
        sample_id = str(item["id"])
        descriptions_dict[sample_id] = item
    
    console.print(f"[green]✓[/green] {len(descriptions_dict)}개의 description을 로드했습니다.")
    return descriptions_dict


def save_descriptions(results: List[dict], csv_path: str = "descriptions.csv", json_path: str = "descriptions.json", data_dir: str = None):
    """
    생성된 description 결과를 CSV와 JSON 파일로 저장한다.

    Args:
        results:
            [{"id": ..., "description_1": ..., "description_2": ...}, ...] 형태의 리스트
        csv_path:
            CSV 파일 저장 경로
        json_path:
            JSON 파일 저장 경로
        data_dir:
            data/ 폴더 경로 (None이면 현재 작업 디렉토리 기준 "data" 사용)
    """
    # CSV 저장
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    console.print(f"[green]✓[/green] 총 {len(results)}개의 description을 [italic]{csv_path}[/italic]에 저장했습니다.")

    # JSON 저장
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    console.print(f"[green]✓[/green] 총 {len(results)}개의 description을 [italic]{json_path}[/italic]에 저장했습니다.")
    
    # data/ 폴더에도 복사
    if data_dir is None:
        data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    data_json_path = os.path.join(data_dir, "descriptions.json")
    with open(data_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    console.print(f"[green]✓[/green] 총 {len(results)}개의 description을 [italic]{data_json_path}[/italic]에도 저장했습니다.")

