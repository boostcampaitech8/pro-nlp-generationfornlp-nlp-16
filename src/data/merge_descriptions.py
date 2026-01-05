import json
import pandas as pd
from typing import Dict, List, Optional


def load_descriptions_json(json_path: str) -> Dict[str, Dict]:
    """
    descriptions.json 파일을 로드하여 id를 키로 하는 딕셔너리로 변환

    Args:
        json_path: descriptions.json 파일 경로

    Returns:
        {id: {"description_1": ..., "description_2": ...}, ...} 형태의 딕셔너리
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        descriptions_list = json.load(f)

    # id를 키로 하는 딕셔너리로 변환
    descriptions_dict = {}
    for item in descriptions_list:
        sample_id = str(item['id'])  # id를 문자열로 변환
        descriptions_dict[sample_id] = {
            k: v for k, v in item.items() if k.startswith('description_')
        }

    return descriptions_dict


def merge_test_with_descriptions(
    test_df: pd.DataFrame,
    descriptions_dict: Dict[str, Dict],
    fill_missing: bool = True
) -> pd.DataFrame:
    """
    test 데이터프레임에 description을 병합

    Args:
        test_df: 테스트 데이터프레임
        descriptions_dict: load_descriptions_json()으로 로드한 딕셔너리
        fill_missing: description이 없는 경우 빈 문자열로 채울지 여부

    Returns:
        description이 추가된 데이터프레임 (원본은 수정하지 않음)
    """
    # 복사본 생성 (원본 수정 방지)
    merged_df = test_df.copy()

    # description 컬럼 개수 확인 (첫 번째 항목 기준)
    if descriptions_dict:
        first_key = next(iter(descriptions_dict))
        desc_columns = sorted(descriptions_dict[first_key].keys())
    else:
        desc_columns = ['description_1', 'description_2']  # 기본값

    # 각 컬럼 초기화
    for col in desc_columns:
        merged_df[col] = None

    # description 병합
    matched_count = 0
    missing_count = 0

    for idx, row in merged_df.iterrows():
        sample_id = str(row['id'])

        if sample_id in descriptions_dict:
            # description이 있는 경우
            for col in desc_columns:
                if col in descriptions_dict[sample_id]:
                    merged_df.at[idx, col] = descriptions_dict[sample_id][col]
            matched_count += 1
        else:
            # description이 없는 경우
            if fill_missing:
                for col in desc_columns:
                    merged_df.at[idx, col] = ""
            missing_count += 1

    print(f"병합 완료: {matched_count}개 매칭, {missing_count}개 누락")

    if missing_count > 0:
        print(f"경고: {missing_count}개 샘플에 대한 description이 없습니다.")

    return merged_df


def merge_and_save(
    test_path: str,
    descriptions_path: str,
    output_path: str,
    fill_missing: bool = True
) -> pd.DataFrame:
    """
    테스트 데이터와 description을 병합하여 저장

    Args:
        test_path: 테스트 데이터 CSV 경로
        descriptions_path: descriptions.json 경로
        output_path: 출력 CSV 경로
        fill_missing: description이 없는 경우 빈 문자열로 채울지 여부

    Returns:
        병합된 데이터프레임
    """
    print(f"테스트 데이터 로드: {test_path}")
    test_df = pd.read_csv(test_path)

    print(f"Descriptions 로드: {descriptions_path}")
    descriptions_dict = load_descriptions_json(descriptions_path)

    print("데이터 병합 중...")
    merged_df = merge_test_with_descriptions(test_df, descriptions_dict, fill_missing)

    print(f"결과 저장: {output_path}")
    merged_df.to_csv(output_path, index=False)

    print("병합 완료!")
    return merged_df


if __name__ == "__main__":
    # 테스트용 실행
    import sys

    if len(sys.argv) >= 4:
        test_path = sys.argv[1]
        descriptions_path = sys.argv[2]
        output_path = sys.argv[3]

        merge_and_save(test_path, descriptions_path, output_path)
    else:
        print("사용법: python merge_descriptions.py <test.csv> <descriptions.json> <output.csv>")
