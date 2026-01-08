import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import random
import ast


class FineTuningDataPreparer:
    """
    OpenAI Fine-tuning을 위한 데이터 준비 클래스

    CSV 데이터를 OpenAI Fine-tuning JSONL 형식으로 변환하고
    Train/Validation 데이터로 분리하여 저장합니다.
    """

    def __init__(self, csv_path: str, output_dir: str = "data/finetuning"):
        """
        FineTuningDataPreparer 초기화

        Args:
            csv_path: 입력 CSV 파일 경로 (품질 평가가 완료된 데이터)
            output_dir: JSONL 파일 출력 디렉토리 경로
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.system_message = (
            "당신은 지문으로 수능형 객관식 문제를 생성하는 전문가입니다."
        )
        self.user_template = """다음 지문을 읽고 JSON 형식으로 문제를 생성하세요.

**출력 형식:**
반드시 아래의 JSON 형식으로만 출력하세요. 다른 설명이나 주석은 포함하지 마세요.

{{
  "paragraph": "지문 (제공된 지문 그대로)",
  "question": "문제",
  "choices": ["선택지1", "선택지2", "선택지3", "선택지4", "선택지5"],
  "answer": 1,
  "question_plus": "내용..." 또는 null
}}

지문:
{paragraph}"""

    def load_top_quality_data(self, top_n: int = 2000) -> pd.DataFrame:
        """
        CSV에서 total_score 상위 N개 데이터 로드

        Args:
            top_n: 선택할 상위 데이터 개수

        Returns:
            total_score 기준 내림차순 정렬된 상위 N개 데이터프레임
        """
        print(f"📊 Loading top {top_n} quality questions from CSV...")
        df = pd.read_csv(self.csv_path)

        # total_score 기준 내림차순 정렬 후 상위 N개 선택
        df_sorted = df.sort_values("total_score", ascending=False)
        df_top = df_sorted.head(top_n)

        print(
            f"   ✓ Loaded {len(df_top)} samples (total_score range: {df_top['total_score'].min()}-{df_top['total_score'].max()})"
        )
        return df_top

    def convert_csv_to_examples(self, df: pd.DataFrame) -> List[Dict]:
        """
        CSV 데이터를 Fine-tuning 예제로 변환

        Args:
            df: 변환할 데이터프레임

        Returns:
            Fine-tuning 예제 딕셔너리 리스트
            각 예제는 source, id, paragraph, assistant_content 필드 포함
        """
        print("🔄 Converting CSV data to training examples...")
        examples = []
        errors = []

        for idx, row in df.iterrows():
            try:
                paragraph = row["paragraph"]

                # problems 컬럼은 Python dict 형식의 문자열 (작은따옴표 사용)
                problems_str = row["problems"]
                if isinstance(problems_str, str):
                    # ast.literal_eval을 사용하여 안전하게 Python dict 파싱
                    problems_dict = ast.literal_eval(problems_str)
                else:
                    problems_dict = problems_str

                # Assistant 응답 구성 (data_augmentation.py 호환 - paragraph 포함)
                assistant_content = {
                    "paragraph": paragraph,
                    "question": problems_dict["question"],
                    "choices": problems_dict["choices"],
                    "answer": problems_dict["answer"],
                }

                # question_plus가 있으면 추가
                if pd.notna(row.get("question_plus")) and row["question_plus"]:
                    assistant_content["question_plus"] = row["question_plus"]

                # description이 있으면 추가
                if pd.notna(row.get("description")) and row["description"]:
                    assistant_content["description"] = row["description"]

                examples.append(
                    {
                        "source": "csv",
                        "id": row["id"],
                        "paragraph": paragraph,
                        "assistant_content": assistant_content,
                    }
                )

            except Exception as e:
                errors.append({"index": idx, "id": row.get("id"), "error": str(e)})

        print(f"   ✓ Converted {len(examples)} examples ({len(errors)} errors)")
        if errors:
            print(f"   ⚠ Errors encountered: {errors[:5]}...")  # 처음 5개만 출력

        return examples

    def validate_example(self, example: Dict) -> tuple[bool, str]:
        """
        예제 검증 (4지선다 또는 5지선다 허용)

        Args:
            example: 검증할 예제 딕셔너리

        Returns:
            (검증 성공 여부, 오류 메시지) 튜플
        """
        try:
            # Paragraph 검증
            if not example.get("paragraph") or not isinstance(
                example["paragraph"], str
            ):
                return False, "Invalid paragraph"

            # Paragraph가 너무 짧으면 제외
            if len(example["paragraph"].strip()) < 50:
                return False, "Paragraph too short"

            assistant = example["assistant_content"]

            # Question 검증
            if not assistant.get("question") or not isinstance(
                assistant["question"], str
            ):
                return False, "Invalid question"

            # Choices 검증 - 4지선다 또는 5지선다만 허용
            choices = assistant.get("choices")
            if not choices or not isinstance(choices, list):
                return False, "Choices must be a list"

            if len(choices) not in [4, 5]:
                return (
                    False,
                    f"Only 4 or 5-choice questions allowed (got {len(choices)})",
                )

            # 모든 선택지가 문자열이고 비어있지 않은지 확인
            for i, choice in enumerate(choices):
                if not isinstance(choice, str) or not choice.strip():
                    return False, f"Invalid choice at index {i}"

            # Answer 검증 (choices 개수에 맞춰 동적으로)
            answer = assistant.get("answer")
            num_choices = len(choices)
            if not isinstance(answer, int) or answer < 1 or answer > num_choices:
                return False, f"Invalid answer (must be 1-{num_choices}, got {answer})"

            return True, "OK"

        except Exception as e:
            return False, str(e)

    def create_openai_format(self, examples: List[Dict]) -> List[Dict]:
        """
        OpenAI Fine-tuning JSONL 형식으로 변환

        Args:
            examples: 변환할 예제 리스트

        Returns:
            OpenAI Fine-tuning 형식의 딕셔너리 리스트
            각 항목은 messages 필드를 포함 (system, user, assistant role)
        """
        print("🎯 Creating OpenAI fine-tuning format...")
        formatted_examples = []
        validation_errors = []

        for example in examples:
            # 검증
            is_valid, error_msg = self.validate_example(example)
            if not is_valid:
                validation_errors.append({"id": example["id"], "error": error_msg})
                continue

            # OpenAI 형식으로 변환
            formatted = {
                "messages": [
                    {"role": "system", "content": self.system_message},
                    {
                        "role": "user",
                        "content": self.user_template.format(
                            paragraph=example["paragraph"]
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            example["assistant_content"], ensure_ascii=False
                        ),
                    },
                ]
            }
            formatted_examples.append(formatted)

        print(f"   ✓ Created {len(formatted_examples)} valid examples")
        if validation_errors:
            print(f"   ⚠ Validation errors: {len(validation_errors)}")
            print(f"      First 5 errors: {validation_errors[:5]}")

        return formatted_examples

    def split_train_val(
        self, examples: List[Dict], val_ratio: float = 0.15
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Train/Validation 데이터 분리

        Args:
            examples: 분리할 예제 리스트
            val_ratio: Validation 데이터 비율 (기본값: 0.15)

        Returns:
            (train_data, val_data) 튜플
        """
        print(f"✂️  Splitting data (validation ratio: {val_ratio})...")

        # 랜덤 셔플
        shuffled = examples.copy()
        random.shuffle(shuffled)

        # 분리
        val_size = int(len(shuffled) * val_ratio)
        val_data = shuffled[:val_size]
        train_data = shuffled[val_size:]

        print(f"   ✓ Train: {len(train_data)} samples")
        print(f"   ✓ Validation: {len(val_data)} samples")

        return train_data, val_data

    def save_jsonl(self, data: List[Dict], filename: str):
        """
        JSONL 파일로 저장

        Args:
            data: 저장할 데이터 리스트
            filename: 저장할 파일명
        """
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"   💾 Saved to {filepath}")

    def generate_report(
        self, train_data: List[Dict], val_data: List[Dict], csv_count: int
    ):
        """
        요약 보고서 생성 및 저장

        Args:
            train_data: Train 데이터 리스트
            val_data: Validation 데이터 리스트
            csv_count: CSV에서 변환된 예제 개수
        """
        report = {
            "total_samples": len(train_data) + len(val_data),
            "train_samples": len(train_data),
            "validation_samples": len(val_data),
            "csv_source_count": csv_count,
            "validation_ratio": len(val_data) / (len(train_data) + len(val_data)),
            "output_files": {
                "train": str(self.output_dir / "train.jsonl"),
                "validation": str(self.output_dir / "validation.jsonl"),
            },
        }

        report_path = self.output_dir / "preparation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, indent=2, ensure_ascii=False, fp=f)

        print("\n" + "=" * 60)
        print("📋 PREPARATION REPORT")
        print("=" * 60)
        print(f"Total samples:      {report['total_samples']}")
        print(f"  - Train:          {report['train_samples']}")
        print(f"  - Validation:     {report['validation_samples']}")
        print(f"\nData sources:")
        print(f"  - CSV (top quality): {report['csv_source_count']}")
        print(f"\nOutput files:")
        print(f"  - Train:       {report['output_files']['train']}")
        print(f"  - Validation:  {report['output_files']['validation']}")
        print(f"  - Report:      {report_path}")
        print("=" * 60)

    def run(self, top_n: int = 2000, val_ratio: float = 0.15):
        """
        전체 Fine-tuning 데이터 준비 파이프라인 실행

        Args:
            top_n: 선택할 상위 데이터 개수
            val_ratio: Validation 데이터 비율

        Process:
            1. CSV에서 top_n개 고품질 데이터 로드
            2. Fine-tuning 예제로 변환
            3. OpenAI 형식으로 변환 및 검증
            4. Train/Validation 분리
            5. JSONL 파일 저장
            6. 요약 보고서 생성
        """
        print("\n🚀 Starting Fine-tuning Data Preparation Pipeline\n")

        # 1. 데이터 로드
        csv_df = self.load_top_quality_data(top_n)

        # 2. 변환
        csv_examples = self.convert_csv_to_examples(csv_df)

        # 3. OpenAI 형식으로 변환
        formatted_examples = self.create_openai_format(csv_examples)

        # 4. Train/Val 분리
        train_data, val_data = self.split_train_val(formatted_examples, val_ratio)

        # 5. 저장
        print("\n💾 Saving files...")
        self.save_jsonl(train_data, "train.jsonl")
        self.save_jsonl(val_data, "validation.jsonl")

        # 6. 보고서 생성
        self.generate_report(train_data, val_data, len(csv_examples))

        print("\n✅ Pipeline completed successfully!\n")


def main():
    """
    메인 실행 함수

    품질 평가가 완료된 CSV 데이터를 OpenAI Fine-tuning 형식으로 변환하고
    Train/Validation JSONL 파일을 생성합니다.
    """
    # 설정
    CSV_PATH = "data/aihub_workbook_top_quality.csv"
    OUTPUT_DIR = "data/finetuning"
    TOP_N = 2000
    VAL_RATIO = 0.15

    # 랜덤 시드 고정 (재현성)
    random.seed(42)

    # 실행
    preparer = FineTuningDataPreparer(CSV_PATH, OUTPUT_DIR)
    preparer.run(top_n=TOP_N, val_ratio=VAL_RATIO)


if __name__ == "__main__":
    main()
