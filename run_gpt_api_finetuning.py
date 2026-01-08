"""
OpenAI GPT-4o-mini Fine-tuning 실행 스크립트

준비된 JSONL 데이터를 OpenAI API에 업로드하고 fine-tuning 작업을 시작합니다.
"""

import os
import time
import json
from pathlib import Path
from openai import OpenAI
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class FineTuningRunner:
    def __init__(self, api_key: Optional[str] = None):
        """
        OpenAI Fine-tuning Runner 초기화

        Args:
            api_key: OpenAI API 키 (None이면 환경변수에서 읽음)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key가 필요합니다. "
                "환경변수 OPENAI_API_KEY를 설정하거나 api_key 파라미터를 전달하세요."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.training_file_id = None
        self.validation_file_id = None
        self.job_id = None

    def upload_file(self, file_path: str, purpose: str = "fine-tune") -> str:
        """
        파일을 OpenAI에 업로드

        Args:
            file_path: 업로드할 파일 경로
            purpose: 파일 용도 ('fine-tune')

        Returns:
            업로드된 파일 ID
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        print(f"📤 Uploading {file_path.name}...")

        with open(file_path, "rb") as f:
            response = self.client.files.create(file=f, purpose=purpose)

        file_id = response.id
        print(f"   ✓ Uploaded successfully: {file_id}")
        print(f"   - Filename: {response.filename}")
        print(f"   - Bytes: {response.bytes:,}")
        print(f"   - Status: {response.status}")

        return file_id

    def wait_for_file_processing(self, file_id: str, max_wait: int = 300):
        """
        파일 처리가 완료될 때까지 대기

        Args:
            file_id: 파일 ID
            max_wait: 최대 대기 시간 (초)
        """
        print(f"⏳ Waiting for file {file_id} to be processed...")

        start_time = time.time()
        while time.time() - start_time < max_wait:
            file_info = self.client.files.retrieve(file_id)
            status = file_info.status

            if status == "processed":
                print(f"   ✓ File processed successfully")
                return
            elif status == "error":
                raise RuntimeError(f"File processing failed: {file_info}")

            print(f"   - Status: {status} (waiting...)")
            time.sleep(5)

        raise TimeoutError(f"File processing timeout after {max_wait}s")

    def create_finetuning_job(
        self,
        training_file_id: str,
        validation_file_id: Optional[str] = None,
        model: str = "gpt-4o-mini-2024-07-18",
        suffix: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
    ) -> str:
        """
        Fine-tuning 작업 생성

        Args:
            training_file_id: 학습 데이터 파일 ID
            validation_file_id: 검증 데이터 파일 ID (선택)
            model: 기본 모델 이름
            suffix: 모델 이름 접미사 (최대 18자)
            hyperparameters: 하이퍼파라미터 설정

        Returns:
            Fine-tuning 작업 ID
        """
        print(f"\n🚀 Creating fine-tuning job...")
        print(f"   - Base model: {model}")
        print(f"   - Training file: {training_file_id}")
        if validation_file_id:
            print(f"   - Validation file: {validation_file_id}")

        # 기본 하이퍼파라미터
        default_hyperparameters = {
            "n_epochs": "auto",  # OpenAI가 자동으로 최적 epoch 수 결정
        }

        if hyperparameters:
            default_hyperparameters.update(hyperparameters)

        # Fine-tuning 작업 생성
        job_params = {
            "training_file": training_file_id,
            "model": model,
            "hyperparameters": default_hyperparameters,
        }

        if validation_file_id:
            job_params["validation_file"] = validation_file_id

        if suffix:
            job_params["suffix"] = suffix[:18]  # 최대 18자 제한

        response = self.client.fine_tuning.jobs.create(**job_params)

        job_id = response.id
        print(f"\n   ✓ Fine-tuning job created: {job_id}")
        print(f"   - Status: {response.status}")
        print(f"   - Created at: {response.created_at}")

        return job_id

    def monitor_job(self, job_id: str, check_interval: int = 60):
        """
        Fine-tuning 작업 모니터링

        Args:
            job_id: Fine-tuning 작업 ID
            check_interval: 상태 확인 간격 (초)
        """
        print(f"\n📊 Monitoring fine-tuning job: {job_id}")
        print(f"   (Checking every {check_interval} seconds)\n")

        last_event_id = None

        while True:
            # 작업 상태 확인
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status

            # 이벤트 로그 출력
            events = self.client.fine_tuning.jobs.list_events(
                fine_tuning_job_id=job_id, limit=10
            )

            for event in reversed(events.data):
                if last_event_id is None or event.created_at > last_event_id:
                    timestamp = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(event.created_at)
                    )
                    print(f"   [{timestamp}] {event.message}")
                    last_event_id = event.created_at

            # 상태별 처리
            if status == "succeeded":
                print(f"\n✅ Fine-tuning completed successfully!")
                print(f"   - Fine-tuned model: {job.fine_tuned_model}")
                print(f"   - Trained tokens: {job.trained_tokens:,}")

                # 결과 저장
                self.save_job_result(job)
                return job

            elif status == "failed":
                print(f"\n❌ Fine-tuning failed!")
                if job.error:
                    print(f"   Error: {job.error}")
                raise RuntimeError(f"Fine-tuning job failed: {job.error}")

            elif status == "cancelled":
                print(f"\n⚠️  Fine-tuning was cancelled")
                return job

            elif status in ["validating_files", "queued", "running"]:
                print(f"   Status: {status}...")
                time.sleep(check_interval)

            else:
                print(f"   Unknown status: {status}")
                time.sleep(check_interval)

    def save_job_result(self, job):
        """Fine-tuning 결과 저장"""
        result = {
            "job_id": job.id,
            "status": job.status,
            "model": job.model,
            "fine_tuned_model": job.fine_tuned_model,
            "trained_tokens": job.trained_tokens,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
        }

        output_path = Path("data/finetuning/job_result.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n   💾 Results saved to: {output_path}")

    def run(
        self,
        train_file: str = "data/finetuning/train.jsonl",
        val_file: str = "data/finetuning/validation.jsonl",
        model: str = "gpt-4o-mini-2024-07-18",
        suffix: str = "korean-qa",
        hyperparameters: Optional[dict] = None,
        monitor: bool = True,
    ):
        """
        전체 Fine-tuning 파이프라인 실행

        Args:
            train_file: 학습 데이터 파일 경로
            val_file: 검증 데이터 파일 경로
            model: 기본 모델
            suffix: 모델 이름 접미사
            hyperparameters: 하이퍼파라미터
            monitor: 작업 모니터링 여부
        """
        print("=" * 70)
        print("🎯 OpenAI Fine-tuning Pipeline")
        print("=" * 70)

        # 1. 파일 업로드
        print("\n[Step 1] Uploading files...")
        self.training_file_id = self.upload_file(train_file)

        if Path(val_file).exists():
            self.validation_file_id = self.upload_file(val_file)
        else:
            print(f"⚠️  Validation file not found: {val_file}")
            self.validation_file_id = None

        # 2. 파일 처리 대기
        print("\n[Step 2] Waiting for file processing...")
        self.wait_for_file_processing(self.training_file_id)
        if self.validation_file_id:
            self.wait_for_file_processing(self.validation_file_id)

        # 3. Fine-tuning 작업 생성
        print("\n[Step 3] Creating fine-tuning job...")
        self.job_id = self.create_finetuning_job(
            training_file_id=self.training_file_id,
            validation_file_id=self.validation_file_id,
            model=model,
            suffix=suffix,
            hyperparameters=hyperparameters,
        )

        # 4. 작업 모니터링
        if monitor:
            print("\n[Step 4] Monitoring fine-tuning job...")
            job = self.monitor_job(self.job_id)
            return job
        else:
            print(f"\n✓ Fine-tuning job started: {self.job_id}")
            print(f"  Monitor at: https://platform.openai.com/finetune/{self.job_id}")
            return None


def main():
    """메인 실행 함수"""
    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("\n다음과 같이 설정하세요:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return

    # Fine-tuning 실행
    runner = FineTuningRunner(api_key=api_key)

    try:
        runner.run(
            train_file="data/finetuning/train.jsonl",
            val_file="data/finetuning/validation.jsonl",
            model="gpt-4o-mini-2024-07-18",
            suffix="ksat-qa-finetuned",
            hyperparameters={
                "n_epochs": "auto",  # 자동으로 최적 epoch 수 결정
                # "n_epochs": 3,  # 또는 직접 지정
            },
            monitor=True,  # 실시간 모니터링
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        if runner.job_id:
            print(f"   Job ID: {runner.job_id}")
            print(
                f"   Monitor at: https://platform.openai.com/finetune/{runner.job_id}"
            )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
