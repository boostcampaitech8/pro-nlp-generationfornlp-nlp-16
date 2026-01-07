import os
import json
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.model.model import load_model_for_inference
from src.data.dataset import load_test_data
from src.inference.generate_description import (
    generate_descriptions_batch,
    prepare_test_sample,
)
from src.data.description import save_descriptions

console = Console()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    description 생성 inference 전체 파이프라인을 실행하는 메인 함수
    """

    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Description Generation Inference[/bold cyan]",
        border_style="cyan"
    ))

    # 1. 설정 및 경로 준비
    checkpoint_path = cfg.inference.checkpoint_dir
    checkpoint_step = cfg.inference.checkpoint_step
    test_data_path = hydra.utils.to_absolute_path(cfg.data.test_path)
    csv_output_path = cfg.inference.get("description_output_csv", "descriptions.csv")
    json_output_path = cfg.inference.get("description_output_json", "descriptions.json")

    original_cwd = hydra.utils.get_original_cwd()

    # 2. 체크포인트 자동 탐색
    if not checkpoint_path:
        console.print("[yellow]Checkpoint 경로가 지정되지 않아 자동 탐색을 수행합니다.[/yellow]")

        outputs_roots = [
            os.path.join(original_cwd, "outputs", "dapt"),
            os.path.join(original_cwd, "outputs", "train"),
            os.path.join(original_cwd, "outputs"),
        ]

        found_path = None
        for root in outputs_roots:
            if not os.path.exists(root):
                continue

            for date in sorted(os.listdir(root), reverse=True):
                date_dir = os.path.join(root, date)
                if not os.path.isdir(date_dir):
                    continue

                for time in sorted(os.listdir(date_dir), reverse=True):
                    run_dir = os.path.join(date_dir, time)
                    if not os.path.isdir(run_dir):
                        continue

                    if os.path.exists(os.path.join(run_dir, "final_model")):
                        found_path = os.path.join(run_dir, "final_model")
                        break
                    elif any(d.startswith("checkpoint-") for d in os.listdir(run_dir)):
                        found_path = run_dir
                        break

                if found_path:
                    break
            if found_path:
                break

        if not found_path:
            raise ValueError("사용 가능한 checkpoint를 찾지 못했습니다.")

        checkpoint_path = found_path
        print(f"자동 탐색된 checkpoint: {checkpoint_path}")

    if not os.path.isabs(checkpoint_path):
        checkpoint_path = hydra.utils.to_absolute_path(checkpoint_path)

    # 3. 특정 checkpoint 선택
    if (
        "final_model" not in checkpoint_path
        and "checkpoint-" not in os.path.basename(checkpoint_path)
        and os.path.isdir(checkpoint_path)
    ):
        if checkpoint_step == "best":
            checkpoints = [d for d in os.listdir(checkpoint_path) if d.startswith("checkpoint-")]
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(checkpoint_path, checkpoints[-1])
        else:
            candidate = os.path.join(checkpoint_path, f"checkpoint-{checkpoint_step}")
            if os.path.exists(candidate):
                checkpoint_path = candidate

    # 경로 정보 테이블
    path_table = Table(title="Configuration Paths", show_header=False, box=None)
    path_table.add_column("Key", style="bold cyan")
    path_table.add_column("Value", style="white")

    path_table.add_row("Checkpoint Path", checkpoint_path)
    path_table.add_row("Test Data Path", test_data_path)
    path_table.add_row("CSV Output Path", csv_output_path)
    path_table.add_row("JSON Output Path", json_output_path)

    console.print("\n")
    console.print(path_table)

    # 4. 모델 로드
    console.print("\n")
    console.print(Panel.fit(
        "[bold yellow]Loading Model & Tokenizer[/bold yellow]",
        border_style="yellow"
    ))
    torch_dtype = cfg.inference.get("torch_dtype", "bfloat16")
    model, tokenizer = load_model_for_inference(
        checkpoint_path,
        torch_dtype=torch_dtype,
    )
    console.print(f"[green]✓[/green] 모델 로드 완료 (dtype={torch_dtype})")

    # 5. 테스트 데이터 로드
    console.print("\n")
    console.print(Panel.fit(
        "[bold yellow]Loading Test Data[/bold yellow]",
        border_style="yellow"
    ))
    test_df = load_test_data(test_data_path)

    # 실사용시 주석처리! (디버그용)
    # test_df = test_df.head(5)
    console.print(f"[green]✓[/green] 테스트 샘플 수: {len(test_df)}")

    # 6. description 생성용 샘플 준비
    test_samples = []
    for _, row in test_df.iterrows():
        sample = prepare_test_sample(
            sample_id=row["id"],
            paragraph=row["paragraph"],
            question=row["question"],
            question_plus=row.get("question_plus", None),
            choices=row["choices"],
        )

        test_samples.append(sample)

    console.print(f"[green]✓[/green] 총 {len(test_samples)}개의 샘플을 준비했습니다.")

    # 7. generation 설정
    gen_cfg = cfg.inference.get("generation", {})
    max_new_tokens = gen_cfg.get("max_new_tokens", 384)
    temperatures = gen_cfg.get("temperatures", [0.7])
    top_p = gen_cfg.get("top_p", 0.9)
    top_k = gen_cfg.get("top_k", 50)
    repetition_penalty = gen_cfg.get("repetition_penalty", 1.1)
    do_sample = gen_cfg.get("do_sample", True)

    console.print("\n[bold]Generation 설정[/bold]")
    gen_table = Table(show_header=False, box=None)
    gen_table.add_column("Parameter", style="cyan")
    gen_table.add_column("Value", style="white")

    gen_table.add_row("max_new_tokens", str(max_new_tokens))
    gen_table.add_row("temperatures", str(temperatures))
    gen_table.add_row("top_p", str(top_p))
    gen_table.add_row("top_k", str(top_k))
    gen_table.add_row("repetition_penalty", str(repetition_penalty))
    gen_table.add_row("do_sample", str(do_sample))

    console.print(gen_table)

    # 8. description 생성
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]Starting Description Generation[/bold green]",
        border_style="green"
    ))
    results = generate_descriptions_batch(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_samples,
        max_new_tokens=max_new_tokens,
        temperatures=temperatures,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        show_progress=True,
    )

    # 9. 결과 저장
    console.print("\n")
    console.print(Panel.fit(
        "[bold blue]Saving Results[/bold blue]",
        border_style="blue"
    ))
    # data/ 폴더는 프로젝트 루트 기준으로 저장
    data_dir = os.path.join(original_cwd, "data")
    save_descriptions(results, csv_path=csv_output_path, json_path=json_output_path, data_dir=data_dir)

    # 10. 샘플 출력
    console.print("\n[bold]샘플 결과 (상위 3개)[/bold]")
    for i in range(min(3, len(results))):
        console.print(f"\n[cyan]--- Sample {i + 1} ---[/cyan]")
        console.print(f"[bold]ID:[/bold] {results[i]['id']}")
        for key in sorted(results[i].keys()):
            if key.startswith("description_"):
                console.print(f"\n[yellow]{key}:[/yellow]")
                console.print(results[i][key])

    console.print("\n")
    console.print(Panel.fit(
        "[bold green]Description Generation Complete![/bold green]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
