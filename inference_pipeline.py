import subprocess
import sys
import os
import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel

console = Console()


def run_sktax_description_generation(cfg: DictConfig):
    """
    Step 1: sktAX 모델로 description 생성
    """
    console.print("\n")
    console.print(Panel.fit(
        "[bold yellow]STEP 1: sktAX Description Generation[/bold yellow]",
        border_style="yellow"
    ))

    cmd = [sys.executable, "inference_description.py"]
    result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode != 0:
        console.print("\n[red]✗ sktAX description 생성 중 오류가 발생했습니다.[/red]")
        return False

    console.print("\n[green]✓ sktAX description 생성 완료![/green]")

    # 출력 검증
    if cfg.pipeline.verify_outputs:
        verify_outputs(cfg.pipeline.required_outputs.description)

    return True


def run_exaone_inference(cfg: DictConfig):
    """
    Step 2: EXAONE 모델로 최종 답변 생성
    """
    console.print("\n")
    console.print(Panel.fit(
        "[bold yellow]STEP 2: EXAONE Inference with MoA[/bold yellow]",
        border_style="yellow"
    ))

    # descriptions.json 존재 확인
    descriptions_path = cfg.data.descriptions_json
    if os.path.exists(descriptions_path):
        console.print(f"[green]✓[/green] {descriptions_path} 파일을 찾았습니다.")
        console.print("[green]✓[/green] MoA 모드로 실행됩니다.\n")
    else:
        console.print(f"[yellow]⚠[/yellow] {descriptions_path} 파일이 없습니다.")
        console.print("[yellow]⚠[/yellow] 기본 모드로 실행됩니다.\n")

    cmd = [sys.executable, "inference_exaone.py"]
    result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode != 0:
        console.print("\n[red]✗ EXAONE inference 중 오류가 발생했습니다.[/red]")
        return False

    console.print("\n[green]✓ EXAONE inference 완료![/green]")

    # 출력 검증
    if cfg.pipeline.verify_outputs:
        verify_outputs(cfg.pipeline.required_outputs.exaone)

    return True


def verify_outputs(file_list):
    """출력 파일 검증"""
    console.print("\n[bold]출력 파일 확인:[/bold]")
    for file_path in file_list:
        if os.path.exists(file_path):
            console.print(f"  [green]✓[/green] {file_path}")
        else:
            console.print(f"  [yellow]⚠[/yellow] {file_path} (없음)")


def print_completion_summary(cfg: DictConfig):
    """파이프라인 완료 요약"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]전체 파이프라인 완료![/bold green]",
        border_style="green"
    ))

    console.print("\n[bold]생성된 파일:[/bold]")
    all_outputs = (
        list(cfg.pipeline.required_outputs.description) +
        list(cfg.pipeline.required_outputs.exaone)
    )

    for file_path in all_outputs:
        if os.path.exists(file_path):
            console.print(f"  [green]✓[/green] {file_path}")

    console.print()


@hydra.main(version_base=None, config_path="conf", config_name="moa_config")
def main(cfg: DictConfig):
    """
    MoA 통합 파이프라인 메인 함수
    """
    console.print("\n")
    console.print(Panel.fit(
        f"[bold cyan]MoA (Mixture of Agents) 통합 파이프라인 - Mode: {cfg.pipeline.mode}[/bold cyan]",
        border_style="cyan"
    ))

    # Validate mode
    valid_modes = ["full", "description", "inference"]
    if cfg.pipeline.mode not in valid_modes:
        console.print(f"\n[red]✗ 잘못된 mode: {cfg.pipeline.mode}[/red]")
        console.print(f"[red]유효한 mode: {', '.join(valid_modes)}[/red]")
        sys.exit(1)

    # Step 1: sktAX Description Generation
    if cfg.pipeline.mode in ["full", "description"]:
        success = run_sktax_description_generation(cfg)
        if not success:
            console.print("\n[red]파이프라인 중단: sktAX description 생성 실패[/red]")
            sys.exit(1)

        if cfg.pipeline.mode == "description":
            console.print("\n")
            console.print(Panel.fit(
                "[bold green]Description 생성만 완료되었습니다.[/bold green]",
                border_style="green"
            ))
            return
    else:
        console.print("\n[yellow]⏭  sktAX description 생성을 건너뜁니다.[/yellow]")

    # Step 2: EXAONE Inference
    success = run_exaone_inference(cfg)
    if not success:
        console.print("\n[red]파이프라인 중단: EXAONE inference 실패[/red]")
        sys.exit(1)

    # 완료 메시지
    print_completion_summary(cfg)


if __name__ == "__main__":
    main()
