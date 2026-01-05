import subprocess
import sys
import os
import argparse
from rich.console import Console
from rich.panel import Panel

console = Console()


def run_sktax_description_generation():
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
    return True


def run_exaone_inference():
    """
    Step 2: EXAONE 모델로 최종 답변 생성
    """
    console.print("\n")
    console.print(Panel.fit(
        "[bold yellow]STEP 2: EXAONE Inference with MoA[/bold yellow]",
        border_style="yellow"
    ))

    # descriptions.json 존재 확인
    if os.path.exists("descriptions.json"):
        console.print("[green]✓[/green] descriptions.json 파일을 찾았습니다.")
        console.print("[green]✓[/green] MoA 모드로 실행됩니다.\n")
    else:
        console.print("[yellow]⚠[/yellow] descriptions.json 파일이 없습니다.")
        console.print("[yellow]⚠[/yellow] 기본 모드로 실행됩니다.\n")

    cmd = [sys.executable, "inference_exaone.py"]

    result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode != 0:
        console.print("\n[red]✗ EXAONE inference 중 오류가 발생했습니다.[/red]")
        return False

    console.print("\n[green]✓ EXAONE inference 완료![/green]")
    return True


def main():
    """
    통합 파이프라인 메인 함수
    """
    parser = argparse.ArgumentParser(
        description="MoA 파이프라인: sktAX description 생성 + EXAONE inference"
    )
    parser.add_argument(
        "--skip-description",
        action="store_true",
        help="description 생성을 건너뛰고 EXAONE inference만 실행",
    )
    parser.add_argument(
        "--description-only",
        action="store_true",
        help="description 생성만 실행하고 EXAONE inference는 건너뜀",
    )

    args = parser.parse_args()

    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]MoA (Mixture of Agents) 통합 파이프라인[/bold cyan]",
        border_style="cyan"
    ))

    # Step 1: sktAX Description Generation
    if not args.skip_description:
        success = run_sktax_description_generation()
        if not success:
            console.print("\n[red]파이프라인 중단: sktAX description 생성 실패[/red]")
            sys.exit(1)

        if args.description_only:
            console.print("\n")
            console.print(Panel.fit(
                "[bold green]Description 생성만 완료되었습니다.[/bold green]",
                border_style="green"
            ))
            return
    else:
        console.print("\n[yellow]⏭  sktAX description 생성을 건너뜁니다.[/yellow]")

    # Step 2: EXAONE Inference
    success = run_exaone_inference()
    if not success:
        console.print("\n[red]파이프라인 중단: EXAONE inference 실패[/red]")
        sys.exit(1)

    # 완료
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]전체 파이프라인 완료![/bold green]",
        border_style="green"
    ))

    console.print("\n[bold]생성된 파일:[/bold]")
    files_created = []
    if os.path.exists("descriptions.csv"):
        files_created.append("descriptions.csv")
    if os.path.exists("descriptions.json"):
        files_created.append("descriptions.json")
    if os.path.exists("TestSet_Inference_EXAONE-4.0-32B.csv"):
        files_created.append("TestSet_Inference_EXAONE-4.0-32B.csv")
    if os.path.exists("outputs/moa/submission.csv"):
        files_created.append("outputs/moa/submission.csv")

    for file in files_created:
        console.print(f"  [green]✓[/green] {file}")
    console.print()


if __name__ == "__main__":
    main()
