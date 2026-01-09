import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.seed import set_seed
from src.dapt.dataset import load_dapt_data, prepare_dapt_dataset
from src.dapt.trainer import load_model_and_tokenizer, apply_peft, create_dapt_trainer

console = Console()

@hydra.main(version_base=None, config_path="conf/dapt", config_name="config")
def main(cfg: DictConfig):
    """
    DAPT 학습의 메인 함수.

    Args:
        cfg (DictConfig): Hydra로 로드된 설정 파일
    """
    # 설정 출력
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]DAPT Training Configuration[/bold cyan]",
        border_style="blue"
    ))
    console.print(OmegaConf.to_yaml(cfg))

    # Hydra 출력 디렉토리 가져오기
    output_dir = HydraConfig.get().runtime.output_dir
    console.print(f"\n[bold green]Saving outputs to:[/bold green] [italic]{output_dir}[/italic]\n")

    set_seed(cfg.seed)

    # 1. 데이터 로드
    console.print(Panel.fit(
        "[bold yellow]1. Loading DAPT Data[/bold yellow]",
        border_style="yellow"
    ))

    # config에서 절대 경로로 변환
    dapt_data_path = hydra.utils.to_absolute_path(cfg.data.dapt_path)
    df = load_dapt_data(dapt_data_path)

    # 2. 모델 & 토크나이저 로드
    console.print(Panel.fit(
        "[bold yellow]2. Loading Model & Tokenizer[/bold yellow]",
        border_style="yellow"
    ))

    model, tokenizer = load_model_and_tokenizer(
        cfg.name,
        torch_dtype=cfg.get('torch_dtype', 'float16')
    )

    # 3. 데이터셋 준비 (토크나이징 & split)
    console.print(Panel.fit(
        "[bold yellow]3. Preparing Dataset[/bold yellow]",
        border_style="yellow"
    ))

    train_dataset, eval_dataset = prepare_dapt_dataset(
        df=df,
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
        test_size=cfg.data.test_size,
        seed=cfg.seed,
    )

    # 샘플 데이터 확인
    console.print("\n[bold]Sample Input Preview[/bold]")
    sample = train_dataset[0]
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    
    sample_table = Table(show_header=False, box=None)
    sample_table.add_column("Key", style="cyan")
    sample_table.add_column("Value", style="white")
    sample_table.add_row("Input length", f"{len(sample['input_ids'])} tokens")
    sample_table.add_row("Preview", decoded[:200] + "..." if len(decoded) > 200 else decoded)
    console.print(sample_table)

    # PEFT(LoRA) 적용 (선택 사항))
    if cfg.get('use_peft', True):
        console.print(Panel.fit(
            "[bold yellow] Applying PEFT (LoRA)[/bold yellow]",
            border_style="yellow"
        ))
        model = apply_peft(
            model,
            OmegaConf.to_container(cfg.peft, resolve=True)
        )
    else:
        console.print("[yellow]Skipping PEFT - Full fine-tuning mode[/yellow]")

    # 4. 트레이너 초기화
    console.print(Panel.fit(
        "[bold yellow]4. Initializing Trainer[/bold yellow]",
        border_style="yellow"
    ))

    # training config 추출 (model, data, seed, hydra 제외)
    training_cfg = {
        k: v for k, v in cfg.items()
        if k not in ['name', 'torch_dtype', 'response_template', 'use_peft', 'peft',
                     'seed', 'data', 'hydra']
    }

    trainer = create_dapt_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        training_config=training_cfg,
    )

    # 5. 학습
    console.print(Panel.fit(
        "[bold green]5. Starting DAPT Training[/bold green]",
        border_style="green"
    ))

    trainer.train()

    # 6. 최종 모델 저장
    console.print(Panel.fit(
        "[bold blue]6. Saving Final Model[/bold blue]",
        border_style="blue"
    ))

    final_model_path = f"{output_dir}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    console.print(f"[green]✓[/green] Model saved to: [italic]{final_model_path}[/italic]")

    # 학습 결과 요약
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]DAPT Training Complete! [/bold green]",
        border_style="green"
    ))
    
    # 학습 결과 테이블 생성
    results_table = Table(title="Training Results", show_header=True)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    if trainer.state.best_metric is not None:
        metric_name = training_cfg.get('metric_for_best_model', 'loss')
        results_table.add_row(f"Best {metric_name}", f"{trainer.state.best_metric:.4f}")
        
        if trainer.state.best_model_checkpoint:
            results_table.add_row("Best checkpoint", trainer.state.best_model_checkpoint)
    
    # 최종 eval loss 추가
    if trainer.state.log_history:
        eval_logs = [log for log in trainer.state.log_history if 'eval_loss' in log]
        if eval_logs:
            final_eval_log = eval_logs[-1]
            results_table.add_row("Final eval loss", f"{final_eval_log['eval_loss']:.4f}")
            
            if 'perplexity' in final_eval_log:
                perplexity = final_eval_log.get('perplexity', 'N/A')
                results_table.add_row("Perplexity", str(perplexity))
    
    console.print(results_table)
    console.print("\n")


if __name__ == "__main__":
    main()
