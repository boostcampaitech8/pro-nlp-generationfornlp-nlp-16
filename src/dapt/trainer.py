import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def load_model_and_tokenizer(model_name: str, torch_dtype: str = "float16"):
    """
    DAPT를 위한 모델과 토크나이저를 로드하는 함수.

    Args:
        model_name (str): HuggingFace 모델 이름
        torch_dtype (str): 모델 가중치 데이터 타입 ("float16", "bfloat16", "float32")

    Returns:
        tuple: (model, tokenizer)
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    console.print(f"\n[bold blue]Loading model:[/bold blue] [cyan]{model_name}[/cyan]")
    console.print(f"[bold blue]Using dtype:[/bold blue] [cyan]{torch_dtype}[/cyan]")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        console.print(f"[yellow]⚠[/yellow] Set pad_token to eos_token: [italic]{tokenizer.eos_token}[/italic]")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )

    # 모델 정보를 테이블로 표시
    table = Table(title="Model Information", show_header=False, box=None)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Status", "✓ Loaded successfully")
    table.add_row("Vocab size", f"{len(tokenizer):,}")
    table.add_row("Model dtype", str(model.dtype))

    return model, tokenizer


def apply_peft(model, peft_config_dict):
    """
    모델에 LoRA를 적용하여 파라미터 효율적인 학습을 수행하는 함수.

    Args:
        model: 베이스 모델
        peft_config_dict (dict): PEFT 설정 딕셔너리

    Returns:
        PEFT가 적용된 모델
    """
    peft_config = LoraConfig(
        r=peft_config_dict['r'],
        lora_alpha=peft_config_dict['lora_alpha'],
        lora_dropout=peft_config_dict['lora_dropout'],
        target_modules=peft_config_dict['target_modules'],
        bias=peft_config_dict['bias'],
        task_type=peft_config_dict['task_type'],
    )

    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / all_params

    # PEFT 정보를 패널로 표시
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold green")
    
    table.add_row("Trainable params", f"{trainable_params:,}")
    table.add_row("All params", f"{all_params:,}")
    table.add_row("Trainable %", f"{trainable_pct:.2f}%")

    return model


def create_dapt_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    training_config: dict,
):
    """
    DAPT 학습을 위한 Trainer를 생성하는 함수.

    Args:
        model: 학습할 모델
        tokenizer: 토크나이저
        train_dataset: 학습 데이터셋
        eval_dataset: 검증 데이터셋
        output_dir (str): 출력물 저장 디렉토리
        training_config (dict): 학습 설정 딕셔너리

    Returns:
        Trainer: HuggingFace Trainer 인스턴스
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config['num_epochs'],
        learning_rate=training_config['learning_rate'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        weight_decay=training_config['weight_decay'],
        logging_steps=training_config['logging_steps'],
        save_strategy=training_config['save_strategy'],
        evaluation_strategy=training_config['evaluation_strategy'],
        save_total_limit=training_config['save_total_limit'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        fp16=training_config['fp16'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        gradient_checkpointing=training_config['gradient_checkpointing'],
        load_best_model_at_end=training_config.get('load_best_model_at_end', True),
        metric_for_best_model=training_config.get('metric_for_best_model', 'loss'),
        greater_is_better=training_config.get('greater_is_better', False),
        report_to=["tensorboard"],
        logging_dir=f"{output_dir}/logs",
        save_safetensors=True,
    )

    # 학습 설정을 테이블로 표시
    table = Table(title="Training Configuration", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Epochs", str(training_config['num_epochs']))
    table.add_row("Learning Rate", f"{training_config['learning_rate']:.2e}")
    table.add_row("Train Batch Size", str(training_config['per_device_train_batch_size']))
    table.add_row("Eval Batch Size", str(training_config['per_device_eval_batch_size']))
    table.add_row("Gradient Accumulation", str(training_config['gradient_accumulation_steps']))
    table.add_row("FP16", "✓" if training_config['fp16'] else "✗")
    table.add_row("Gradient Checkpointing", "✓" if training_config['gradient_checkpointing'] else "✗")
    
    console.print(table)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    console.print("[green]✓[/green] Trainer initialized successfully")

    return trainer
