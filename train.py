"""
Korean SAT Solver - Training Script
"""
import pandas as pd
from datasets import Dataset
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from src.utils import set_seed
from src.data import (
    load_train_data,
    add_full_question,
    dataframe_to_dataset,
    process_train_dataset,
    tokenize_dataset,
)
from src.model import (
    load_model_and_tokenizer,
    get_peft_config,
    setup_tokenizer_for_training,
    get_response_template,
)
from src.training import (
    get_data_collator,
    get_sft_config,
    get_trainer,
    get_metrics_functions,
)

# After print -> logging으로 변경 Plz
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 로깅용
    print(OmegaConf.to_yaml(cfg))
    
    # 현재 실행 중인 실험의 저장 경로 가져오기
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Saving outputs to: {output_dir}")

    # =====================
    # Set seed
    set_seed(cfg.seed)

    print("Loading training data...")
    # Hydra changes cwd, so we need absolute path
    train_data_path = hydra.utils.to_absolute_path(cfg.data.train_path)
    df = load_train_data(train_data_path)
    df = add_full_question(df)

    print(f"Dataset size: {len(df)}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # Convert to HuggingFace Dataset
    dataset = dataframe_to_dataset(df)

    # Process dataset to chat format
    print("Processing dataset...")
    processed_dataset = process_train_dataset(dataset)
    processed_dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    torch_dtype = cfg.model.get('torch_dtype', 'float16')
    quant_config = OmegaConf.to_container(cfg.model.quantization, resolve=True) \
        if hasattr(cfg.model, 'quantization') else None
    
    model, tokenizer = load_model_and_tokenizer(
        cfg.model.name, 
        torch_dtype=torch_dtype,
        quantization_config=quant_config
    )

    # Tokenize dataset
    print("Tokenizing dataset...")
    train_dataset, eval_dataset = tokenize_dataset(
        processed_dataset,
        tokenizer,
        max_length=cfg.data.max_length,
        test_size=cfg.data.test_size,
        seed=cfg.seed,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Sample: {tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=True)}")

    # Setup training
    tokenizer = setup_tokenizer_for_training(tokenizer)

    # Get configurations
    peft_config = get_peft_config(
        r=cfg.model.peft.r,
        lora_alpha=cfg.model.peft.lora_alpha,
        lora_dropout=cfg.model.peft.lora_dropout,
        target_modules=OmegaConf.to_container(cfg.model.peft.target_modules, resolve=True),
        bias=cfg.model.peft.bias,
        task_type=cfg.model.peft.task_type
    )

    if hasattr(cfg.model, 'response_template') and cfg.model.response_template:
        response_template = cfg.model.response_template
        print(f"✓ Config에서 지정된 response template 사용: '{response_template}'")
    else:
        response_template = get_response_template(cfg.model.name, tokenizer)
        print(f"⚠ 자동 추출된 response template 사용: '{response_template}'")
    
    data_collator = get_data_collator(tokenizer, response_template=response_template)
    
    sft_config = get_sft_config(
        output_dir=output_dir,
        max_seq_length=cfg.data.max_length,
        num_train_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        weight_decay=cfg.training.weight_decay,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        evaluation_strategy=cfg.training.evaluation_strategy,
        save_total_limit=cfg.training.save_total_limit,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        load_best_model_at_end=cfg.training.load_best_model_at_end,       
        metric_for_best_model=cfg.training.metric_for_best_model,         
        greater_is_better=cfg.training.greater_is_better,
        fp16=cfg.training.fp16,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
    )

    # Get metrics functions
    preprocess_logits_for_metrics, compute_metrics = get_metrics_functions(tokenizer)

    # Get trainer
    trainer = get_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        peft_config=peft_config,
        sft_config=sft_config,
    )

    #적용된 Trainer 확인용
    print("\n" + "="*30)
    print(f"🚀 적용된 Trainer Class: {type(trainer).__name__}")
    print("="*30 + "\n")

    # Train
    trainer.train()
    
    # Save final model explicitly to the hydra output dir (although Trainer saves checkpoints there)
    # trainer.save_model(output_dir) # Optional, but Trainer handles checkpoints

    # =====================
    # Best metrics logging
    # =====================
    print("\n" + "=" * 50)
    print("!!!학습 완료!!!")
    print("=" * 50)
    
    if trainer.state.best_metric is not None:
        print(f"Best {cfg.training.metric_for_best_model}: {trainer.state.best_metric:.4f}")
        print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    
    if trainer.state.log_history:
        eval_logs = [log for log in trainer.state.log_history if 'eval_f1' in log]
        if eval_logs:
            best_f1_log = max(eval_logs, key=lambda x: x.get('eval_f1', 0))
            best_acc_log = max(eval_logs, key=lambda x: x.get('eval_accuracy', 0))
            
            print(f"\nBest F1: {best_f1_log['eval_f1']:.4f} (epoch {best_f1_log['epoch']})")
            print(f"Best Accuracy: {best_acc_log['eval_accuracy']:.4f} (epoch {best_acc_log['epoch']})")
    
    trainer.save_state()
    print("=" * 50)


if __name__ == "__main__":
    main()
