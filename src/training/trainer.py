from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig


def get_data_collator(tokenizer, response_template: str = "<start_of_turn>model"):
    """
    Get data collator for completion only LM
    """
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    return data_collator


def get_sft_config(
    output_dir: str = "outputs_gemma",
    max_seq_length: int = 1024,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    logging_steps: int = 1,
    save_strategy: str = "epoch",
    evaluation_strategy: str = "epoch",
    save_total_limit: int = 2,
    lr_scheduler_type: str = "cosine",
    fp16: bool = False,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
) -> SFTConfig:
    """
    Get SFT configuration
    """
    sft_config = SFTConfig(
        do_train=True,
        do_eval=True,
        lr_scheduler_type=lr_scheduler_type,
        max_seq_length=max_seq_length,
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy,
        save_total_limit=save_total_limit,
        fp16=fp16,  # V100에서 float16 mixed precision
        gradient_accumulation_steps=gradient_accumulation_steps,  # 메모리 효율성
        gradient_checkpointing=gradient_checkpointing,  # 메모리 절약
        save_only_model=True,
        report_to="none",
    )
    return sft_config


def get_trainer(
    model,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
    compute_metrics,
    preprocess_logits_for_metrics,
    peft_config,
    sft_config,
) -> SFTTrainer:
    """
    Get SFT Trainer
    """
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        peft_config=peft_config,
        args=sft_config,
    )
    return trainer
