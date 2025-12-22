"""
Korean SAT Solver - Training Script
"""
import pandas as pd
from datasets import Dataset

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
)
from src.training import (
    get_data_collator,
    get_sft_config,
    get_trainer,
    get_metrics_functions,
)


def main():
    # =====================
    # Configuration
    # =====================
    SEED = 42
    TRAIN_DATA_PATH = "train.csv"  # TODO: Train Data 경로 입력
    MODEL_NAME = "beomi/gemma-ko-2b"
    OUTPUT_DIR = "outputs_gemma"
    MAX_SEQ_LENGTH = 1024
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5

    # =====================
    # Set seed
    # =====================
    set_seed(SEED)

    # =====================
    # Load and preprocess data
    # =====================
    print("Loading training data...")
    df = load_train_data(TRAIN_DATA_PATH)
    df = add_full_question(df)

    print(f"Dataset size: {len(df)}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # Convert to HuggingFace Dataset
    dataset = dataframe_to_dataset(df)

    # Process dataset to chat format
    print("Processing dataset...")
    processed_dataset = process_train_dataset(dataset)
    processed_dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))

    # =====================
    # Load model and tokenizer
    # =====================
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # =====================
    # Tokenize dataset
    # =====================
    print("Tokenizing dataset...")
    train_dataset, eval_dataset = tokenize_dataset(
        processed_dataset,
        tokenizer,
        max_length=MAX_SEQ_LENGTH,
        test_size=0.1,
        seed=SEED,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Sample: {tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=True)}")

    # =====================
    # Setup training
    # =====================
    print("Setting up training...")
    tokenizer = setup_tokenizer_for_training(tokenizer)

    # Get configurations
    peft_config = get_peft_config()
    data_collator = get_data_collator(tokenizer)
    sft_config = get_sft_config(
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
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

    # =====================
    # Train
    # =====================
    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == "__main__":
    main()
