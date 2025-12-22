"""
Korean SAT Solver - Inference Script
"""
from src.data import load_test_data, process_test_dataset
from src.model import load_model_for_inference
from src.inference import run_inference, save_predictions


def main():
    # =====================
    # Configuration
    # =====================
    CHECKPOINT_PATH = "outputs_gemma/checkpoint-4491"  # TODO: 학습된 Checkpoint 경로 입력
    TEST_DATA_PATH = "test.csv"  # TODO: Test Data 경로 입력
    OUTPUT_PATH = "output.csv"

    # =====================
    # Load model
    # =====================
    print("Loading model from checkpoint...")
    model, tokenizer = load_model_for_inference(CHECKPOINT_PATH)

    # =====================
    # Load and preprocess test data
    # =====================
    print("Loading test data...")
    test_df = load_test_data(TEST_DATA_PATH)
    print(f"Test dataset size: {len(test_df)}")

    # Process test dataset
    print("Processing test dataset...")
    test_dataset = process_test_dataset(test_df)

    # =====================
    # Run inference
    # =====================
    print("Running inference...")
    infer_results = run_inference(model, tokenizer, test_dataset)

    # =====================
    # Save results
    # =====================
    print(f"Saving predictions to {OUTPUT_PATH}...")
    save_predictions(infer_results, OUTPUT_PATH)
    print(f"Inference completed! Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
