import os
import ast
import hydra
import pandas as pd
from omegaconf import DictConfig

from src.model import load_model_for_inferecne
from src.inference import run_inference, save_predictions_cot

def process_test_data_for_cot(test_df: pd.DataFrame) -> list:
    dataset = []

    for idx, row in test_df.iterrows():
        choices = row["choices"]
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        
        dataset.append({
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": row["question"],
            "choices": choices,
        })

    
    return dataset

def find_latest_checkpoint(original_cwd: str) -> str:

    outputs_roots = [
        os.path.join(original_cwd, "outputs"),
        os.path.join(original_cwd, "outputs", "train"),
    ]

    for outputs_root in outputs_roots:
        if not os.path.exists(outputs_root):
            continue

        dates = sorted([d for d in os.listdir(outputs_root)
                        if os.path.isdir(os.path.join(outputs_root,d))], reverse=True)

        for date in dates:
            date_dir = os.path.join(outputs_root, date)
            times = sorted([t for t in os.listdir(date_dir)
                            if os.path.isdir(os.path.join(date_dir, t))], reverse=True)

            for time in times:
                run_dir = os.path.join(date_dir, time)
                if any(d.startwith("checkpoint-") for d in os.listdir(run_dir)):
                    return run_dir

    raise ValueError("No checkpoint found in outputs/")

def get_best_checkpoint(checkpoint_path: str, checkpoint_step:str) -> str:

    if "checkpoint-" in os.path.basename(checkpoint_path):
        return checkpoint_path

    if not os.path.isdir(checkpoint_path):
        return checkpoint_path

    checkpoints = [d for d in os.listdir(checkpoint_path) if d.startswith("checkpoint-")]

    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_path}")

    if checkpoint_step == "best":
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        return os.path.join(checkpoint_path, checkpoints[-1])
    else:
        target = f"checkpoint-{checkpoint_step}"
        if target in checkpoints:
            return os.path.join(checkpoint_path, traget)
        raise ValueError(f"Checkpoint {target} not found")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=" * 60)
    print("CoT (Chain of Thought) Prompting Inference")
    print("=" * 60)
    
   
    original_cwd = hydra.utils.get_original_cwd()
    
    checkpoint_path = cfg.inference.checkpoint_dir
    checkpoint_step = cfg.inference.checkpoint_step
    
    if not checkpoint_path:
        print("Checkpoint path empty. Auto-detecting...")
        checkpoint_path = find_latest_checkpoint(original_cwd)
        print(f"Found: {checkpoint_path}")
    
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = hydra.utils.to_absolute_path(checkpoint_path)
    
    checkpoint_path = get_best_checkpoint(checkpoint_path, checkpoint_step)
    
    print(f"Using checkpoint: {checkpoint_path}")
    
   
    print("\nLoading model...")
    torch_dtype = cfg.inference.torch_dtype
    model, tokenizer = load_model_for_inference(checkpoint_path, torch_dtype=torch_dtype)
    

    test_path = hydra.utils.to_absolute_path(cfg.data.test_path)
    print(f"\nLoading test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    print(f"Test size: {len(test_df)}")
    
 
    test_dataset = process_test_data_for_cot(test_df)
    
   

    print("\nRunning CoT inference...")
    infer_results, reasoning_results = run_inference_cot(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        max_new_tokens=150, 
        temperature=0.3,     
        verbose=True,        
    )
    

    output_path = cfg.inference.output_file.replace(".csv", "_cot.csv")
    print(f"\nSaving results to {output_path}...")
    save_predictions_cot(infer_results, reasoning_results, output_path)
    
    print(f"\n{'='*60}")
    print("CoT Inference completed!")
    print(f"Results saved to: {os.getcwd()}/{output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()