import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.data_augmentation import main

if __name__ == "__main__":
    # Test with small number of problems
    main(
        file_path="data/data4gen/newspaper",
        output_csv="data/train_newspaper_augmented.csv",
        num_problems=10,  # Small batch for testing
        problems_per_article=1,
        seed=42,
    )
