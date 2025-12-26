import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_gen.data_augmentation import main

if __name__ == "__main__":
    # 신문 데이터로 테스트
    # main(
    #     file_path="data/data4gen/newspaper",
    #     output_csv="data/train_newspaper_augmented.csv",
    #     num_problems=10,
    #     problems_per_article=1,
    #     seed=50,
    #     data_type="newspaper",
    # )

    # 도서 데이터로 테스트
    # main(
    #     file_path="data/data4gen/written",
    #     output_csv="data/train_book_augmented.csv",
    #     num_problems=10,
    #     problems_per_article=1,
    #     seed=50,
    #     data_type="book",
    # )

    # 신문 + 도서 혼합 데이터로 테스트
    main(
        output_csv="data/train_combined_augmented.csv",
        problems_per_article=1,
        seed=50,
        data_type="both",
        newspaper_path="data/data4gen/newspaper",
        book_path="data/data4gen/written",
        num_newspaper_problems=5,
        num_book_problems=5,
    )
