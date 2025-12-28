import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_gen.data_augmentation import main


@hydra.main(version_base=None, config_path="conf", config_name="config")
def generate_data(cfg: DictConfig):
    """Hydra로 설정을 관리하는 데이터 생성 메인 함수"""

    # 설정 출력
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # data_augmentation의 main 함수 호출
    main(
        newspaper_path=cfg.get("newspaper_path"),
        book_path=cfg.get("book_path"),
        output_csv=cfg.output_csv,
        problems_per_article=cfg.problems_per_article,
        seed=cfg.seed,
        data_type=cfg.data_type,
        num_newspaper_problems=cfg.get("num_newspaper_problems"),
        num_book_problems=cfg.get("num_book_problems"),
        batch_size=cfg.get("api_batch_size", 200),
    )


if __name__ == "__main__":
    generate_data()
