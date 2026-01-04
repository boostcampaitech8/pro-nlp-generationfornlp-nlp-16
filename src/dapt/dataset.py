import pandas as pd
from ast import literal_eval
from datasets import Dataset
from transformers import PreTrainedTokenizer
from rich.console import Console
from rich.progress import track

console = Console()

def load_dapt_data(file_path: str) -> pd.DataFrame:
    """
    AIHub 국어 지문형 문제 데이터를 DAPT(Domain-Adaptive Pretraining) 학습용으로
    로드하고 전처리하는 함수.

    DAPT 특성상 문제의 모든 구성 요소 지문, 문제, 선택지, 정답, 해설)를 항상 포함한다.

    Args:
        file_path (str): aihub_workbook_final.csv 파일 경로

    Returns:
        pd.DataFrame:
            - id: 원본 데이터의 고유 ID
            - text: DAPT 학습에 사용할 연속 텍스트
    """
    df = pd.read_csv(file_path)
    console.print(f"[green]✓[/green] Loaded [bold cyan]{len(df)}[/bold cyan] samples from [italic]{file_path}[/italic]")

    # 각 샘플(row)을 순회하면 DAPT용 텍스트로 합치기
    records = []
    for _, row in track(df.iterrows(), description="[yellow]Processing samples...[/yellow]", total=len(df)):
        problems = literal_eval(row['problems'])
        parts = []

        if pd.notna(row['paragraph']):
            parts.append(f"[지문]\n{row['paragraph'].strip()}")

        question = problems.get('question', '')
        if question:
            parts.append(f"\n[문제]\n{question}")

        choices = problems.get('choices', [])
        if choices:
            formatted_choices = '\n'.join([f"{i}. {c}" for i, c in enumerate(choices, 1)])
            parts.append(f"\n[선택지]\n{formatted_choices}")

        answer = problems.get('answer')
        if answer is not None:
            parts.append(f"\n[정답]\n{answer}")

        if pd.notna(row.get('description')):
            parts.append(f"\n[해설]\n{row['description'].strip()}")

        records.append({
            'id': row['id'],
            'text': '\n'.join(parts)
        })

    result_df = pd.DataFrame(records)
    console.print(f"[green]✓[/green] Preprocessing complete! Total: [bold cyan]{len(result_df)}[/bold cyan] samples")
    return result_df


def _tokenize_dapt(examples, tokenizer, max_length):
    """
    DAPT(Domain-Adaptive Pretraining)를 위한
    Causal Language Modeling용 토크나이징 함수.

    입력 텍스트를 토큰화한 뒤,
    labels를 input_ids와 동일하게 설정하여
    다음 토큰 예측(next-token prediction) 학습이 가능하도록 함.
    """
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized


def prepare_dapt_dataset(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    DAPT(Domain-Adaptive Pretraining) 학습을 위한 데이터셋을 준비하는 함수.

    load_dapt_data()로 생성된 DataFrame을 HuggingFace Dataset으로 변환한 뒤,
    토크나이징 및 train / validation 분할을 수행한다.

    Args:
        df (pd.DataFrame): "id", "text" 컬럼을 포함한 DataFrame
        tokenizer (PreTrainedTokenizer): 사용할 토크나이저
        max_length (int): 최대 시퀀스 길이
        test_size (float): 검증 데이터 비율 (0.0이면 검증셋 생성 안 함)
        seed (int): 데이터 분할을 위한 랜덤 시드

    Returns:
        tuple:
            - train_dataset: 학습용 Dataset
            - eval_dataset: 검증용 Dataset (test_size=0이면 None)
    """

    dataset = Dataset.from_pandas(df)

    console.print(f"[yellow]Tokenizing {len(dataset)} samples...[/yellow]")
    tokenized_dataset = dataset.map(
        lambda examples: _tokenize_dapt(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    if test_size > 0:
        split_dataset = tokenized_dataset.train_test_split(
            test_size=test_size,
            seed=seed,
        )
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        console.print(f"[green]✓[/green] Train dataset: [bold cyan]{len(train_dataset)}[/bold cyan] samples")
        console.print(f"[green]✓[/green] Eval dataset: [bold cyan]{len(eval_dataset)}[/bold cyan] samples")
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
        console.print(f"[green]✓[/green] Train dataset: [bold cyan]{len(train_dataset)}[/bold cyan] samples")

    return train_dataset, eval_dataset
