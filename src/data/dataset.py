import pandas as pd
from ast import literal_eval
from datasets import Dataset


def load_train_data(train_path: str) -> pd.DataFrame:
    """
    Load the train dataset
    [수정] reasoning 컬럼 추가
    """
    dataset = pd.read_csv(train_path)

    # Flatten the JSON dataset
    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer', None),
            "question_plus": problems.get('question_plus', None),
            "reasoning": row.get('reasoning', None),  # [추가] reasoning 컬럼
        }
        # Include 'question_plus' if it exists in problems
        if 'question_plus' in problems:
            record['question_plus'] = problems['question_plus']
        # Include 'question_plus' if it exists in row (원본 CSV)
        if 'question_plus' in row and pd.notna(row['question_plus']):
            record['question_plus'] = row['question_plus']
        records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(records)
    return df


def load_test_data(test_path: str) -> pd.DataFrame:
    """
    Load the test dataset
    """
    test_df = pd.read_csv(test_path)

    # Flatten the JSON dataset
    records = []
    for _, row in test_df.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer', None),
            "question_plus": problems.get('question_plus', None),
        }
        # Include 'question_plus' if it exists
        if 'question_plus' in problems:
            record['question_plus'] = problems['question_plus']
        records.append(record)

    # Convert to DataFrame
    test_df = pd.DataFrame(records)
    return test_df


def add_full_question(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine 'question' and 'question_plus' if available
    """
    df['question_plus'] = df['question_plus'].fillna('')
    df['full_question'] = df.apply(
        lambda x: x['question'] + ' ' + x['question_plus'] if x['question_plus'] else x['question'],
        axis=1
    )
    # Calculate the length of each question
    df['question_length'] = df['full_question'].apply(len)
    return df


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    """
    Convert DataFrame to HuggingFace Dataset
    """
    return Dataset.from_pandas(df)
