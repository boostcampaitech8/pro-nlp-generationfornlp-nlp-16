import pandas as pd
import asyncio
from openai import AsyncOpenAI
import ast
import time
import os
import sys
import json
from collections import Counter
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from src.inference.exaone_prompts import (
    SYSTEM_PROMPT_BASIC,
    SYSTEM_PROMPT_MOA,
    SYSTEM_PROMPT_MOA_NON_REASONING,
    create_user_prompt_basic,
    create_user_prompt_with_descriptions,
)
from src.data.description import load_descriptions_json

console = Console()


# Async inference functions
async def get_inference(client, system_msg, user_content, seed):
    """
    Inference 실행하고 content와 reasoning_content를 모두 반환
    
    Returns:
        tuple: (content, reasoning_content)
    """
    try:
        response = await client.chat.completions.create(
            model="EXAONE-4.0-32B",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ],
            max_tokens=8192,  # Reduced to encourage concise answers
            temperature=0.7,  # Reduced for more focused responses
            top_p=0.95,
            seed=seed,
        )
        msg = response.choices[0].message
        
        # content와 reasoning_content 추출
        content = msg.content
        reasoning_content = getattr(msg, 'reasoning_content', None)  # reasoning_content가 없을 수 있음

        # Check if response was cut off
        if response.choices[0].finish_reason == "length":
            console.print(f"[yellow]⚠ Warning:[/yellow] Response {seed} was truncated due to max_tokens limit")

        return content, reasoning_content

    except Exception as e:
        console.print(f"[red]✗ Inference {seed} Error:[/red] {e}")
        return "Error", None


async def process_row(client, descriptions_dict, index, row, seed):
    """
    3가지 모드로 추론:
    1. EXAONE만 (reasoning + descriptions 없이)
    2. EXAONE + descriptions (reasoning + SKT A.X descriptions)
    3. EXAONE + descriptions (non-reasoning + SKT A.X descriptions)
    
    Returns:
        tuple: (index, results, reasoning_contents)
        results: [content0, content1, content2]
        reasoning_contents: [reasoning0, reasoning1, reasoning2]
    """
    problem = ast.literal_eval(row["problems"])
    sample_id = str(row["id"])
    desc_data = descriptions_dict.get(sample_id, {})

    tasks = []

    # 모드 1: EXAONE만 (reasoning + descriptions 없이)
    user_content_1 = create_user_prompt_basic(
        paragraph=row["paragraph"],
        question=problem["question"],
        choices=problem["choices"],
        question_plus=(
            row["question_plus"] if pd.notna(row["question_plus"]) else None
        ),
    )
    tasks.append(get_inference(client, SYSTEM_PROMPT_BASIC, user_content_1, seed))

    # 모드 2: EXAONE + descriptions (reasoning + SKT A.X descriptions)
    user_content_2 = create_user_prompt_with_descriptions(
        paragraph=row["paragraph"],
        question=problem["question"],
        choices=problem["choices"],
        description_1=desc_data.get("description_1", ""),
        description_2=desc_data.get("description_2", ""),
        question_plus=(
            row["question_plus"] if pd.notna(row["question_plus"]) else None
        ),
    )
    tasks.append(get_inference(client, SYSTEM_PROMPT_MOA, user_content_2, seed))

    # 모드 3: EXAONE + descriptions (non-reasoning + SKT A.X descriptions)
    user_content_3 = create_user_prompt_with_descriptions(
        paragraph=row["paragraph"],
        question=problem["question"],
        choices=problem["choices"],
        description_1=desc_data.get("description_1", ""),
        description_2=desc_data.get("description_2", ""),
        question_plus=(
            row["question_plus"] if pd.notna(row["question_plus"]) else None
        ),
    )
    tasks.append(get_inference(client, SYSTEM_PROMPT_MOA_NON_REASONING, user_content_3, seed))

    results = await asyncio.gather(*tasks)
    
    # results는 [(content, reasoning_content), ...] 형태
    contents = [r[0] if isinstance(r, tuple) else r for r in results]
    reasoning_contents = [r[1] if isinstance(r, tuple) and len(r) > 1 else None for r in results]

    return index, contents, reasoning_contents


async def main(df_test, descriptions_dict, client):
    console.print(Panel.fit(
        "[bold green]Starting Inference with EXAONE-4.0-32B[/bold green]",
        border_style="green"
    ))

    # Create output directory if it doesn't exist
    output_dir = "outputs/moa"
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "TestSet_Inference_EXAONE-4.0-32B.csv")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        for s in range(0, 1):
            task = progress.add_task(f"[cyan]Processing loop s={s}", total=len(df_test))

            for i in range(len(df_test)):
                idx, results, reasoning_contents = await process_row(client, descriptions_dict, i, df_test.loc[i], s)

                # 3가지 모드 결과 저장
                # resp_0: EXAONE만 (reasoning + descriptions 없이)
                # resp_1: EXAONE + descriptions (reasoning)
                # resp_2: EXAONE + descriptions (non-reasoning)
                for r, output in enumerate(results):
                    df_test.loc[idx, f"resp_{r}_{s}"] = output
                
                # reasoning_content 저장
                for r, reasoning in enumerate(reasoning_contents):
                    if reasoning is not None:
                        df_test.loc[idx, f"reasoning_{r}_{s}"] = reasoning

                progress.update(task, advance=1, description=f"[cyan]Processing s={s}, sample {i+1}/{len(df_test)}")

                if i % 5 == 4:
                    df_test.to_csv(output_csv_path, index=False)

            df_test.to_csv(output_csv_path, index=False)

    console.print(f"\n[green]✓[/green] Inference 완료: [italic]{output_csv_path}[/italic]")

    # Create submission.csv with answer extraction
    console.print("\n")
    console.print(Panel.fit(
        "[bold blue]Creating Submission File[/bold blue]",
        border_style="blue"
    ))
    submission_data = []
    detailed_results = []  # 각 모드별 상세 결과 저장용

    for i in range(len(df_test)):
        list_choice = []
        mode_answers = {}  # 각 모드별 정답 저장
        mode_reasonings = {}  # 각 모드별 reasoning_content 저장

        # 3가지 모드의 결과를 모두 수집
        mode_names = [
            "EXAONE_only_reasoning",
            "EXAONE_with_descriptions_reasoning",
            "EXAONE_with_descriptions_non_reasoning"
        ]
        
        for r in range(3):  # 3가지 모드: 0, 1, 2
            mode_answer = None
            mode_reasoning = None
            for s in range(1):  # Only one iteration as we only run once
                try:
                    choice = (
                        df_test.loc[i, f"resp_{r}_{s}"]
                        .split('{"정답": "')[-1]
                        .split('"}')[0]
                    )
                    if choice in ["1", "2", "3", "4", "5"]:
                        list_choice.append(choice)
                        mode_answer = choice
                except:
                    pass
                
                # reasoning_content 추출
                try:
                    reasoning_col = f"reasoning_{r}_{s}"
                    if reasoning_col in df_test.columns:
                        mode_reasoning = df_test.loc[i, reasoning_col]
                except:
                    pass
            
            # 각 모드별 정답 저장 (추출 실패 시 None)
            mode_answers[mode_names[r]] = mode_answer
            # 각 모드별 reasoning_content 저장
            mode_reasonings[mode_names[r]] = mode_reasoning if mode_reasoning is not None and str(mode_reasoning) != "nan" else None

        # 다수결로 최종 정답 선택
        list_choice.sort()
        count_choices = Counter(list_choice)
        top_choices = count_choices.most_common(2)

        try:
            # 1:1:1 동점인 경우 (모든 모드가 다른 답변) mode_1 우선
            if len(count_choices) == 3 and all(count == 1 for count in count_choices.values()):
                # 1:1:1 동점인 경우 mode_1의 답변을 우선시
                answer = mode_answers["EXAONE_with_descriptions_reasoning"] or top_choices[0][0]
            # 1:1 동점인 경우 (2가지 답변이 각각 1표씩) mode_1 우선
            elif len(top_choices) >= 2 and top_choices[0][1] == top_choices[1][1] and top_choices[0][1] == 1:
                # 동점인 경우 mode_1의 답변을 우선시
                if mode_answers["EXAONE_with_descriptions_reasoning"] in [top_choices[0][0], top_choices[1][0]]:
                    answer = mode_answers["EXAONE_with_descriptions_reasoning"]
                else:
                    # mode_1의 답변이 동점 후보에 없으면 첫 번째 선택
                    answer = top_choices[0][0]
            else:
                # 동점이 아니면 가장 많이 선택된 답변
                answer = top_choices[0][0]
        except:
            answer = "1"

        # Create submission row with only id and answer
        submission_data.append({"id": df_test.loc[i, "id"], "answer": answer})
        
        # 상세 결과 저장
        detailed_results.append({
            "id": df_test.loc[i, "id"],
            "mode_0_EXAONE_only_reasoning": mode_answers["EXAONE_only_reasoning"],
            "mode_1_EXAONE_with_descriptions_reasoning": mode_answers["EXAONE_with_descriptions_reasoning"],
            "mode_2_EXAONE_with_descriptions_non_reasoning": mode_answers["EXAONE_with_descriptions_non_reasoning"],
            "mode_0_reasoning_content": mode_reasonings["EXAONE_only_reasoning"],
            "mode_1_reasoning_content": mode_reasonings["EXAONE_with_descriptions_reasoning"],
            "mode_2_reasoning_content": mode_reasonings["EXAONE_with_descriptions_non_reasoning"],
            "final_answer_majority_vote": answer,
            "vote_counts": dict(count_choices)
        })

    df_submission = pd.DataFrame(submission_data)
    
    # Create output directory if it doesn't exist
    output_dir = "outputs/moa"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save submission.csv
    df_submission.to_csv("outputs/moa/submission.csv", index=False)
    console.print(f"[green]✓[/green] Submission 파일 생성 완료: [italic]outputs/moa/submission.csv[/italic]")
    
    # Save detailed results as JSON
    detailed_json_path = "outputs/moa/detailed_results.json"
    with open(detailed_json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    console.print(f"[green]✓[/green] 상세 결과 JSON 파일 생성 완료: [italic]{detailed_json_path}[/italic]")

    console.print("\n")
    console.print(Panel.fit(
        "[bold green]EXAONE Inference Complete![/bold green]",
        border_style="green"
    ))


if __name__ == "__main__":
    DESCRIPTIONS_PATH = "data/descriptions.json"
    
    # 테스트 데이터 로드
    df_test = pd.read_csv("./data/test.csv")
    # 실사용시 주석처리! (디버그용)
    # df_test = df_test.head(5).reset_index(drop=True)
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]EXAONE Inference with MoA (3-Mode Ensemble)[/bold cyan]",
        border_style="cyan"
    ))
    
    # descriptions.json 로드
    if not os.path.exists(DESCRIPTIONS_PATH):
        console.print(f"[red]✗[/red] Description 파일을 찾을 수 없습니다: [italic]{DESCRIPTIONS_PATH}[/italic]")
        console.print("[red]파이프라인을 중단합니다.[/red]")
        sys.exit(1)
    
    descriptions_dict = load_descriptions_json(DESCRIPTIONS_PATH)
    
    console.print("[bold yellow]3가지 모드로 추론합니다:[/bold yellow]")
    console.print("  1. EXAONE만 (reasoning + descriptions 없이)")
    console.print("  2. EXAONE + descriptions (reasoning + SKT A.X descriptions)")
    console.print("  3. EXAONE + descriptions (non-reasoning + SKT A.X descriptions)")
    console.print("  → 다수결로 최종 정답 선택\n")
    
    # Initialize client
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="sk-no-key-required")
    
    time.sleep(5)
    asyncio.run(main(df_test, descriptions_dict, client))
    time.sleep(5)
