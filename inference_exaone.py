import pandas as pd
import asyncio
from openai import AsyncOpenAI
import ast
import time
import os
import json
from collections import Counter
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from src.inference.exaone_prompts import (
    SYSTEM_PROMPT_BASIC,
    SYSTEM_PROMPT_MOA,
    create_user_prompt_basic,
    create_user_prompt_with_descriptions,
)

console = Console()

# Load test data
df_test = pd.read_csv("./data/test.csv")


# # TEST MODE: Use only first 10 rows (comment out for full run)
# df_test = df_test.head(10).reset_index(drop=True)

# Load descriptions if available
descriptions_dict = {}
descriptions_path = "descriptions.json"

console.print("\n")
console.print(Panel.fit(
    "[bold cyan]EXAONE Inference with MoA[/bold cyan]",
    border_style="cyan"
))

if os.path.exists(descriptions_path):
    console.print(f"[green]✓[/green] Description 파일을 찾았습니다: [italic]{descriptions_path}[/italic]")
    with open(descriptions_path, "r", encoding="utf-8") as f:
        descriptions_list = json.load(f)

    # id를 키로 하는 딕셔너리로 변환
    for item in descriptions_list:
        sample_id = str(item["id"])
        descriptions_dict[sample_id] = item

    console.print(f"[green]✓[/green] {len(descriptions_dict)}개의 description을 로드했습니다.")
    console.print("[bold yellow]MoA 모드로 실행됩니다.[/bold yellow]\n")
    use_descriptions = True
    system_msg = SYSTEM_PROMPT_MOA
else:
    console.print("[yellow]⚠[/yellow] Description 파일이 없습니다.")
    console.print("[bold yellow]기본 모드로 실행합니다.[/bold yellow]\n")
    use_descriptions = False
    system_msg = SYSTEM_PROMPT_BASIC

# Initialize client
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="sk-no-key-required")

# list_system_msg = [system_msg_0, system_msg_3, system_msg_4]
list_system_msg = [system_msg]


# Async inference functions
async def get_inference(system_msg, user_content, seed):
    try:
        response = await client.chat.completions.create(
            model="EXAONE-4.0-32B",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ],
            max_tokens=4096,  # Reduced to encourage concise answers
            temperature=0.7,  # Reduced for more focused responses
            top_p=0.95,
            seed=seed,
        )
        msg = response.choices[0].message
        # Only use content, skip reasoning_content
        ret = msg.content

        # Check if response was cut off
        if response.choices[0].finish_reason == "length":
            console.print(f"[yellow]⚠ Warning:[/yellow] Response {seed} was truncated due to max_tokens limit")

        return ret

    except Exception as e:
        console.print(f"[red]✗ Inference {seed} Error:[/red] {e}")
        return "Error"


async def process_row(index, row, seed):
    problem = ast.literal_eval(row["problems"])
    sample_id = str(row["id"])

    # description 유무에 따라 프롬프트 생성 분기
    if use_descriptions and sample_id in descriptions_dict:
        # MoA 스타일 프롬프트 (description 포함)
        desc_data = descriptions_dict[sample_id]
        user_content = create_user_prompt_with_descriptions(
            paragraph=row["paragraph"],
            question=problem["question"],
            choices=problem["choices"],
            description_1=desc_data.get("description_1", ""),
            description_2=desc_data.get("description_2", ""),
            question_plus=(
                row["question_plus"] if pd.notna(row["question_plus"]) else None
            ),
        )
    else:
        # 기본 프롬프트 (description 없이)
        user_content = create_user_prompt_basic(
            paragraph=row["paragraph"],
            question=problem["question"],
            choices=problem["choices"],
            question_plus=(
                row["question_plus"] if pd.notna(row["question_plus"]) else None
            ),
        )

    tasks = []
    for r in range(len(list_system_msg)):
        tasks.append(get_inference(list_system_msg[r], user_content, seed))

    results = await asyncio.gather(*tasks)

    return index, results


async def main():
    console.print(Panel.fit(
        "[bold green]Starting Inference with EXAONE-4.0-32B[/bold green]",
        border_style="green"
    ))

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
                idx, results = await process_row(i, df_test.loc[i], s)

                for r, output in enumerate(results):
                    df_test.loc[idx, f"resp_{r}_{s}"] = output

                progress.update(task, advance=1, description=f"[cyan]Processing s={s}, sample {i+1}/{len(df_test)}")

                if i % 5 == 4:
                    df_test.to_csv("TestSet_Inference_EXAONE-4.0-32B.csv", index=False)

            df_test.to_csv("TestSet_Inference_EXAONE-4.0-32B.csv", index=False)

    console.print(f"\n[green]✓[/green] Inference 완료: [italic]TestSet_Inference_EXAONE-4.0-32B.csv[/italic]")

    # Create submission.csv with answer extraction
    console.print("\n")
    console.print(Panel.fit(
        "[bold blue]Creating Submission File[/bold blue]",
        border_style="blue"
    ))
    submission_data = []

    for i in range(len(df_test)):
        list_choice = []

        for r in range(len(list_system_msg)):
            for s in range(1):  # Only one iteration as we only run once
                try:
                    choice = (
                        df_test.loc[i, f"resp_{r}_{s}"]
                        .split('{"정답": "')[-1]
                        .split('"}')[0]
                    )
                    if choice in ["1", "2", "3", "4", "5"]:
                        list_choice.append(choice)
                except:
                    pass

        list_choice.sort()
        count_choices = Counter(list_choice)
        top_choices = count_choices.most_common(2)

        try:
            answer = top_choices[0][0]
        except:
            answer = "1"

        # Create submission row with only id and answer
        submission_data.append({"id": df_test.loc[i, "id"], "answer": answer})

    df_submission = pd.DataFrame(submission_data)
    
    # Create output directory if it doesn't exist
    output_dir = "outputs/moa"
    os.makedirs(output_dir, exist_ok=True)
    
    df_submission.to_csv("outputs/moa/submission.csv", index=False)
    console.print(f"[green]✓[/green] Submission 파일 생성 완료: [italic]outputs/moa/submission.csv[/italic]")

    console.print("\n")
    console.print(Panel.fit(
        "[bold green]EXAONE Inference Complete![/bold green]",
        border_style="green"
    ))


if __name__ == "__main__":
    time.sleep(5)
    asyncio.run(main())
    time.sleep(5)
