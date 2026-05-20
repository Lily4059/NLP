import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(r"d:\deadline520")
RAW_DATA_DIR = BASE_DIR / "dataset"
LLF_DATA_DIR = BASE_DIR / "LLaMA-Factory" / "data"

INSTRUCTION = "Read the multiple-choice question and output only the correct option letter."


def _build_sample(row: pd.Series) -> dict:
    choices = row["choices"]
    labels = choices["label"]
    texts = choices["text"]

    option_lines = [f"{label}. {text}" for label, text in zip(labels, texts)]
    input_text = "Question: " + row["question"] + "\n" + "\n".join(option_lines)

    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": row["answerKey"],
    }


def convert_split(parquet_name: str, output_name: str) -> None:
    df = pd.read_parquet(RAW_DATA_DIR / parquet_name)
    records = []

    for _, row in df.iterrows():
        answer = row.get("answerKey")
        if pd.isna(answer):
            continue
        records.append(_build_sample(row))

    output_path = LLF_DATA_DIR / output_name
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"saved: {output_path} ({len(records)} samples)")


def main() -> None:
    LLF_DATA_DIR.mkdir(parents=True, exist_ok=True)
    convert_split("train-00000-of-00001.parquet", "csqa_train.json")
    convert_split("validation-00000-of-00001.parquet", "csqa_valid.json")


if __name__ == "__main__":
    main()
