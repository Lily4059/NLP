import json
from pathlib import Path


ROOT = Path(r"d:\deadline520\LLaMA-Factory\data")


STRICT_INSTRUCTION = (
    "Read the multiple-choice question carefully and answer with exactly one "
    "uppercase letter from A, B, C, D, E. Do not output any word, explanation, "
    "punctuation, or extra text."
)

STRICT_SUFFIX = (
    "\n\nOutput requirement: return exactly one uppercase letter among A, B, C, D, E."
)


def transform_dataset(src_name: str, dst_name: str) -> None:
    src_path = ROOT / src_name
    dst_path = ROOT / dst_name

    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    strict_data = []
    for item in data:
        strict_data.append(
            {
                "instruction": STRICT_INSTRUCTION,
                "input": f'{item["input"]}{STRICT_SUFFIX}',
                "output": item["output"].strip().upper(),
            }
        )

    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(strict_data, f, ensure_ascii=False, indent=2)

    print(f"saved: {dst_path} ({len(strict_data)} samples)")


def main() -> None:
    transform_dataset("csqa_train.json", "csqa_train_strict.json")
    transform_dataset("csqa_valid.json", "csqa_valid_strict.json")


if __name__ == "__main__":
    main()
