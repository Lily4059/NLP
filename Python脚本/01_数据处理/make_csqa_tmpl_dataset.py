import json
from pathlib import Path


ROOT = Path(r"d:\deadline520\LLaMA-Factory\data")


TMPL_INSTRUCTION = (
    "Read the multiple-choice question carefully and answer using exactly one "
    "sentence in this format: The correct answer is X. Replace X with one "
    "uppercase letter from A, B, C, D, E. State the answer only once. Do not "
    "output any extra explanation."
)

TMPL_SUFFIX = (
    "\n\nOutput requirement: respond with exactly one sentence in this format: "
    "The correct answer is X. State the answer only once."
)


def transform_dataset(src_name: str, dst_name: str) -> None:
    src_path = ROOT / src_name
    dst_path = ROOT / dst_name

    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tmpl_data = []
    for item in data:
        answer = item["output"].strip().upper()
        tmpl_data.append(
            {
                "instruction": TMPL_INSTRUCTION,
                "input": f'{item["input"]}{TMPL_SUFFIX}',
                "output": f"The correct answer is {answer}.",
            }
        )

    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(tmpl_data, f, ensure_ascii=False, indent=2)

    print(f"saved: {dst_path} ({len(tmpl_data)} samples)")


def main() -> None:
    transform_dataset("csqa_train.json", "csqa_train_tmpl.json")
    transform_dataset("csqa_valid.json", "csqa_valid_tmpl.json")


if __name__ == "__main__":
    main()
