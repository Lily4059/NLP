import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


CHOICE_PATTERN = re.compile(r"\b([A-E])\b")
ANSWER_PATTERNS = [
    re.compile(r"(?:CORRECT ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*[\(\[]?\s*([A-E])\b"),
    re.compile(r"^[\s\"'`]*[\(\[]?\s*([A-E])\s*[\)\].,:;!?\s\"'`]*$"),
]


def load_data(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(tokenizer, instruction: str, input_text: str) -> str:
    output_rule = "Output exactly one uppercase letter from A, B, C, D, E. Do not output any other words."
    user_text = f"{instruction}\n{output_rule}\n\n{input_text}"
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_choice(text: str) -> str:
    upper_text = text.upper().strip()

    for pattern in ANSWER_PATTERNS:
        match = pattern.search(upper_text)
        if match:
            return match.group(1)

    matches = CHOICE_PATTERN.findall(upper_text)
    return matches[-1] if matches else ""


def load_model(base_model_path: str, adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)

    model.eval()
    return tokenizer, model


@torch.inference_mode()
def predict_one(tokenizer, model, sample: dict, max_new_tokens: int) -> tuple[str, str]:
    prompt = build_prompt(tokenizer, sample["instruction"], sample["input"])
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    pred = extract_choice(text)
    return pred, text


def evaluate(args):
    data = load_data(Path(args.data_path))
    if args.max_samples is not None:
        data = data[: args.max_samples]

    tokenizer, model = load_model(args.base_model_path, args.adapter_path)

    results = []
    correct = 0

    for idx, sample in enumerate(data, start=1):
        pred, raw_output = predict_one(tokenizer, model, sample, args.max_new_tokens)
        gold = sample.get("output", "")
        is_correct = pred == gold
        correct += int(is_correct)

        results.append(
            {
                "id": idx,
                "input": sample["input"],
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "raw_output": raw_output,
            }
        )

        if idx % 50 == 0:
            print(f"processed {idx}/{len(data)}")

    accuracy = correct / len(data) if data else 0.0
    summary = {
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "data_path": args.data_path,
        "samples": len(data),
        "correct": correct,
        "accuracy": accuracy,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base model or LoRA adapter on CSQA-style json data.")
    parser.add_argument("--base_model_path", required=True, help="Path to the local base model.")
    parser.add_argument("--adapter_path", default=None, help="Path to the LoRA adapter directory.")
    parser.add_argument("--data_path", required=True, help="Path to evaluation json file.")
    parser.add_argument("--output_path", required=True, help="Path to save evaluation results.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on evaluation samples.")
    parser.add_argument("--max_new_tokens", type=int, default=16, help="Max generated tokens per sample.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
