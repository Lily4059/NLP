import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


CHOICES = ["A", "B", "C", "D", "E"]
STRICT_ANSWER_PATTERN = re.compile(
    r"^\s*THE CORRECT ANSWER IS\s*([A-E])\.\s*$",
    re.IGNORECASE,
)


def load_data(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(tokenizer, instruction: str, input_text: str) -> str:
    output_rule = (
        "Respond with exactly one sentence in this format: "
        "The correct answer is X. Replace X with one uppercase letter from A, B, C, D, E. "
        "State the answer only once. Do not output any extra explanation."
    )
    user_text = f"{instruction}\n{output_rule}\n\n{input_text}"
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_choice(text: str) -> str:
    match = STRICT_ANSWER_PATTERN.match(text.strip())
    if match:
        return match.group(1).upper()
    return ""


def extract_gold_choice(text: str) -> str:
    match = STRICT_ANSWER_PATTERN.match(text.strip())
    if match:
        return match.group(1).upper()
    return ""


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


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(results: list[dict]) -> dict:
    labels = CHOICES
    confusion = {gold: {pred: 0 for pred in labels + ["EMPTY", "OTHER"]} for gold in labels}
    gold_counter = Counter()
    pred_counter = Counter()
    correct = 0
    valid_choice_predictions = 0

    for item in results:
        gold = item["gold"]
        pred = item["pred"] if item["pred"] in labels else ("EMPTY" if item["pred"] == "" else "OTHER")
        gold_counter[gold] += 1
        pred_counter[pred] += 1
        confusion[gold][pred] += 1
        if item["correct"]:
            correct += 1
        if item["pred"] in labels:
            valid_choice_predictions += 1

    total = len(results)
    per_class = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in labels if gold != label)
        fn = sum(confusion[label][pred] for pred in confusion[label] if pred != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        support = gold_counter[label]
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    macro_precision /= len(labels)
    macro_recall /= len(labels)
    macro_f1 /= len(labels)

    return {
        "samples": total,
        "correct": correct,
        "accuracy": safe_div(correct, total),
        "valid_choice_rate": safe_div(valid_choice_predictions, total),
        "empty_prediction_rate": safe_div(pred_counter["EMPTY"], total),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "gold_distribution": dict(gold_counter),
        "pred_distribution": dict(pred_counter),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def evaluate(args):
    data = load_data(Path(args.data_path))
    if args.max_samples is not None:
        data = data[: args.max_samples]

    tokenizer, model = load_model(args.base_model_path, args.adapter_path)

    results = []
    for idx, sample in enumerate(data, start=1):
        pred, raw_output = predict_one(tokenizer, model, sample, args.max_new_tokens)
        gold = sample.get("output", "").strip().upper()
        gold_choice = extract_gold_choice(gold)
        is_correct = pred == gold_choice

        results.append(
            {
                "id": idx,
                "input": sample["input"],
                "gold": gold_choice,
                "correct_text": gold,
                "pred": pred,
                "correct": is_correct,
                "raw_output": raw_output,
            }
        )

        if idx % 50 == 0:
            print(f"processed {idx}/{len(data)}")

    metrics = compute_metrics(results)
    summary = {
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "data_path": args.data_path,
        **metrics,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Strict rich evaluation for the CSQA template-answer route.")
    parser.add_argument("--base_model_path", required=True, help="Path to the local base model.")
    parser.add_argument("--adapter_path", default=None, help="Path to the LoRA adapter directory.")
    parser.add_argument("--data_path", required=True, help="Path to evaluation json file.")
    parser.add_argument("--output_path", required=True, help="Path to save evaluation results.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on evaluation samples.")
    parser.add_argument("--max_new_tokens", type=int, default=16, help="Max generated tokens per sample.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
