import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(r"d:\deadline520\base_result")

FULLSET_STATE = (
    ROOT
    / "03_旧版Fullset路线"
    / "训练输出"
    / "qwen25_base_csqa_fullset_lora"
    / "trainer_state.json"
)
STRICT_STATE = (
    ROOT
    / "01_严格主线"
    / "训练输出"
    / "qwen25_base_csqa_strict_lora"
    / "trainer_state.json"
)
TMPL_STATE = (
    ROOT
    / "04_模板输出路线"
    / "qwen25_base_csqa_tmpl_lora"
    / "trainer_state.json"
)
OUT_DIR = ROOT / "05_对比图"


def load_trainer_state(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_series(log_history: list[dict], key: str) -> tuple[list[int], list[float]]:
    steps = []
    values = []
    for item in log_history:
        if key in item and "step" in item:
            steps.append(item["step"])
            values.append(item[key])
    return steps, values


def plot_overlay(
    series_list: list[tuple[str, list[int], list[float]]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 5.5))
    for label, steps, values in series_list:
        plt.plot(steps, values, marker="o", markersize=3, linewidth=1.7, label=label)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"saved: {output_path}")


def main() -> None:
    fullset_log = load_trainer_state(FULLSET_STATE).get("log_history", [])
    strict_log = load_trainer_state(STRICT_STATE).get("log_history", [])
    tmpl_log = load_trainer_state(TMPL_STATE).get("log_history", [])

    fullset_loss_steps, fullset_loss_values = extract_series(fullset_log, "loss")
    strict_loss_steps, strict_loss_values = extract_series(strict_log, "loss")
    tmpl_loss_steps, tmpl_loss_values = extract_series(tmpl_log, "loss")
    plot_overlay(
        [
            ("Old Fullset LoRA", fullset_loss_steps, fullset_loss_values),
            ("Strict LoRA", strict_loss_steps, strict_loss_values),
            ("Template LoRA", tmpl_loss_steps, tmpl_loss_values),
        ],
        title="Training Loss Comparison",
        ylabel="loss",
        output_path=OUT_DIR / "旧版Fullset_vs_严格主线_vs_模板路线_training_loss_三线对比.png",
    )

    fullset_eval_steps, fullset_eval_values = extract_series(fullset_log, "eval_loss")
    strict_eval_steps, strict_eval_values = extract_series(strict_log, "eval_loss")
    tmpl_eval_steps, tmpl_eval_values = extract_series(tmpl_log, "eval_loss")
    plot_overlay(
        [
            ("Old Fullset LoRA", fullset_eval_steps, fullset_eval_values),
            ("Strict LoRA", strict_eval_steps, strict_eval_values),
            ("Template LoRA", tmpl_eval_steps, tmpl_eval_values),
        ],
        title="Validation Loss Comparison",
        ylabel="eval_loss",
        output_path=OUT_DIR / "旧版Fullset_vs_严格主线_vs_模板路线_validation_loss_三线对比.png",
    )


if __name__ == "__main__":
    main()
