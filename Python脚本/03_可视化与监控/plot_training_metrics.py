import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


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


def plot_one(steps: list[int], values: list[float], title: str, ylabel: str, output_path: Path):
    if not steps:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, marker="o", markersize=3, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from LLaMA-Factory trainer_state.json.")
    parser.add_argument("--trainer_state", required=True, help="Path to trainer_state.json")
    parser.add_argument("--output_dir", required=True, help="Directory to save the plots")
    args = parser.parse_args()

    trainer_state = load_trainer_state(Path(args.trainer_state))
    log_history = trainer_state.get("log_history", [])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_specs = [
        ("loss", "Training Loss", "loss", "training_loss.png"),
        ("eval_loss", "Validation Loss", "eval_loss", "validation_loss.png"),
        ("grad_norm", "Gradient Norm", "grad_norm", "grad_norm.png"),
        ("learning_rate", "Learning Rate", "learning_rate", "learning_rate.png"),
    ]

    for key, title, ylabel, filename in series_specs:
        steps, values = extract_series(log_history, key)
        plot_one(steps, values, title, ylabel, output_dir / filename)

    summary = {
        "trainer_state": str(Path(args.trainer_state)),
        "output_dir": str(output_dir),
        "generated_files": [filename for _, _, _, filename in series_specs],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
