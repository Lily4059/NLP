import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(r"d:\deadline520")
STRICT_EVAL_DIR = ROOT / "base_result" / "01_严格主线" / "评估结果"
OUT_DIR = ROOT / "base_result" / "05_对比图"


def configure_chinese_font() -> None:
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    chosen = None
    for font_name in candidates:
        if font_name in available_fonts:
            chosen = font_name
            break

    if chosen is not None:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [chosen]
    else:
        plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def load_summary(file_name: str) -> dict:
    with (STRICT_EVAL_DIR / file_name).open("r", encoding="utf-8") as f:
        return json.load(f)["summary"]


def build_series(file_map: list[tuple[str, str]]) -> tuple[list[str], list[float], list[float]]:
    labels = []
    accuracies = []
    macro_f1s = []
    for label, file_name in file_map:
        summary = load_summary(file_name)
        labels.append(label)
        accuracies.append(summary["accuracy"])
        macro_f1s.append(summary["macro_f1"])
    return labels, accuracies, macro_f1s


def pick_bar_colors(labels: list[str], accuracies: list[float], default_label: str) -> list[str]:
    best_idx = max(range(len(accuracies)), key=lambda i: accuracies[i])
    worst_idx = min(range(len(accuracies)), key=lambda i: accuracies[i])
    colors = []
    for idx, label in enumerate(labels):
        if idx == best_idx and label == default_label:
            colors.append("#9b59b6")
        elif idx == best_idx:
            colors.append("#27ae60")
        elif idx == worst_idx:
            colors.append("#e74c3c")
        elif label == default_label:
            colors.append("#f39c12")
        else:
            colors.append("#7fb3d5")
    return colors


def plot_panel(
    ax,
    title: str,
    labels: list[str],
    accuracies: list[float],
    macro_f1s: list[float],
    default_label: str,
) -> None:
    x = list(range(len(labels)))
    bar_colors = pick_bar_colors(labels, accuracies, default_label)
    bars = ax.bar(x, accuracies, color=bar_colors, edgecolor="#34495e", linewidth=0.8, alpha=0.9)
    ax.plot(
        x,
        macro_f1s,
        color="#2c3e50",
        marker="o",
        markersize=6,
        linewidth=2,
        markerfacecolor="white",
        markeredgewidth=1.5,
    )
    ax.set_title(title, fontsize=15, pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("分数")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, accuracies):
        label_y = value + 0.018 if value < 0.9 else value - 0.06
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1f1f1f",
            fontweight="bold",
        )

    legend_handles = [
        Patch(facecolor="#7fb3d5", edgecolor="#34495e", label="普通设置"),
        Patch(facecolor="#f39c12", edgecolor="#34495e", label="默认设置"),
        Patch(facecolor="#27ae60", edgecolor="#34495e", label="本组最优"),
        Patch(facecolor="#e74c3c", edgecolor="#34495e", label="本组最差"),
        Patch(facecolor="#9b59b6", edgecolor="#34495e", label="默认且最优"),
        Line2D([0], [0], color="#2c3e50", marker="o", markerfacecolor="white", markeredgewidth=1.5, linewidth=2, label="Macro-F1"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, frameon=False, ncol=3)


def save_single_panel(
    file_name: str,
    title: str,
    labels: list[str],
    accuracies: list[float],
    macro_f1s: list[float],
    default_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.6))
    plot_panel(ax, title, labels, accuracies, macro_f1s, default_label)
    fig.tight_layout()
    out_path = OUT_DIR / file_name
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_chinese_font()
    lr_labels, lr_acc, lr_f1 = build_series(
        [
            ("1e-5", "csqa_base_strict_lora_lr1e5_rich_eval.json"),
            ("5e-5", "csqa_base_strict_lora_lr5e5_rich_eval.json"),
            ("1e-4", "csqa_base_strict_lora_rich_eval.json"),
            ("2e-4", "csqa_base_strict_lora_lr2e4_rich_eval.json"),
            ("5e-4", "csqa_base_strict_lora_lr5e4_rich_eval.json"),
        ]
    )
    rank_labels, rank_acc, rank_f1 = build_series(
        [
            ("1", "csqa_base_strict_lora_rank1_rich_eval.json"),
            ("4", "csqa_base_strict_lora_rank4_rich_eval.json"),
            ("8", "csqa_base_strict_lora_rich_eval.json"),
            ("16", "csqa_base_strict_lora_rank16_rich_eval.json"),
            ("32", "csqa_base_strict_lora_rank32_rich_eval.json"),
        ]
    )
    dropout_labels, dropout_acc, dropout_f1 = build_series(
        [
            ("0", "csqa_base_strict_lora_dropout0_rich_eval.json"),
            ("0.05", "csqa_base_strict_lora_rich_eval.json"),
            ("0.1", "csqa_base_strict_lora_dropout1_rich_eval.json"),
            ("0.2", "csqa_base_strict_lora_dropout2_rich_eval.json"),
            ("0.3", "csqa_base_strict_lora_dropout3_rich_eval.json"),
        ]
    )
    epoch_labels, epoch_acc, epoch_f1 = build_series(
        [
            ("1", "csqa_base_strict_lora_epoch1_rich_eval.json"),
            ("2", "csqa_base_strict_lora_rich_eval.json"),
            ("3", "csqa_base_strict_lora_epoch3_rich_eval.json"),
        ]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_single_panel("strict主线_学习率消融_柱状折线.png", "学习率消融实验", lr_labels, lr_acc, lr_f1, "1e-4")
    save_single_panel("strict主线_rank消融_柱状折线.png", "LoRA Rank 消融实验", rank_labels, rank_acc, rank_f1, "8")
    save_single_panel("strict主线_dropout消融_柱状折线.png", "Dropout 消融实验", dropout_labels, dropout_acc, dropout_f1, "0.05")
    save_single_panel("strict主线_epoch消融_柱状折线.png", "训练轮数消融实验", epoch_labels, epoch_acc, epoch_f1, "2")


if __name__ == "__main__":
    main()
