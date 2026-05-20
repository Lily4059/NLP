import csv
import json
from pathlib import Path


ROOT = Path(r"d:\deadline520")
BASE_RESULT = ROOT / "base_result"
OUT_DIR = BASE_RESULT / "06_导出表格_最新版"


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["summary"]


def load_peak_memory_mb(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    peaks = data["summary"].get("peak_memory_by_gpu_mb", [])
    if not peaks:
        return 0
    return max(item["memory_used_mb"] for item in peaks)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_csv(file_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {file_path}")


def route_row(label: str, summary: dict, note: str) -> dict:
    return {
        "模型/设置": label,
        "评估数据": Path(summary["data_path"]).name,
        "样本数": summary["samples"],
        "Accuracy": fmt(summary["accuracy"]),
        "Macro-F1": fmt(summary["macro_f1"]),
        "Valid Choice Rate": fmt(summary["valid_choice_rate"]),
        "Empty Prediction Rate": fmt(summary["empty_prediction_rate"]),
        "说明": note,
    }


def ablation_row(setting: str, summary: dict) -> dict:
    return {
        "设置": setting,
        "样本数": summary["samples"],
        "Accuracy": fmt(summary["accuracy"]),
        "Macro-F1": fmt(summary["macro_f1"]),
        "Valid Choice Rate": fmt(summary["valid_choice_rate"]),
        "Empty Prediction Rate": fmt(summary["empty_prediction_rate"]),
    }


def cmp_row(label: str, summary: dict, peak_memory_mb: int, note: str) -> dict:
    return {
        "模型/设置": label,
        "评估数据": Path(summary["data_path"]).name,
        "样本数": summary["samples"],
        "峰值显存(MB)": peak_memory_mb,
        "Accuracy": fmt(summary["accuracy"]),
        "Macro-F1": fmt(summary["macro_f1"]),
        "Valid Choice Rate": fmt(summary["valid_choice_rate"]),
        "Empty Prediction Rate": fmt(summary["empty_prediction_rate"]),
        "说明": note,
    }


def main() -> None:
    strict_eval = BASE_RESULT / "01_严格主线" / "评估结果"
    strict_first_valid_eval = BASE_RESULT / "02_严格主线_补充评估" / "评估结果"
    fullset_eval = BASE_RESULT / "03_旧版Fullset路线" / "评估结果"
    tmpl_eval = BASE_RESULT / "04_模板输出路线" / "评估结果"

    old_baseline = load_summary(fullset_eval / "csqa_base_baseline_rich_eval.json")
    old_fullset = load_summary(fullset_eval / "csqa_base_fullset_lora_rich_eval.json")
    strict_baseline = load_summary(strict_eval / "csqa_base_baseline_strict_rich_eval.json")
    strict_main = load_summary(strict_eval / "csqa_base_strict_lora_rich_eval.json")
    tmpl_baseline = load_summary(tmpl_eval / "csqa_base_baseline_tmpl_rich_eval.json")
    tmpl_main = load_summary(tmpl_eval / "csqa_base_tmpl_lora_rich_eval.json")

    write_csv(
        OUT_DIR / "01_旧版Fullset路线_微调前后对比.csv",
        [
            route_row("基础模型 baseline", old_baseline, "未微调，采用原始输出方式"),
            route_row("旧版 fullset + LoRA", old_fullset, "微调后输出格式不稳定，空预测增多"),
        ],
        ["模型/设置", "评估数据", "样本数", "Accuracy", "Macro-F1", "Valid Choice Rate", "Empty Prediction Rate", "说明"],
    )

    write_csv(
        OUT_DIR / "02_strict路线_微调前后对比.csv",
        [
            route_row("strict baseline", strict_baseline, "未微调，但施加严格输出约束"),
            route_row("strict + LoRA 主实验", strict_main, "strict 主线当前主实验"),
        ],
        ["模型/设置", "评估数据", "样本数", "Accuracy", "Macro-F1", "Valid Choice Rate", "Empty Prediction Rate", "说明"],
    )

    write_csv(
        OUT_DIR / "03_模板输出路线_微调前后对比.csv",
        [
            route_row("tmpl baseline", tmpl_baseline, "未微调，要求输出 The correct answer is X."),
            route_row("tmpl + LoRA", tmpl_main, "模板输出路线主实验"),
        ],
        ["模型/设置", "评估数据", "样本数", "Accuracy", "Macro-F1", "Valid Choice Rate", "Empty Prediction Rate", "说明"],
    )

    strict_lr_rows = [
        ablation_row("1e-5", load_summary(strict_eval / "csqa_base_strict_lora_lr1e5_rich_eval.json")),
        ablation_row("5e-5", load_summary(strict_eval / "csqa_base_strict_lora_lr5e5_rich_eval.json")),
        ablation_row("1e-4", strict_main),
        ablation_row("2e-4", load_summary(strict_eval / "csqa_base_strict_lora_lr2e4_rich_eval.json")),
        ablation_row("5e-4", load_summary(strict_eval / "csqa_base_strict_lora_lr5e4_rich_eval.json")),
    ]
    write_csv(
        OUT_DIR / "04_strict_学习率消融.csv",
        strict_lr_rows,
        ["设置", "样本数", "Accuracy", "Macro-F1", "Valid Choice Rate", "Empty Prediction Rate"],
    )

    strict_rank_rows = [
        ablation_row("1", load_summary(strict_eval / "csqa_base_strict_lora_rank1_rich_eval.json")),
        ablation_row("4", load_summary(strict_eval / "csqa_base_strict_lora_rank4_rich_eval.json")),
        ablation_row("8", strict_main),
        ablation_row("16", load_summary(strict_eval / "csqa_base_strict_lora_rank16_rich_eval.json")),
        ablation_row("32", load_summary(strict_eval / "csqa_base_strict_lora_rank32_rich_eval.json")),
    ]
    write_csv(
        OUT_DIR / "05_strict_rank消融.csv",
        strict_rank_rows,
        ["设置", "样本数", "Accuracy", "Macro-F1", "Valid Choice Rate", "Empty Prediction Rate"],
    )

    strict_dropout_rows = [
        ablation_row("0.0", load_summary(strict_eval / "csqa_base_strict_lora_dropout0_rich_eval.json")),
        ablation_row("0.05", strict_main),
        ablation_row("0.1", load_summary(strict_eval / "csqa_base_strict_lora_dropout1_rich_eval.json")),
        ablation_row("0.2", load_summary(strict_eval / "csqa_base_strict_lora_dropout2_rich_eval.json")),
        ablation_row("0.3", load_summary(strict_eval / "csqa_base_strict_lora_dropout3_rich_eval.json")),
    ]
    write_csv(
        OUT_DIR / "06_strict_dropout消融.csv",
        strict_dropout_rows,
        ["设置", "样本数", "Accuracy", "Macro-F1", "Valid Choice Rate", "Empty Prediction Rate"],
    )

    strict_epoch_rows = [
        ablation_row("1", load_summary(strict_eval / "csqa_base_strict_lora_epoch1_rich_eval.json")),
        ablation_row("2", strict_main),
        ablation_row("3", load_summary(strict_eval / "csqa_base_strict_lora_epoch3_rich_eval.json")),
    ]
    write_csv(
        OUT_DIR / "07_strict_epoch消融.csv",
        strict_epoch_rows,
        ["设置", "样本数", "Accuracy", "Macro-F1", "Valid Choice Rate", "Empty Prediction Rate"],
    )

    strict_first_valid_rows = [
        route_row(
            "strict baseline (first_valid)",
            load_summary(strict_first_valid_eval / "csqa_base_baseline_strict_first_valid_rich_eval.json"),
            "strict 补充评估口径 baseline",
        ),
        route_row(
            "strict + LoRA 主实验 (first_valid)",
            load_summary(strict_first_valid_eval / "csqa_base_strict_lora_first_valid_rich_eval.json"),
            "strict 补充评估口径主实验",
        ),
    ]
    write_csv(
        OUT_DIR / "08_strict_first_valid_微调前后对比.csv",
        strict_first_valid_rows,
        ["模型/设置", "评估数据", "样本数", "Accuracy", "Macro-F1", "Valid Choice Rate", "Empty Prediction Rate", "说明"],
    )

    cmp_rows = [
        cmp_row(
            "strict LoRA cmp",
            load_summary(BASE_RESULT / "csqa_base_strict_lora_cmp_rich_eval.json"),
            load_peak_memory_mb(BASE_RESULT / "csqa_base_strict_lora_cmp_memory.json"),
            "拓展部分公平对照 LoRA",
        ),
        cmp_row(
            "strict Full cmp",
            load_summary(BASE_RESULT / "csqa_base_strict_full_cmp_rich_eval.json"),
            load_peak_memory_mb(BASE_RESULT / "csqa_base_strict_full_cmp_memory.json"),
            "拓展部分公平对照 Full",
        ),
        cmp_row(
            "strict LoRA bestcmp",
            load_summary(BASE_RESULT / "csqa_base_strict_lora_bestcmp_rich_eval.json"),
            load_peak_memory_mb(BASE_RESULT / "csqa_base_strict_lora_bestcmp_memory.json"),
            "拓展部分较优设置 LoRA",
        ),
    ]
    write_csv(
        OUT_DIR / "09_拓展部分_LoRA_vs_Full.csv",
        cmp_rows,
        ["模型/设置", "评估数据", "样本数", "峰值显存(MB)", "Accuracy", "Macro-F1", "Valid Choice Rate", "Empty Prediction Rate", "说明"],
    )

    all_rows = [
        route_row("普通 baseline", old_baseline, "原始输出口径"),
        route_row("旧版 fullset + LoRA", old_fullset, "旧路线主实验"),
        route_row("strict baseline", strict_baseline, "strict 口径 baseline"),
        route_row("strict + LoRA 主实验", strict_main, "当前 strict 主线"),
        route_row("tmpl baseline", tmpl_baseline, "模板输出 baseline"),
        route_row("tmpl + LoRA", tmpl_main, "模板输出主实验"),
    ]
    write_csv(
        OUT_DIR / "00_当前主要结果总表.csv",
        all_rows,
        ["模型/设置", "评估数据", "样本数", "Accuracy", "Macro-F1", "Valid Choice Rate", "Empty Prediction Rate", "说明"],
    )


if __name__ == "__main__":
    main()
