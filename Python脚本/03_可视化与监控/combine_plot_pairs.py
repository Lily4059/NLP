from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"d:\deadline520\base_result")
OUT_DIR = ROOT / "05_对比图"

PAIRS = [
    {
        "left": ROOT / "03_旧版Fullset路线" / "可视化图" / "qwen25_base_csqa_fullset_lora_plots" / "training_loss.png",
        "right": ROOT / "01_严格主线" / "可视化图" / "qwen25_base_csqa_strict_lora_plots" / "training_loss.png",
        "output": OUT_DIR / "旧版Fullset_vs_严格主线_training_loss_对比.png",
        "title": "Training Loss Comparison",
        "left_label": "Old Fullset LoRA",
        "right_label": "Strict LoRA",
    },
    {
        "left": ROOT / "03_旧版Fullset路线" / "可视化图" / "qwen25_base_csqa_fullset_lora_plots" / "validation_loss.png",
        "right": ROOT / "01_严格主线" / "可视化图" / "qwen25_base_csqa_strict_lora_plots" / "validation_loss.png",
        "output": OUT_DIR / "旧版Fullset_vs_严格主线_validation_loss_对比.png",
        "title": "Validation Loss Comparison",
        "left_label": "Old Fullset LoRA",
        "right_label": "Strict LoRA",
    },
]


def load_font(size: int):
    candidates = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_pair(left_path: Path, right_path: Path, output_path: Path, title: str, left_label: str, right_label: str) -> None:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    width = left.width + right.width
    height = max(left.height, right.height)
    top_margin = 90
    label_margin = 36
    canvas = Image.new("RGB", (width, height + top_margin + label_margin), "white")

    canvas.paste(left, (0, top_margin))
    canvas.paste(right, (left.width, top_margin))

    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    label_font = load_font(20)

    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((width - title_width) // 2, 20), title, fill="black", font=title_font)

    left_bbox = draw.textbbox((0, 0), left_label, font=label_font)
    left_label_width = left_bbox[2] - left_bbox[0]
    draw.text(((left.width - left_label_width) // 2, 58), left_label, fill="black", font=label_font)

    right_bbox = draw.textbbox((0, 0), right_label, font=label_font)
    right_label_width = right_bbox[2] - right_bbox[0]
    draw.text((left.width + (right.width - right_label_width) // 2, 58), right_label, fill="black", font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"saved: {output_path}")


def main() -> None:
    for pair in PAIRS:
        make_pair(
            left_path=pair["left"],
            right_path=pair["right"],
            output_path=pair["output"],
            title=pair["title"],
            left_label=pair["left_label"],
            right_label=pair["right_label"],
        )


if __name__ == "__main__":
    main()
