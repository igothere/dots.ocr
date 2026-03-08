#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

from dots_ocr.parser import DotsOCRParser


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_images(crop_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = [p for p in crop_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=natural_key)


def main():
    ap = argparse.ArgumentParser(description="Run OCR on crop images and merge outputs into one Markdown.")
    ap.add_argument("crop_dir", help="Directory containing crop images")
    ap.add_argument("--merged-md", required=True, help="Output merged markdown file path")
    ap.add_argument("--tmp-output-dir", default="./output_crop_ocr", help="Temp parser output directory")
    ap.add_argument("--protocol", default="http", choices=["http", "https"])
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-completion-tokens", type=int, default=4096)
    ap.add_argument("--use-hf", action="store_true", help="Use local HF model instead of vLLM")
    args = ap.parse_args()

    crop_dir = Path(args.crop_dir).resolve()
    if not crop_dir.exists() or not crop_dir.is_dir():
        raise FileNotFoundError(f"crop_dir not found: {crop_dir}")

    images = list_images(crop_dir)
    if not images:
        raise RuntimeError(f"No images found in: {crop_dir}")

    parser = DotsOCRParser(
        protocol=args.protocol,
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=1,
        output_dir=args.tmp_output_dir,
        use_hf=args.use_hf,
    )

    merged_md = Path(args.merged_md).resolve()
    merged_md.parent.mkdir(parents=True, exist_ok=True)

    blocks = []
    total = len(images)
    for i, img_path in enumerate(images, start=1):
        print(f"[{i}/{total}] OCR: {img_path.name}")
        results = parser.parse_file(
            str(img_path),
            prompt_mode="prompt_ocr",
            fitz_preprocess=False,
        )
        if not results or "md_content_path" not in results[0]:
            raise RuntimeError(f"Missing OCR markdown output for {img_path}")
        md_path = Path(results[0]["md_content_path"])
        content = md_path.read_text(encoding="utf-8").strip()
        blocks.append(f"## {img_path.stem}\n\n{content}")

    merged_md.write_text("\n\n".join(blocks).strip() + "\n", encoding="utf-8")
    print(f"Saved merged markdown: {merged_md}")


if __name__ == "__main__":
    main()
