#!/usr/bin/env python3
import argparse
import json
import re
from html import unescape
from pathlib import Path
from statistics import median
from xml.etree import ElementTree as ET

from PIL import Image, ImageEnhance, ImageFilter
from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.format_transformer import normalize_table_html
from dots_ocr.utils.prompts import dict_promptmode_to_prompt

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_images(crop_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = [p for p in crop_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=natural_key)


def extract_table_html_blocks(result: dict):
    layout_info_path = result.get("layout_info_path")
    if not layout_info_path:
        return []
    path = Path(layout_info_path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []

    table_blocks = []
    for cell in payload:
        if not isinstance(cell, dict):
            continue
        if cell.get("category") != "Table":
            continue
        html = normalize_table_html(cell.get("text", ""))
        if html:
            table_blocks.append(html)
    return table_blocks


def extract_table_blocks_from_text(text: str):
    if not isinstance(text, str):
        return []
    pattern = re.compile(r"<table\b[\s\S]*?</table>", re.IGNORECASE)
    matches = pattern.findall(text)
    blocks = [normalize_table_html(x) for x in matches]
    return [x for x in blocks if x]


def join_table_blocks(blocks):
    return "\n\n".join([x.strip() for x in blocks if x.strip()]).strip()


def strip_tags(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


def parse_table_xml(table_html: str):
    if not isinstance(table_html, str) or not table_html.strip():
        return None
    table_html = normalize_table_html(table_html)
    try:
        return ET.fromstring(table_html)
    except Exception:
        return None


def get_trs(root):
    return [x for x in root.iter() if x.tag.lower() == "tr"]


def get_cells(tr):
    return [x for x in list(tr) if x.tag.lower() in ("td", "th")]


def as_int(value, default=1):
    try:
        v = int(str(value))
        return v if v > 0 else default
    except Exception:
        return default


def row_effective_cols(tr):
    return sum(as_int(cell.attrib.get("colspan", 1), 1) for cell in get_cells(tr))


def table_span_count(table_html: str) -> int:
    return len(re.findall(r"\b(?:rowspan|colspan)\s*=\s*\"?\d+", table_html or "", flags=re.IGNORECASE))


def table_top_text(table_html: str, max_rows=2) -> str:
    root = parse_table_xml(table_html)
    if root is None:
        return ""
    texts = []
    for tr in get_trs(root)[:max_rows]:
        for cell in get_cells(tr):
            t = strip_tags("".join(cell.itertext()))
            if t:
                texts.append(t)
    return " ".join(texts).lower()


def validate_table_grid(table_html: str):
    root = parse_table_xml(table_html)
    if root is None:
        return {
            "valid": False,
            "score": -5.0,
            "row_cols": [],
            "span_count": 0,
            "reasons": ["parse_failed"],
        }

    trs = get_trs(root)
    row_cols = [row_effective_cols(tr) for tr in trs if get_cells(tr)]
    if not row_cols:
        return {
            "valid": False,
            "score": -4.0,
            "row_cols": [],
            "span_count": 0,
            "reasons": ["empty_rows"],
        }

    span_count = table_span_count(table_html)
    max_cols = max(row_cols)
    min_cols = min(row_cols)
    expected = int(median(row_cols))
    mismatch_rows = sum(1 for c in row_cols if abs(c - expected) > 2)
    width_stable = (max_cols - min_cols) <= 2

    score = 0.0
    score += 2.0
    score += min(float(span_count), 4.0)
    if width_stable:
        score += 2.0
    if mismatch_rows:
        score -= 0.8 * mismatch_rows

    valid = (max_cols > 0) and (mismatch_rows <= max(1, len(row_cols) // 4))
    reasons = []
    if not width_stable:
        reasons.append("row_width_variance")
    if mismatch_rows:
        reasons.append("header_or_grid_mismatch")
    return {
        "valid": valid,
        "score": score,
        "row_cols": row_cols,
        "span_count": span_count,
        "reasons": reasons,
    }


def should_trigger_stage2(table_html: str):
    meta = validate_table_grid(table_html)
    top = table_top_text(table_html, max_rows=2)
    keywords = ("구분", "합계", "계획", "실적")
    has_keyword = any(k in top for k in keywords)

    row_cols = meta["row_cols"]
    header_multi = len(row_cols) >= 2
    header_mismatch = False
    if len(row_cols) >= 2:
        header_mismatch = abs(row_cols[0] - row_cols[1]) >= 2

    no_span = meta["span_count"] == 0
    if no_span and header_multi:
        return True, "no_span_with_multi_header"
    if no_span and header_mismatch:
        return True, "no_span_header_col_mismatch"
    if no_span and has_keyword and header_multi:
        return True, "no_span_keyword_multilevel_header"
    if not meta["valid"]:
        return True, "invalid_grid"
    return False, "stage1_ok"


def preprocess_table_image(image_path: Path, mode: str, enhance_lines: bool):
    image = Image.open(image_path).convert("RGB")
    if mode == "upscale":
        image = image.resize((image.width * 2, image.height * 2), RESAMPLE_LANCZOS)
    elif mode == "contrast":
        image = ImageEnhance.Contrast(image).enhance(2.0)
    elif mode == "binarize":
        gray = image.convert("L")
        image = gray.point(lambda x: 255 if x > 180 else 0, mode="1").convert("RGB")
    elif mode == "upscale_contrast":
        image = image.resize((image.width * 2, image.height * 2), RESAMPLE_LANCZOS)
        image = ImageEnhance.Contrast(image).enhance(1.8)

    if enhance_lines:
        gray = image.convert("L")
        # Weak morphology-like effect to sharpen straight table lines.
        gray = gray.filter(ImageFilter.MaxFilter(size=3)).filter(ImageFilter.MinFilter(size=3))
        image = gray.convert("RGB")
    return image


def infer_table_html_stage2(parser: DotsOCRParser, image, max_tokens: int):
    prompt = (
        "You are given one table image. Output ONLY ONE HTML <table>...</table>.\n"
        "Rules:\n"
        "1) Use valid HTML table tags only.\n"
        "2) Preserve merged cells with exact rowspan/colspan.\n"
        "3) Keep empty cells as <td></td>.\n"
        "4) Put multi-row headers inside <thead>.\n"
        "5) Do not add explanations or markdown fences."
    )

    old_tokens = parser.max_completion_tokens
    parser.max_completion_tokens = max_tokens
    try:
        if parser.use_hf:
            response = parser._inference_with_hf(image, prompt, "prompt_general")
        else:
            response = parser._inference_with_vllm(image, prompt, "prompt_general")
    finally:
        parser.max_completion_tokens = old_tokens

    blocks = extract_table_blocks_from_text(response)
    if blocks:
        return blocks
    cleaned = normalize_table_html(str(response))
    return [cleaned] if cleaned else []


def select_best_table(stage1_html: str, stage2_html: str):
    v1 = validate_table_grid(stage1_html)
    v2 = validate_table_grid(stage2_html)
    if v2["score"] > v1["score"] + 0.4:
        return stage2_html, "stage2_better_score", v1, v2
    if (not v1["valid"]) and v2["valid"]:
        return stage2_html, "stage2_valid_stage1_invalid", v1, v2
    return stage1_html, "stage1_kept", v1, v2


def repair_table_html(table_html: str):
    root = parse_table_xml(table_html)
    if root is None:
        return normalize_table_html(table_html), [], True

    reasons = []
    changed = False

    thead = None
    for child in list(root):
        if child.tag.lower() == "thead":
            thead = child
            break
    if thead is not None:
        header_rows = [x for x in list(thead) if x.tag.lower() == "tr"]
        if len(header_rows) >= 2:
            r1 = get_cells(header_rows[0])
            r2 = get_cells(header_rows[1])
            r1_has_span = any(as_int(c.attrib.get("colspan", 1), 1) > 1 for c in r1)
            r1_n = len(r1)
            r2_n = sum(as_int(c.attrib.get("colspan", 1), 1) for c in r2)
            if (not r1_has_span) and r1_n > 0 and r2_n > r1_n and (r2_n % r1_n == 0):
                ratio = r2_n // r1_n
                if 1 < ratio <= 4:
                    for c in r1:
                        c.set("colspan", str(ratio))
                    changed = True
                    reasons.append("colspan_inferred_from_header_ratio")

    tbody = None
    for child in list(root):
        if child.tag.lower() == "tbody":
            tbody = child
            break
    if tbody is not None:
        rows = [x for x in list(tbody) if x.tag.lower() == "tr"]
        i = 0
        while i < len(rows) - 1:
            cells_i = get_cells(rows[i])
            if not cells_i:
                i += 1
                continue
            c0 = cells_i[0]
            if as_int(c0.attrib.get("rowspan", 1), 1) > 1:
                i += 1
                continue
            t0 = strip_tags("".join(c0.itertext()))
            if not t0:
                i += 1
                continue

            j = i + 1
            merge_count = 0
            while j < len(rows):
                cells_j = get_cells(rows[j])
                if not cells_j:
                    break
                first_j = cells_j[0]
                tj = strip_tags("".join(first_j.itertext()))
                if tj:
                    break
                rows[j].remove(first_j)
                merge_count += 1
                j += 1
            if merge_count > 0:
                c0.set("rowspan", str(merge_count + 1))
                changed = True
                reasons.append("rowspan_inferred_from_empty_first_cells")
                i = j
            else:
                i += 1

    out = ET.tostring(root, encoding="unicode")
    out = normalize_table_html(out)
    post_meta = validate_table_grid(out)
    needs_review = (not post_meta["valid"]) or (post_meta["span_count"] == 0 and len(post_meta["row_cols"]) >= 2)
    if not changed and needs_review:
        reasons.append("needs_review_no_safe_fix")
    return out, reasons, needs_review


def main():
    ap = argparse.ArgumentParser(description="Run OCR on crop images and merge outputs into one Markdown.")
    ap.add_argument("crop_dir", help="Directory containing crop images")
    ap.add_argument("--merged-md", required=True, help="Output merged markdown file path")
    ap.add_argument("--tmp-output-dir", default="./output_crop_ocr", help="Temp parser output directory")
    ap.add_argument(
        "--prompt-mode",
        default="prompt_layout_all_en",
        choices=sorted(dict_promptmode_to_prompt.keys()),
        help="Prompt mode for crop parsing (default: prompt_layout_all_en for better table structure retention)",
    )
    ap.add_argument("--protocol", default="http", choices=["http", "https"])
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-completion-tokens", type=int, default=4096)
    ap.add_argument(
        "--table-preprocess",
        default="upscale_contrast",
        choices=["none", "upscale", "contrast", "binarize", "upscale_contrast"],
        help="Optional table image preprocessing before stage2 re-inference",
    )
    ap.add_argument(
        "--enhance-lines",
        action="store_true",
        help="Apply weak morphology-like line enhancement before stage2 re-inference",
    )
    ap.add_argument(
        "--stage2-max-completion-tokens",
        type=int,
        default=4096,
        help="Max completion tokens for strict table stage2 re-inference",
    )
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
        print(f"[{i}/{total}] Parse: {img_path.name} (prompt={args.prompt_mode})")
        debug_dir = Path(args.tmp_output_dir).resolve() / img_path.stem
        debug_dir.mkdir(parents=True, exist_ok=True)

        results = parser.parse_file(
            str(img_path),
            prompt_mode=args.prompt_mode,
            fitz_preprocess=False,
        )
        if not results:
            raise RuntimeError(f"Missing parser output for {img_path}")
        result = results[0]

        stage1_blocks = extract_table_html_blocks(result)
        stage1_content = join_table_blocks(stage1_blocks)
        if not stage1_content:
            md_content_path = result.get("md_content_path")
            if not md_content_path:
                raise RuntimeError(f"Missing markdown output for {img_path}")
            md_path = Path(md_content_path)
            stage1_content = md_path.read_text(encoding="utf-8").strip()
            stage1_blocks = extract_table_blocks_from_text(stage1_content)

        (debug_dir / f"{img_path.stem}_stage1.html").write_text(stage1_content + "\n", encoding="utf-8")

        trigger, trigger_reason = (False, "no_table_detected")
        if stage1_blocks:
            trigger, trigger_reason = should_trigger_stage2(stage1_blocks[0])

        selected_stage = "stage1"
        select_reason = "stage1_only"
        validation = {"trigger_reason": trigger_reason}
        final_content = stage1_content

        if trigger:
            prep = preprocess_table_image(img_path, args.table_preprocess, args.enhance_lines)
            stage2_blocks = infer_table_html_stage2(parser, prep, args.stage2_max_completion_tokens)
            stage2_content = join_table_blocks(stage2_blocks)
            if stage2_content:
                (debug_dir / f"{img_path.stem}_stage2.html").write_text(stage2_content + "\n", encoding="utf-8")
                best, select_reason, v1, v2 = select_best_table(stage1_blocks[0], stage2_blocks[0])
                selected_stage = "stage2" if best == stage2_blocks[0] else "stage1"
                validation["stage1"] = v1
                validation["stage2"] = v2
                final_content = best
            else:
                validation["stage2"] = {"valid": False, "score": -10.0, "reasons": ["empty_stage2_output"]}

        repaired, repair_reasons, needs_review = repair_table_html(final_content)
        final_content = repaired if repaired else final_content
        validation["selected_stage"] = selected_stage
        validation["selection_reason"] = select_reason
        validation["repair_reasons"] = repair_reasons
        validation["needs_review"] = needs_review

        (debug_dir / f"{img_path.stem}_final.html").write_text(final_content + "\n", encoding="utf-8")
        (debug_dir / f"{img_path.stem}_validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        blocks.append(f"## {img_path.stem}\n\n{final_content}")

    merged_md.write_text("\n\n".join(blocks).strip() + "\n", encoding="utf-8")
    print(f"Saved merged markdown: {merged_md}")


if __name__ == "__main__":
    main()
