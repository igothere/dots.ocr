import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.output_cleaner import OutputCleaner


def _extract_json_like_text(text: str) -> str:
    """Extract the most likely JSON payload from model output."""
    if not text:
        return text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _clean_json_prompt_output(text: str, prompt_mode: str) -> str:
    """Repair malformed JSON-like outputs for layout-style prompts."""
    if prompt_mode not in {"prompt_layout_only_en", "prompt_web_parsing", "prompt_grounding_ocr"}:
        return text

    candidate = _extract_json_like_text(text)
    try:
        parsed = json.loads(candidate)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        cleaner = OutputCleaner()
        cleaned = cleaner.clean_model_output(candidate)
        if isinstance(cleaned, list):
            return json.dumps(cleaned, ensure_ascii=False, indent=2)
        return str(cleaned)


def _generation_config_for_prompt(prompt_mode: str):
    # Task-specific caps to avoid long repetitive degeneration.
    max_new_tokens_by_mode = {
        "prompt_layout_only_en": 2500,
        "prompt_layout_all_en": 5000,
        "prompt_scene_spotting": 4500,
        "prompt_grounding_ocr": 1500,
        "prompt_image_to_svg": 6500,
        "prompt_web_parsing": 3500,
        "prompt_ocr": 2500,
        "prompt_general": 1500,
    }
    return {
        "max_new_tokens": max_new_tokens_by_mode.get(prompt_mode, 3000),
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "repetition_penalty": 1.08,
        "no_repeat_ngram_size": 8,
    }


def inference(image_path, prompt, prompt_mode, model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    with torch.inference_mode():
        generation_cfg = _generation_config_for_prompt(prompt_mode)
        generated_ids = model.generate(
            **inputs,
            **generation_cfg,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return _clean_json_prompt_output(output_text[0], prompt_mode)


def save_result(result: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(result)

    os.replace(tmp_path, output_path)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="demo/test.jpg", help="input image path")
    parser.add_argument(
        "--prompt-mode",
        default="prompt_layout_all_en",
        help="prompt mode key from dict_promptmode_to_prompt"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="directory to save result files"
    )
    parser.add_argument(
        "--all-prompts",
        action="store_true",
        help="run all prompt modes instead of one"
    )
    args = parser.parse_args()

    model_path = "./weights/DotsOCR_1_5"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        # use_fast=False,  # 필요하면 fast processor 경고 줄이려고 사용
    )

    image_path = args.image
    image_stem = Path(image_path).stem
    output_dir = Path(args.output_dir)

    if args.all_prompts:
        prompt_items = dict_promptmode_to_prompt.items()
    else:
        if args.prompt_mode not in dict_promptmode_to_prompt:
            print(f"Invalid prompt mode: {args.prompt_mode}")
            print("Available prompt modes:")
            for key in dict_promptmode_to_prompt.keys():
                print(f"  - {key}")
            return
        prompt_items = [(args.prompt_mode, dict_promptmode_to_prompt[args.prompt_mode])]

    for prompt_mode, prompt in prompt_items:
        print("=" * 80)
        print(f"prompt_mode: {prompt_mode}")
        print(f"image_path : {image_path}")

        result = inference(image_path, prompt, prompt_mode, model, processor)

        output_path = output_dir / f"{image_stem}_{prompt_mode}.md"
        save_result(result, output_path)


if __name__ == "__main__":
    main()
