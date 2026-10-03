#!/usr/bin/env python3
"""Generate one short test dialogue line with OmniVoice.

This script assumes you already have OmniVoice installed and a checkpoint
available either as a local directory or in the Hugging Face cache.
"""

import argparse
import gc
import os
from pathlib import Path

import soundfile as sf

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from omnivoice import OmniVoice
import json

from omnivoice.utils.common import get_best_device
from omnivoice.utils.lora import load_lora_adapter

DEFAULT_TEXT = (
    "W tej wiosce są sami słabi chłopi. Nie ma nikogo, kto potrafi porządnie walczyć."
)
DEFAULT_INSTRUCT = ""
DEFAULT_LANGUAGE = "Polish"
DEFAULT_LORA_ADAPTER = "trained/Ass_Fat_Bearish/checkpoint-5000/model.safetensors"


def clear_device_cache(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_lora_adapter_path(path: str) -> Path:
    adapter_path = Path(path).expanduser()
    if adapter_path.is_file():
        adapter_path = adapter_path.parent

    adapter_config = adapter_path / "adapter_config.json"
    adapter_weights = adapter_path / "adapter_model.safetensors"
    if not adapter_config.is_file() or not adapter_weights.is_file():
        raise FileNotFoundError(
            "LoRA adapter path must point to a checkpoint directory containing "
            f"adapter_config.json and adapter_model.safetensors: {adapter_path}"
        )
    return adapter_path


def resolve_case_insensitive_path(path: str | Path) -> Path:
    file_path = Path(path).expanduser()
    if file_path.exists():
        return file_path

    parent = file_path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {parent}")

    matches = [
        child
        for child in parent.iterdir()
        if child.name.casefold() == file_path.name.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        match_names = ", ".join(str(match) for match in matches)
        raise FileExistsError(
            f"Ambiguous case-insensitive path for {file_path}: {match_names}"
        )
    raise FileNotFoundError(f"File does not exist: {file_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one OmniVoice test dialogue WAV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Local checkpoint directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="The single dialogue line to synthesize.",
    )
    parser.add_argument(
        "--output",
        default="test_dialogue.wav",
        help="Output WAV path.",
    )
    parser.add_argument(
        "--instruct",
        default=DEFAULT_INSTRUCT,
        help="Voice-design prompt. Use an empty string for auto voice.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Language name or code. Use an empty string for auto-detection.",
    )
    parser.add_argument(
        "--lora-adapter",
        "--lora_adapter",
        default=DEFAULT_LORA_ADAPTER,
        help="LoRA checkpoint directory, or a file inside it. Use an empty string to disable LoRA.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, for example cuda, cuda:0, xpu, mps, or cpu.",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=32,
        help="Diffusion steps. Lower values are faster but may reduce quality.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speaking speed. Values above 1.0 are faster.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or get_best_device()
    dtype = torch.float32 if device == "cpu" else torch.float16

    print(f"Loading OmniVoice model from {args.model!r} on {device}...")

    with open("modelToLinesWithRefs.json", "r", encoding="utf-8") as file:
        modelToLinesWithRefs = json.load(file)
    trained_directories = [d for d in os.listdir(f"{os.getcwd()}/trained")]
    with open("modelsEval.json", "r", encoding="utf-8") as file:
        jsonObject = json.load(file)
        filtered_models = [
            model for model, value in jsonObject.items() if (value == "Finetune_150")
        ]
    for i in range(0, 5):
        lora_type = "general"
        if i >= 1:
            lora_type = "finetune"
        for model_name, lines in modelToLinesWithRefs.items():
            # if "Ass_Elite_Insidious_Carlos" not in model_name:
            #    continue
            # if model_name not in filtered_models:
            #    continue
            if len(lines) == 0:
                continue
            if model_name not in trained_directories:
                continue
            steps = 45000
            lora_adapter = f"trained/GeneralModel/checkpoint-{steps}/model.safetensors"
            if lora_type == "finetune":
                steps = min(i * 100, 5000)
                lora_adapter = (
                    f"trained/{model_name}/checkpoint-{steps}/model.safetensors"
                )
            if not Path(lora_adapter).exists():
                continue
            lora_adapter_path = resolve_lora_adapter_path(lora_adapter)

            pending_lines = []
            for index, line in enumerate(lines[4:7], start=4):
                line["filename"] = f"{model_name}_{index + 1}_{lora_type}_{steps}.wav"
                output_path = Path(f"generated/{line["filename"]}")
                if output_path.exists():
                    continue
                pending_lines.append((line, output_path))

            if not pending_lines:
                continue

            model = None
            try:
                clear_device_cache(device)
                model = OmniVoice.from_pretrained(
                    args.model,
                    device_map=device,
                    dtype=dtype,
                    audio_tokenizer_device=(
                        "cpu" if str(device).startswith("cuda") else device
                    ),
                )
                print(f"Applying LoRA adapter from {lora_adapter_path}...")
                model = load_lora_adapter(model, str(lora_adapter_path))
                clear_device_cache(device)

                instruct = args.instruct.strip() or None
                language = args.language.strip() or None
                for line, output_path in pending_lines:
                    clear_device_cache(device)
                    text = line["line"]
                    refLine = line["simillar"]
                    ref_audio_path = resolve_case_insensitive_path(
                        f"SPEECH_MERGED/{refLine["filename"]}"
                    )
                    print(f"Generating: {line!r}")
                    audio = model.generate(
                        text=text,
                        language=language,
                        instruct=instruct,
                        num_step=args.num_step,
                        speed=args.speed,
                        ref_audio=str(ref_audio_path),
                        ref_text=refLine["line"],
                    )

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(output_path, audio[0], model.sampling_rate)
                    print(f"Saved {output_path} at {model.sampling_rate} Hz")
                    del audio
                    clear_device_cache(device)
            finally:
                if model is not None:
                    del model
                clear_device_cache(device)


if __name__ == "__main__":
    main()
