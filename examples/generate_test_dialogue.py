#!/usr/bin/env python3
"""Generate one short test dialogue line with OmniVoice.

This script assumes you already have OmniVoice installed and a checkpoint
available either as a local directory or in the Hugging Face cache.
"""

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

BASE_MODEL = "k2-fsa/OmniVoice"
TRAINED_ROOT = Path("trained")
CHECKPOINT_NAME = ""  # Use "" to generate from every checkpoint-* folder.
GENERATED_ROOT = Path("generated")
INSTRUCT = ""
LANGUAGE = "Polish"
LORA_ADAPTER = ""
DEVICE = None
NUM_STEP = 32
SPEED = 1.0


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


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.removeprefix("checkpoint-")), path.name
    except ValueError:
        return -1, path.name


def find_checkpoint_dirs(checkpoints_folder: Path, checkpoint_name: str) -> list[Path]:
    if checkpoint_name:
        checkpoint_dir = checkpoints_folder / checkpoint_name
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {checkpoint_dir}"
            )
        return [checkpoint_dir]

    return sorted(
        [
            path
            for path in checkpoints_folder.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-")
        ],
        key=checkpoint_sort_key,
    )


def load_model_for_checkpoint(
    checkpoint_dir: Path, device: str, dtype: torch.dtype, CHECKPOINT_MODE
) -> OmniVoice:
    audio_tokenizer_device = "cpu" if str(device).startswith("cuda") else device

    if CHECKPOINT_MODE == "full":
        print(f"Loading full checkpoint from {checkpoint_dir} on {device}...")
        return OmniVoice.from_pretrained(
            str(checkpoint_dir),
            device_map=device,
            dtype=dtype,
            audio_tokenizer_device=audio_tokenizer_device,
        )

    lora_adapter = LORA_ADAPTER.strip() or str(checkpoint_dir)
    lora_adapter_path = resolve_lora_adapter_path(lora_adapter)
    print(f"Loading base model from {BASE_MODEL!r} on {device}...")
    model = OmniVoice.from_pretrained(
        BASE_MODEL,
        device_map=device,
        dtype=dtype,
        audio_tokenizer_device=audio_tokenizer_device,
    )
    print(f"Applying LoRA adapter from {lora_adapter_path}...")
    return load_lora_adapter(model, str(lora_adapter_path))


def main() -> None:
    CHECKPOINT_MODE = "full"  # "full" or "lora"
    if CHECKPOINT_MODE not in {"full", "lora"}:
        raise ValueError(f"CHECKPOINT_MODE must be 'full' or 'lora': {CHECKPOINT_MODE}")

    device = DEVICE or get_best_device()
    dtype = torch.float32 if device == "cpu" else torch.float16

    with open("modelToLinesWithRefs.json", "r", encoding="utf-8") as file:
        modelToLinesWithRefs = json.load(file)
    trained_root = Path(TRAINED_ROOT)
    if not trained_root.is_dir():
        raise FileNotFoundError(
            f"Trained root directory does not exist: {trained_root}"
        )
    trained_directories = {
        path.name for path in trained_root.iterdir() if path.is_dir()
    }
    with open("modelsEval.json", "r", encoding="utf-8") as file:
        jsonObject = json.load(file)
        filtered_models = [model for model, value in jsonObject.items()]

    MODEL_FOLDER_NAME = "GeneralModelWithEveryone"  # Use "" to use each voice folder.
    fixed_model_folder_name = None
    fixed_model_folder_name = MODEL_FOLDER_NAME.strip()

    LINE_START = 0
    LINE_COUNT = 10
    for model_name, lines in modelToLinesWithRefs.items():
        # if "Ass_Elite_Insidious_Carlos" not in model_name:
        #    continue
        # if model_name in filtered_models:
        #    continue
        if len(lines) == 0:
            continue
        if model_name in trained_directories:
            continue
        model_folder_name = fixed_model_folder_name or model_name
        checkpoints_folder = trained_root / model_folder_name
        if not checkpoints_folder.is_dir():
            continue

        checkpoints = find_checkpoint_dirs(checkpoints_folder, CHECKPOINT_NAME.strip())
        for checkpoint_dir in checkpoints:
            if not (checkpoint_dir / "model.safetensors").is_file():
                continue
            pending_lines = []
            selected_lines = lines[LINE_START : LINE_START + LINE_COUNT]
            for index, line in enumerate(selected_lines, start=LINE_START):
                line["filename"] = (
                    f"{model_name}_{index + 1}_{checkpoint_dir.name.removeprefix('checkpoint-')}.wav"
                )
                output_path = GENERATED_ROOT / line["filename"]
                if output_path.exists():
                    continue
                pending_lines.append((line, output_path))
            if not pending_lines:
                continue
            model = None
            try:
                clear_device_cache(device)
                model = load_model_for_checkpoint(
                    checkpoint_dir,
                    device=device,
                    dtype=dtype,
                    CHECKPOINT_MODE=CHECKPOINT_MODE,
                )
                clear_device_cache(device)
                instruct = INSTRUCT.strip() or None
                language = LANGUAGE.strip() or None
                for line, output_path in pending_lines:
                    clear_device_cache(device)
                    text = line["line"]
                    refLine = line["simillar"]
                    ref_audio_path = resolve_case_insensitive_path(
                        f"SPEECH_MERGED/{refLine['filename']}"
                    )
                    print(f"Generating: {line!r}")
                    audio = model.generate(
                        text=text,
                        language=language,
                        instruct=instruct,
                        num_step=NUM_STEP,
                        speed=SPEED,
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
