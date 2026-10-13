#!/usr/bin/env python3
"""Prepare TXT/WAV pairs and fine-tune OmniVoice.

Edit the constants near the top of this file, then run:

    python examples/train_custom_model.py

This script intentionally has no command-line arguments.
"""

import json
import os
import random
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

from safetensors import safe_open

# None means train every top-level folder under data/ that contains raws/.
# To train only selected voices, set this to a list like ["Orc_Warrior_Bearish_Barrack"].
MODEL_NAMES: list[str] | None = None


def discover_model_names(project_root: Path) -> list[str]:
    data_root = project_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_root}")

    model_names = [
        path.name
        for path in sorted(data_root.iterdir())
        if path.is_dir() and (path / "raws").is_dir()
    ]
    if not model_names:
        raise FileNotFoundError(
            f"No model folders with a raws/ directory found under {data_root}"
        )
    return model_names


def main(
    model_name: str, TRAINING_STEPS=900, SAVE_STEPS=300, EVAL_STEPS=25, USE_LORA=True
) -> None:

    # -----------------------------------------------------------------------------
    # Your training files
    # -----------------------------------------------------------------------------
    # Put your files in matching TXT/WAV pairs. The simplest layout is:
    #
    # data/my_polish_voice/raw/
    #   line_0001.txt   # contains the exact transcript for line_0001.wav
    #   line_0001.wav   # contains only that spoken line
    #   line_0002.txt
    #   line_0002.wav
    #
    # The matching rule is: same relative path and same filename stem.
    # Subfolders are OK, for example:
    #
    # data/my_polish_voice/texts/chapter_1/line_0001.txt
    # data/my_polish_voice/wavs/chapter_1/line_0001.wav
    #
    # Each .txt file should contain only the words spoken in its matching .wav file.
    # Use short, clean clips when possible. A few seconds per clip is ideal; very long
    # files make training slower and can reduce quality. WAV files can be any common
    # sample rate because OmniVoice resamples them during tokenization.
    #
    # The script writes JSONL rows like:
    # {"id": "line_0001", "audio_path": "/abs/path/line_0001.wav", "text": "...", "language_id": "pl"}

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # If your TXT and WAV files are in one folder, point both constants to that folder.
    MODEL_NAME = model_name
    TEXT_DIR = PROJECT_ROOT / f"data/{MODEL_NAME}/raws"
    AUDIO_DIR = PROJECT_ROOT / f"data/{MODEL_NAME}/raws"

    LANGUAGE_ID = "pl"
    DEV_SPLIT_RATIO = 0.05
    RANDOM_SEED = 42

    # -----------------------------------------------------------------------------
    # Model/training settings
    # -----------------------------------------------------------------------------
    BASE_MODEL = "k2-fsa/OmniVoice"
    AUDIO_TOKENIZER = "eustlb/higgs-audio-v2-tokenizer"

    WORK_DIR = PROJECT_ROOT / f"data/{MODEL_NAME}"
    MANIFEST_DIR = WORK_DIR / "manifests"
    TOKEN_DIR = WORK_DIR / "tokens"
    CONFIG_DIR = WORK_DIR / "config"
    OUTPUT_DIR = PROJECT_ROOT / f"trained/{MODEL_NAME}"

    GPU_IDS = "0"
    NUM_GPUS = 1

    # Turn individual stages off after they have already succeeded.
    BUILD_JSONL_MANIFESTS = True
    TOKENIZE_AUDIO = True
    TRAIN_MODEL = True

    # LoRA is the practical default for a custom voice or small dataset. Set
    # USE_LORA = False only if you really intend to full-fine-tune the model.

    BATCH_TOKENS = 2048
    GRADIENT_ACCUMULATION_STEPS = 2
    TRAIN_DATALOADER_WORKERS = 0
    MIXED_PRECISION = "bf16"
    ATTN_IMPLEMENTATION = "sdpa"

    # Tokenization settings. Increase workers if you have more GPU memory/CPU room.
    NJ_PER_GPU = 1
    LOADER_WORKERS = 4
    SAMPLES_PER_SHARD = 1000
    MIN_NUM_SHARDS = 1

    def get_checkpoint_step(checkpoint_dir: Path) -> int | None:
        if not checkpoint_dir.is_dir() or not checkpoint_dir.name.startswith(
            "checkpoint-"
        ):
            return None
        try:
            return int(checkpoint_dir.name.removeprefix("checkpoint-"))
        except ValueError:
            return None

    def validate_safetensors_file(path: Path) -> str | None:
        try:
            with safe_open(path, framework="pt", device="cpu") as safetensors_file:
                list(safetensors_file.keys())
        except Exception as exc:
            return f"{path.name} is not readable: {exc}"
        return None

    def get_checkpoint_problem(checkpoint_dir: Path) -> str | None:
        required_files = [
            "model.safetensors",
            "optimizer.bin",
            "scheduler.bin",
        ]
        if USE_LORA:
            required_files.extend(["adapter_config.json", "adapter_model.safetensors"])

        for filename in required_files:
            path = checkpoint_dir / filename
            if not path.is_file():
                return f"missing {filename}"
            if path.stat().st_size == 0:
                return f"{filename} is empty"

        if not any(checkpoint_dir.glob("random_states_*.pkl")):
            return "missing random_states_*.pkl"

        for safetensors_path in checkpoint_dir.glob("*.safetensors"):
            problem = validate_safetensors_file(safetensors_path)
            if problem is not None:
                return problem

        return None

    def find_latest_checkpoint(
        output_dir: Path, max_step: int
    ) -> tuple[Path, int] | None:
        if not output_dir.is_dir():
            return None

        checkpoints = []
        for checkpoint_dir in output_dir.iterdir():
            step = get_checkpoint_step(checkpoint_dir)
            if step is not None and step <= max_step:
                checkpoints.append((step, checkpoint_dir))

        if not checkpoints:
            return None

        for step, checkpoint_dir in sorted(checkpoints, reverse=True):
            problem = get_checkpoint_problem(checkpoint_dir)
            if problem is None:
                return checkpoint_dir, step
            print(f"Skipping checkpoint {checkpoint_dir}: {problem}")

        return None

    def read_transcript(path: Path) -> str:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Transcript is empty: {path}")
        return " ".join(text.split())

    def find_audio_for_text(text_path: Path) -> Path:
        relative_path = text_path.relative_to(TEXT_DIR)
        for extension in (".wav", ".flac", ".mp3"):
            audio_path = (AUDIO_DIR / relative_path).with_suffix(extension)
            if audio_path.is_file():
                return audio_path
        raise FileNotFoundError(
            f"No matching audio file found for {text_path}. Expected the same "
            f"relative path under {AUDIO_DIR} with .wav, .flac, or .mp3."
        )

    def collect_records() -> list[dict[str, str]]:
        if not TEXT_DIR.is_dir():
            raise FileNotFoundError(f"TEXT_DIR does not exist: {TEXT_DIR}")
        if not AUDIO_DIR.is_dir():
            raise FileNotFoundError(f"AUDIO_DIR does not exist: {AUDIO_DIR}")

        text_paths = sorted(TEXT_DIR.rglob("*.txt"))
        if not text_paths:
            raise FileNotFoundError(f"No .txt files found under {TEXT_DIR}")

        records = []
        used_ids = set()
        for text_path in text_paths:
            audio_path = find_audio_for_text(text_path)
            sample_id = text_path.relative_to(TEXT_DIR).with_suffix("").as_posix()
            sample_id = sample_id.replace("/", "_")
            if sample_id in used_ids:
                raise ValueError(
                    f"Duplicate sample id after path normalization: {sample_id}"
                )
            used_ids.add(sample_id)
            records.append(
                {
                    "id": sample_id,
                    "audio_path": str(audio_path.resolve()),
                    "text": read_transcript(text_path),
                    "language_id": LANGUAGE_ID,
                }
            )
        return records

    def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def build_jsonl_manifests() -> tuple[Path, Path | None]:
        records = collect_records()
        random.Random(RANDOM_SEED).shuffle(records)

        if len(records) == 1:
            train_records = records
            dev_records = []
        else:
            dev_count = max(1, int(len(records) * DEV_SPLIT_RATIO))
            dev_count = min(dev_count, len(records) - 1)
            dev_records = records[:dev_count]
            train_records = records[dev_count:]

        train_jsonl = MANIFEST_DIR / "train.jsonl"
        dev_jsonl = MANIFEST_DIR / "dev.jsonl"
        write_jsonl(train_jsonl, train_records)
        if dev_records:
            write_jsonl(dev_jsonl, dev_records)
            dev_path = dev_jsonl
        else:
            dev_path = None

        print(f"Wrote {len(train_records)} train rows to {train_jsonl}")
        if dev_path is not None:
            print(f"Wrote {len(dev_records)} dev rows to {dev_path}")
        else:
            print("No dev set was created because there is only one sample.")
        return train_jsonl, dev_path

    def run_command(command: list[str], env: dict[str, str] | None = None) -> None:
        print("Running:", " ".join(command))
        subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)

    def require_file(path: Path, stage_name: str, flag_name: str) -> Path:
        if not path.is_file():
            raise FileNotFoundError(
                f"{stage_name} output is missing: {path}. Set {flag_name} = True "
                "and rerun this script to create it."
            )
        return path

    def check_training_dependencies() -> None:
        if USE_LORA and find_spec("peft") is None:
            raise RuntimeError(
                "USE_LORA is True, but the 'peft' package is not installed. "
                "Install it in this uv environment with: uv sync --extra lora"
            )

    def tokenize_split(split_name: str, input_jsonl: Path) -> Path:
        output_base = TOKEN_DIR / split_name
        command = [
            sys.executable,
            "-m",
            "omnivoice.scripts.extract_audio_tokens",
            "--input_jsonl",
            str(input_jsonl),
            "--tar_output_pattern",
            str(output_base / "audios/shard-%06d.tar"),
            "--jsonl_output_pattern",
            str(output_base / "txts/shard-%06d.jsonl"),
            "--tokenizer_path",
            AUDIO_TOKENIZER,
            "--nj_per_gpu",
            str(NJ_PER_GPU),
            "--loader_workers",
            str(LOADER_WORKERS),
            "--samples_per_shard",
            str(SAMPLES_PER_SHARD),
            "--min_num_shards",
            str(MIN_NUM_SHARDS),
            "--shuffle",
            "True",
            "--skip_errors",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = GPU_IDS
        run_command(command, env=env)
        return output_base / "data.lst"

    def write_configs(
        train_manifest: Path,
        dev_manifest: Path | None,
        resume_from_checkpoint: Path | None,
    ) -> tuple[Path, Path]:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        train_config_path = CONFIG_DIR / "train_config.json"
        data_config_path = CONFIG_DIR / "data_config.json"

        train_config = {
            "llm_name_or_path": "Qwen/Qwen3-0.6B",
            "audio_vocab_size": 1025,
            "audio_mask_id": 1024,
            "num_audio_codebook": 8,
            "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
            "drop_cond_ratio": 0.1,
            "prompt_ratio_range": [0.0, 0.3],
            "mask_ratio_range": [0.0, 1.0],
            "language_ratio": 0.8,
            "use_pinyin_ratio": 0.0,
            "instruct_ratio": 0.0,
            "only_instruct_ratio": 0.0,
            "resume_from_checkpoint": (
                str(resume_from_checkpoint.resolve())
                if resume_from_checkpoint is not None
                else None
            ),
            "init_from_checkpoint": BASE_MODEL,
            "use_lora": USE_LORA,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_bias": "none",
            "lora_target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "lora_modules_to_save": ["audio_embeddings", "audio_heads"],
            "learning_rate": 1e-4 if USE_LORA else 1e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "steps": TRAINING_STEPS,
            "seed": RANDOM_SEED,
            "warmup_type": "ratio",
            "warmup_ratio": 0.01,
            "warmup_steps": 0,
            "attn_implementation": ATTN_IMPLEMENTATION,
            "max_sample_tokens": 2000,
            "min_sample_tokens": 50,
            "max_batch_size": 64,
            "batch_tokens": BATCH_TOKENS,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "num_workers": TRAIN_DATALOADER_WORKERS,
            "mixed_precision": MIXED_PRECISION,
            "allow_tf32": True,
            "logging_steps": 50,
            "eval_steps": EVAL_STEPS,
            "save_steps": SAVE_STEPS,
            "keep_last_n_checkpoints": -1,
        }
        data_config = {
            "train": [
                {
                    "language_id": LANGUAGE_ID,
                    "manifest_path": [str(train_manifest.resolve())],
                    "repeat": 1,
                }
            ]
        }
        if dev_manifest is not None:
            data_config["dev"] = [
                {
                    "language_id": LANGUAGE_ID,
                    "manifest_path": [str(dev_manifest.resolve())],
                    "repeat": 1,
                }
            ]

        train_config_path.write_text(
            json.dumps(train_config, indent=4) + "\n", encoding="utf-8"
        )
        data_config_path.write_text(
            json.dumps(data_config, indent=4) + "\n", encoding="utf-8"
        )
        print(f"Wrote training config to {train_config_path}")
        print(f"Wrote data config to {data_config_path}")
        return train_config_path, data_config_path

    def train(train_config_path: Path, data_config_path: Path) -> None:
        command = [
            "accelerate",
            "launch",
            "--gpu_ids",
            GPU_IDS,
            "--num_processes",
            str(NUM_GPUS),
            "-m",
            "omnivoice.cli.train",
            "--train_config",
            str(train_config_path),
            "--data_config",
            str(data_config_path),
            "--output_dir",
            str(OUTPUT_DIR),
        ]
        run_command(command)

    train_jsonl = MANIFEST_DIR / "train.jsonl"
    dev_jsonl = MANIFEST_DIR / "dev.jsonl"

    latest_checkpoint = find_latest_checkpoint(OUTPUT_DIR, TRAINING_STEPS)
    if latest_checkpoint is not None:
        resume_checkpoint, checkpoint_step = latest_checkpoint
        if checkpoint_step == TRAINING_STEPS:
            print(
                f"Skipping {MODEL_NAME}: latest checkpoint {resume_checkpoint} "
                f"already reached {checkpoint_step}/{TRAINING_STEPS} steps."
            )
            return
        print(
            f"Resuming {MODEL_NAME} from {resume_checkpoint} "
            f"at step {checkpoint_step}/{TRAINING_STEPS}."
        )
    else:
        resume_checkpoint = None

    if BUILD_JSONL_MANIFESTS:
        train_jsonl, dev_jsonl_or_none = build_jsonl_manifests()
    else:
        train_jsonl = require_file(
            train_jsonl, "JSONL manifest", "BUILD_JSONL_MANIFESTS"
        )
        dev_jsonl_or_none = dev_jsonl if dev_jsonl.is_file() else None

    train_manifest = TOKEN_DIR / "train/data.lst"
    dev_manifest = TOKEN_DIR / "dev/data.lst"

    if TOKENIZE_AUDIO:
        train_manifest = tokenize_split("train", train_jsonl)
        if dev_jsonl_or_none is not None:
            dev_manifest_or_none = tokenize_split("dev", dev_jsonl_or_none)
        else:
            dev_manifest_or_none = None
    else:
        train_manifest = require_file(
            train_manifest, "Tokenized train manifest", "TOKENIZE_AUDIO"
        )
        dev_manifest_or_none = dev_manifest if dev_manifest.is_file() else None

    train_config_path, data_config_path = write_configs(
        train_manifest=train_manifest,
        dev_manifest=dev_manifest_or_none,
        resume_from_checkpoint=resume_checkpoint,
    )

    if TRAIN_MODEL:
        check_training_dependencies()
        train(train_config_path, data_config_path)
    else:
        print("TRAIN_MODEL is False, so training was not launched.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    model_names = (
        MODEL_NAMES if MODEL_NAMES is not None else discover_model_names(project_root)
    )
    with open("modelsEval.json", "r", encoding="utf-8") as file:
        evaled_models = json.load(file)
    with open("modelsEvalGeneralOrUnclear.json", "r", encoding="utf-8") as file:
        do_not_touch_models = json.load(file)
    for model_index, model_name in enumerate(model_names, start=1):
        if model_name == "GeneralModelWithEveryone":
            # continue
            main(
                model_name,
                TRAINING_STEPS=45000,
                SAVE_STEPS=1000,
                EVAL_STEPS=500,
                USE_LORA=False,
            )
            break
        else:
            continue
        if model_name == "GeneralModel" or model_name in evaled_models:
            continue
        print(f"\n=== Training {model_name} ({model_index}/{len(model_names)}) ===")
        if model_name in evaled_models:
            desired_steps = sorted(evaled_models[model_name])
            for steps in desired_steps:
                main(
                    model_name,
                    TRAINING_STEPS=steps,
                    SAVE_STEPS=9999999999,
                    EVAL_STEPS=9999999,
                )
        else:
            main(model_name)
