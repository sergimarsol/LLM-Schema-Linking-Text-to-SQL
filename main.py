#!/usr/bin/env python3
"""Final inference entrypoint for CSE/DSC 234 Project 2.

Required grader contract:
    python3 main.py --input input_filename --output output_filename

The LoRA adapter weights (adapter_model.safetensors) are stored on Google Drive
because the file is too large for git. This script downloads them automatically
on first run if they are not present. All other adapter files (adapter_config.json,
run_config.json, tokenizer files) are committed to the repo in ./adapter/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema_linking_core import (
    RetrievalConfig,
    SYSTEM_PROMPT_IDS,
    SYSTEM_PROMPT_NAMES,
    build_candidate_schema,
    build_prompt_text,
    build_user_content,
    load_raw_schema,
    load_run_config,
    load_schema_as_dict,
    parse_prediction_to_schema_links,
)

DEFAULT_BASE_MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_DIR   = "./adapter"
DEFAULT_SCHEMAS_DIR   = "./schemas"
DEFAULT_MAX_NEW_TOKENS = 160

# Google Drive file ID for adapter_model.safetensors
# Source: https://drive.google.com/file/d/14Onm9xnTDMaPOGjPUPyz57JX-xCElO5Y/view
GDRIVE_FILE_ID = "14Onm9xnTDMaPOGjPUPyz57JX-xCElO5Y"
ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"


# ── Google Drive download ──────────────────────────────────────────────────────

def download_adapter_weights(adapter_dir: str) -> None:
    """Download adapter_model.safetensors from Google Drive if not present."""
    dest = Path(adapter_dir) / ADAPTER_WEIGHTS_FILENAME
    if dest.exists():
        print(f"Adapter weights already present at {dest}", flush=True)
        return

    print(f"Adapter weights not found at {dest}. Downloading from Google Drive...", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Try gdown first (clean, handles large file confirm pages automatically)
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, str(dest), quiet=False)
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"Downloaded adapter weights to {dest} ({dest.stat().st_size // 1_000_000} MB)", flush=True)
            return
        else:
            print("gdown download produced an empty or tiny file, trying fallback...", flush=True)
            dest.unlink(missing_ok=True)
    except ImportError:
        print("gdown not installed, trying pip install...", flush=True)
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, str(dest), quiet=False)
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"Downloaded adapter weights ({dest.stat().st_size // 1_000_000} MB)", flush=True)
            return
        dest.unlink(missing_ok=True)
    except Exception as e:
        print(f"gdown failed: {e}. Trying requests fallback...", flush=True)

    # Fallback: requests with manual cookie handling for large files
    try:
        import requests
        session = requests.Session()
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        response = session.get(url, stream=True)

        # Handle Google's virus-scan warning page for large files
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break
        # Also check response text for confirmation token
        if token is None and b"confirm=" in response.content[:2000]:
            import re
            m = re.search(rb"confirm=([0-9A-Za-z_\-]+)", response.content[:2000])
            if m:
                token = m.group(1).decode()

        if token:
            url = f"https://drive.google.com/uc?export=download&confirm={token}&id={GDRIVE_FILE_ID}"
            response = session.get(url, stream=True)

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        size_mb = dest.stat().st_size // 1_000_000
        if size_mb < 1:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded file is too small ({dest.stat().st_size} bytes). "
                               "The Google Drive link may require additional permissions.")
        print(f"Downloaded adapter weights to {dest} ({size_mb} MB)", flush=True)

    except Exception as e:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download adapter weights from Google Drive.\n"
            f"Error: {e}\n"
            f"Manual fix: download https://drive.google.com/uc?id={GDRIVE_FILE_ID} "
            f"and place it at {dest}"
        ) from e


# ── Model loading ──────────────────────────────────────────────────────────────

def read_base_model_from_adapter(adapter_dir: str) -> str | None:
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
        value = cfg.get("base_model_name_or_path")
        return str(value) if value else None
    except Exception:
        return None


def tokenizer_source(adapter_dir: str, base_model: str) -> str:
    adapter = Path(adapter_dir)
    tokenizer_files = ["tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt"]
    if any((adapter / name).exists() for name in tokenizer_files):
        return str(adapter)
    return base_model


def resolve_dtype(dtype_name: str) -> Any:
    norm = dtype_name.lower()
    if norm == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if norm in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if norm in {"fp16", "float16", "half"}:
        return torch.float16
    if norm in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(
    args: argparse.Namespace, run_cfg: Dict[str, Any]
) -> Tuple[Any, Any, str]:
    adapter_dir = Path(args.adapter_dir)
    adapter_cfg = adapter_dir / "adapter_config.json"
    if not adapter_cfg.exists():
        raise FileNotFoundError(
            f"No LoRA adapter config found at {adapter_cfg}. "
            "Make sure ./adapter/ is committed to the repo."
        )

    # Download weights if missing (e.g. fresh clone where safetensors not in git)
    download_adapter_weights(str(adapter_dir))

    base_model = (
        args.base_model
        or run_cfg.get("base_model")
        or read_base_model_from_adapter(str(adapter_dir))
        or DEFAULT_BASE_MODEL
    )
    tok_src = tokenizer_source(str(adapter_dir), base_model)

    print(f"Loading tokenizer: {tok_src}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    print(f"Loading base model: {base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=args.device_map,
        torch_dtype=resolve_dtype(args.torch_dtype),
        trust_remote_code=True,
        use_cache=True,
    )

    print(f"Loading LoRA adapter: {adapter_dir}", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer, base_model


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_one(
    question: str,
    db_id: str,
    schemas_dir: str,
    model: Any,
    tokenizer: Any,
    retrieval_cfg: RetrievalConfig,
    output_format: str,
    max_new_tokens: int,
    max_input_length: int,
) -> Dict[str, List[str]]:
    raw_schema = load_raw_schema(db_id, schemas_dir)
    schema = load_schema_as_dict(db_id, schemas_dir)
    candidates, table_id, column_id = build_candidate_schema(
        question=question,
        raw_schema=raw_schema,
        cfg=retrieval_cfg,
        force_links=None,
    )
    system_prompt = SYSTEM_PROMPT_IDS if output_format == "ids" else SYSTEM_PROMPT_NAMES
    user_content = build_user_content(
        question, candidates, table_id, column_id, retrieval_cfg, output_format
    )
    prompt = build_prompt_text(system_prompt, user_content, tokenizer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_prediction_to_schema_links(
        response, schema, candidates, table_id, column_id, output_format
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Schema linking inference — CSE/DSC 234 Project 2"
    )
    parser.add_argument("--input",           required=True,
                        help="Input JSON (list of {question_id, db_id, question})")
    parser.add_argument("--output",          required=True,
                        help="Output JSON (list of {question_id, schema_links})")
    parser.add_argument("--schemas_dir",     default=DEFAULT_SCHEMAS_DIR)
    parser.add_argument("--adapter_dir",     default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--base_model",      default=None)
    parser.add_argument("--max_new_tokens",  type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max_input_length",type=int, default=None)
    parser.add_argument("--torch_dtype",     default="auto")
    parser.add_argument("--device_map",      default="auto")
    args = parser.parse_args()

    run_cfg = load_run_config(args.adapter_dir)
    output_format = str(run_cfg.get("output_format", "ids")).lower()
    if output_format not in {"ids", "names"}:
        raise ValueError(f"Unsupported output_format in run_config.json: {output_format}")
    retrieval_cfg  = RetrievalConfig.from_dict(run_cfg.get("retrieval"))
    max_input_length = int(args.max_input_length or run_cfg.get("max_input_length", 2048))

    with open(args.input, encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list):
        raise ValueError("Input file must be a JSON list of question objects.")

    print(f"Loaded {len(questions)} questions.", flush=True)
    print(f"Output format: {output_format}", flush=True)
    print(f"Max input length: {max_input_length}", flush=True)

    model, tokenizer, base_model = load_model_and_tokenizer(args, run_cfg)
    print(f"Using base model: {base_model}", flush=True)

    predictions = []
    for i, item in enumerate(questions, start=1):
        links = predict_one(
            question=str(item["question"]),
            db_id=str(item["db_id"]),
            schemas_dir=args.schemas_dir,
            model=model,
            tokenizer=tokenizer,
            retrieval_cfg=retrieval_cfg,
            output_format=output_format,
            max_new_tokens=args.max_new_tokens,
            max_input_length=max_input_length,
        )
        predictions.append({"question_id": item["question_id"], "schema_links": links})
        print(f"[{i}/{len(questions)}] q{item['question_id']} -> {links}", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(predictions)} predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()