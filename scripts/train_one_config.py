#!/usr/bin/env python3
"""Train one Qwen LoRA schema-linking adapter.

This is the reliable single-run trainer used by run_8_configs.py. It avoids
heuristic answer generation and trains the adapter to emit the same structured
format that main.py expects.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch.utils.data import Dataset

from schema_linking_core import (
    RetrievalConfig,
    build_prompt_text,
    load_raw_schema,
    make_example_payload,
    write_run_config,
)


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | os.PathLike[str], obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def maybe_balance_items(items: List[Dict[str, Any]], seed: int, enabled: bool) -> List[Dict[str, Any]]:
    if not enabled:
        return list(items)
    rng = random.Random(seed)
    by_db: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        by_db.setdefault(str(item["db_id"]), []).append(item)

    # Oversample small database modules so SAP submodules do not get drowned by NTSB/NYSED.
    target_per_db = max(18, int(sum(len(v) for v in by_db.values()) / max(len(by_db), 1)))
    balanced: List[Dict[str, Any]] = []
    for db_id, rows in sorted(by_db.items()):
        balanced.extend(rows)
        if len(rows) < target_per_db:
            balanced.extend(rng.choices(rows, k=target_per_db - len(rows)))

    # Multi-table questions are the ones current runs miss most often. Give them one extra pass.
    for item in items:
        if len(item.get("schema_links", {})) >= 2:
            balanced.append(item)

    rng.shuffle(balanced)
    return balanced


class SchemaLinkDataset(Dataset):
    def __init__(
        self,
        items: List[Mapping[str, Any]],
        tokenizer: Any,
        schemas_dir: str,
        retrieval_cfg: RetrievalConfig,
        output_format: str,
        max_seq_length: int,
        force_gold_schema_in_prompt: bool,
    ) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        eos = tokenizer.eos_token or "<|endoftext|>"

        too_long = 0
        for item in items:
            payload = make_example_payload(
                item=item,
                schemas_dir=schemas_dir,
                retrieval_cfg=retrieval_cfg,
                output_format=output_format,
                force_gold=force_gold_schema_in_prompt,
            )
            prompt = build_prompt_text(payload["system"], payload["user"], tokenizer)
            answer = payload["assistant"] + eos

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            if len(answer_ids) >= max_seq_length:
                # This should never happen for schema links, but keep the trainer safe.
                answer_ids = answer_ids[: max_seq_length - 1] + [tokenizer.eos_token_id]
            max_prompt = max_seq_length - len(answer_ids)
            if len(prompt_ids) > max_prompt:
                too_long += 1
                # Keep the end of the prompt, where candidate schema and question live.
                prompt_ids = prompt_ids[-max_prompt:]
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            self.rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )
        if too_long:
            print(f"Warning: truncated {too_long}/{len(items)} prompts to max_seq_length={max_seq_length}", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.rows[idx]


class DataCollatorForCausalSchemaLinks:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def training_args_kwargs(config: Mapping[str, Any], output_dir: str) -> Dict[str, Any]:
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": float(config.get("num_train_epochs", 8)),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(config.get("learning_rate", 1e-4)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.05)),
        "lr_scheduler_type": str(config.get("lr_scheduler_type", "cosine")),
        "logging_steps": int(config.get("logging_steps", 5)),
        "save_strategy": "no",
        "report_to": str(config.get("report_to", "none")),
        "remove_unused_columns": False,
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "optim": str(config.get("optim", "adamw_torch")),
        "bf16": bool(config.get("bf16", torch.cuda.is_available())),
        "fp16": bool(config.get("fp16", False)),
        "seed": int(config.get("seed", 42)),
        "dataloader_pin_memory": False,
    }
    if config.get("max_steps") is not None:
        kwargs["max_steps"] = int(config["max_steps"])
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = str(config.get("eval_strategy", "epoch"))
    else:
        kwargs["evaluation_strategy"] = str(config.get("eval_strategy", "epoch"))
    if kwargs.get("eval_strategy") == "steps" or kwargs.get("evaluation_strategy") == "steps":
        kwargs["eval_steps"] = int(config.get("eval_steps", 20))
    return kwargs


def load_model_tokenizer(config: Mapping[str, Any]) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    base_model = str(config["base_model"])
    qlora = bool(config.get("qlora", False))
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "use_cache": False,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if qlora:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError("qlora=True requires bitsandbytes and a recent transformers install.") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if qlora:
        model = prepare_model_for_kbit_training(model)
    elif bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    init_adapter = config.get("init_adapter")
    if init_adapter:
        print(f"Continuing from adapter: {init_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
    else:
        target_modules = config.get("target_modules") or [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ]
        peft_cfg = LoraConfig(
            r=int(config.get("lora_r", 32)),
            lora_alpha=int(config.get("lora_alpha", int(config.get("lora_r", 32)) * 2)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(target_modules),
        )
        model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to one run config JSON.")
    parser.add_argument("--train", default="train.json")
    parser.add_argument("--validation", default="validation.json")
    parser.add_argument("--schemas_dir", default="schemas")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "config.json", config)

    train_items = load_json(args.train)
    valid_items = load_json(args.validation)
    if not isinstance(train_items, list) or not isinstance(valid_items, list):
        raise ValueError("train and validation files must be JSON lists.")
    train_items = maybe_balance_items(train_items, seed=seed, enabled=bool(config.get("balance_training", False)))

    retrieval_cfg = RetrievalConfig.from_dict(config.get("retrieval"))
    output_format = str(config.get("output_format", "ids")).lower()
    if output_format not in {"ids", "names"}:
        raise ValueError("output_format must be 'ids' or 'names'.")

    print("Loading model/tokenizer...", flush=True)
    model, tokenizer = load_model_tokenizer(config)

    print("Building train/eval datasets...", flush=True)
    train_ds = SchemaLinkDataset(
        train_items,
        tokenizer,
        args.schemas_dir,
        retrieval_cfg,
        output_format,
        int(config.get("max_seq_length", 2048)),
        force_gold_schema_in_prompt=True,
    )
    eval_ds = SchemaLinkDataset(
        valid_items,
        tokenizer,
        args.schemas_dir,
        retrieval_cfg,
        output_format,
        int(config.get("max_seq_length", 2048)),
        force_gold_schema_in_prompt=True,
    )
    print(f"Train examples after balancing: {len(train_ds)}", flush=True)
    print(f"Eval examples: {len(eval_ds)}", flush=True)

    from transformers import Trainer, TrainingArguments

    targs = TrainingArguments(**training_args_kwargs(config, str(output_dir / "trainer")))
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForCausalSchemaLinks(tokenizer),
    )
    trainer.train()

    print(f"Saving adapter to {adapter_dir}", flush=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    write_run_config(
        adapter_dir,
        {
            "base_model": config["base_model"],
            "output_format": output_format,
            "retrieval": retrieval_cfg.to_dict(),
            "max_input_length": int(config.get("max_input_length", config.get("max_seq_length", 2048))),
            "train_config": config,
        },
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
