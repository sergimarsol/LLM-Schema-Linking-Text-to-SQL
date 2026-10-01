#!/usr/bin/env python3
"""Targeted RapidFire recovery/improvement grid for Project 2.

This script is intentionally separate from rapidfire_train_8.py. The first grid
establishes the required broad experiment sweep. This follow-up grid spends time
on the observed weak families from validation: NTSB, NYSED_SRC2022, and
SBODemoUS modules. It oversamples those training rows and tries stronger variants
near the best run from the first grid.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List as TypingList

from rapidfire_train_8 import (
    create_model,
    load_json,
    make_rf_rows,
    schema_link_formatting_function,
)
from schema_linking_core import RetrievalConfig

TARGET_DBS = {"NTSB", "NYSED_SRC2022"}
TARGET_PREFIXES = ("SBODemoUS",)


def is_target_item(item: Dict[str, Any]) -> bool:
    db_id = str(item.get("db_id", ""))
    return db_id in TARGET_DBS or db_id.startswith(TARGET_PREFIXES)


def oversample_targets(items: TypingList[Dict[str, Any]], extra_copies: int, seed: int) -> TypingList[Dict[str, Any]]:
    rng = random.Random(seed)
    out: TypingList[Dict[str, Any]] = []
    for item in items:
        out.append(item)
        if is_target_item(item):
            out.extend([item] * max(0, extra_copies))
    rng.shuffle(out)
    return out


def retrieval_config(max_tables: int, max_cols: int) -> Dict[str, Any]:
    return RetrievalConfig(
        max_tables=max_tables,
        max_columns_per_table=max_cols,
        min_columns_per_table=4,
        include_types=True,
        include_keys=True,
        include_join_neighbors=True,
        join_neighbor_limit=3,
    ).to_dict()


def build_targeted_config_set(max_steps: int = 420, max_length: int = 2048):
    from rapidfireai.automl import List, RFModelConfig, RFLoraConfig, RFSFTConfig

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    common_kwargs = {"device_map": "auto", "torch_dtype": "auto", "trust_remote_code": True, "use_cache": False}

    def sft(lr: float, steps: int = max_steps, grad_accum: int = 8):
        return RFSFTConfig(
            learning_rate=lr,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            max_steps=steps,
            warmup_ratio=0.05,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=max(20, steps // 4),
            bf16=True,
            max_length=max_length,
            packing=False,
            report_to="none",
        )

    def lora(r: int, dropout: float):
        return RFLoraConfig(
            r=r,
            lora_alpha=2 * r,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    return List([
        RFModelConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            model_type="causal_lm",
            model_kwargs=common_kwargs,
            peft_config=lora(64, 0.05),
            training_args=sft(8e-5),
            formatting_func=schema_link_formatting_function,
        ),
        RFModelConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            model_type="causal_lm",
            model_kwargs=common_kwargs,
            peft_config=lora(64, 0.03),
            training_args=sft(1.2e-4),
            formatting_func=schema_link_formatting_function,
        ),
        RFModelConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            model_type="causal_lm",
            model_kwargs=common_kwargs,
            peft_config=lora(96, 0.05),
            training_args=sft(6e-5),
            formatting_func=schema_link_formatting_function,
        ),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.json")
    parser.add_argument("--validation", default="validation.json")
    parser.add_argument("--schemas_dir", default="schemas")
    parser.add_argument("--experiment_name", default="schema_linking_qwen_lora_targeted")
    parser.add_argument("--experiment_path", default="rf_experiments")
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_steps", type=int, default=420)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--target_extra_copies", type=int, default=3)
    parser.add_argument("--max_tables", type=int, default=16)
    parser.add_argument("--max_columns_per_table", type=int, default=24)
    args = parser.parse_args()

    from datasets import Dataset
    from rapidfireai import Experiment
    from rapidfireai.automl import RFGridSearch

    retrieval = retrieval_config(args.max_tables, args.max_columns_per_table)
    output_format = "ids"
    run_config = {
        "base_model": None,
        "output_format": output_format,
        "retrieval": retrieval,
        "note": (
            "Targeted recovery/improvement grid. Training rows from NTSB, "
            "NYSED_SRC2022, and SBODemoUS modules are oversampled; inference "
            "uses wider schema retrieval. Base model is read from adapter_config.json."
        ),
    }

    exp_dir = Path(args.experiment_path) / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / "default_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    train_items = oversample_targets(load_json(args.train), args.target_extra_copies, args.seed)
    val_items = load_json(args.validation)
    target_count = sum(1 for item in train_items if is_target_item(item))
    print(
        f"Training rows after oversampling: {len(train_items)} "
        f"({target_count} target-family rows including copies).",
        flush=True,
    )

    train_rows = make_rf_rows(train_items, args.schemas_dir, retrieval, output_format)
    val_rows = make_rf_rows(val_items, args.schemas_dir, retrieval, output_format)
    train_dataset = Dataset.from_list(train_rows).shuffle(seed=args.seed)
    eval_dataset = Dataset.from_list(val_rows)

    config_group = RFGridSearch(configs=build_targeted_config_set(args.max_steps, args.max_length), trainer_type="SFT")
    print(f"Targeted grid expands to {len(config_group.get_runs(seed=args.seed))} RapidFire runs.", flush=True)

    experiment = Experiment(
        experiment_name=args.experiment_name,
        mode="fit",
        experiment_path=args.experiment_path,
    )
    experiment.run_fit(
        config_group,
        create_model,
        train_dataset,
        eval_dataset,
        num_chunks=args.num_chunks,
        seed=args.seed,
        num_gpus=1,
    )

    print("Targeted RapidFire training complete.", flush=True)
    print(f"Default run config written to {exp_dir / 'default_run_config.json'}", flush=True)


if __name__ == "__main__":
    main()
