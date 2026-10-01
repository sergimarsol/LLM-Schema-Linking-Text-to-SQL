#!/usr/bin/env python3
"""Launch the 8 required SFT experiments with RapidFire AI RFGridSearch.

This script is for project-compliant RapidFire training/logging. It uses the
same candidate-schema prompt and ID-output contract as main.py.

After this finishes, run evaluate_experiment_grid.py on the RapidFire final
checkpoints to produce validation metrics and copy the best adapter to ./adapter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List as TypingList

from schema_linking_core import RetrievalConfig, make_example_payload


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def base_retrieval(max_tables: int = 10, max_cols: int = 18) -> Dict[str, Any]:
    return RetrievalConfig(
        max_tables=max_tables,
        max_columns_per_table=max_cols,
        min_columns_per_table=4,
        include_types=True,
        include_keys=True,
        include_join_neighbors=True,
        join_neighbor_limit=2,
    ).to_dict()


def make_rf_rows(items: TypingList[Dict[str, Any]], schemas_dir: str, retrieval: Dict[str, Any], output_format: str) -> TypingList[Dict[str, Any]]:
    cfg = RetrievalConfig.from_dict(retrieval)
    rows = []
    for item in items:
        payload = make_example_payload(
            item=item,
            schemas_dir=schemas_dir,
            retrieval_cfg=cfg,
            output_format=output_format,
            force_gold=True,
        )
        rows.append(
            {
                "question_id": payload["question_id"],
                "db_id": payload["db_id"],
                "system": payload["system"],
                "user": payload["user"],
                "assistant": payload["assistant"],
            }
        )
    return rows


def schema_link_formatting_function(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["user"]},
        ],
        "completion": [
            {"role": "assistant", "content": row["assistant"]},
        ],
    }


def create_model(model_config: Dict[str, Any]):
    """RapidFire create_model_fn. Must return (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_config["model_name"]
    model_kwargs = dict(model_config.get("model_kwargs") or {})
    if model_kwargs.get("torch_dtype") == "auto" and torch.cuda.is_available():
        # Keeping "auto" also works for Qwen, but this removes ambiguity.
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    return model, tokenizer


def build_config_set(max_steps_short: int = 180, max_steps_med: int = 260, max_steps_long: int = 340):
    from rapidfireai.automl import List, RFModelConfig, RFLoraConfig, RFSFTConfig

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def sft(lr: float, steps: int, grad_accum: int = 8):
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
            max_length=2048,
            packing=False,
            report_to="none",
        )

    def lora(r: int, dropout: float = 0.05):
        return RFLoraConfig(
            r=r,
            lora_alpha=2 * r,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    common_kwargs = {"device_map": "auto", "torch_dtype": "auto", "trust_remote_code": True, "use_cache": False}

    configs = List([
        RFModelConfig(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            model_type="causal_lm",
            model_kwargs=common_kwargs,
            peft_config=lora(16, 0.05),
            training_args=sft(2e-4, max_steps_short),
            formatting_func=schema_link_formatting_function,
        ),
        RFModelConfig(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            model_type="causal_lm",
            model_kwargs=common_kwargs,
            peft_config=lora(32, 0.05),
            training_args=sft(1e-4, max_steps_med),
            formatting_func=schema_link_formatting_function,
        ),
        RFModelConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            model_type="causal_lm",
            model_kwargs=common_kwargs,
            peft_config=lora(16, 0.05),
            training_args=sft(1e-4, max_steps_short),
            formatting_func=schema_link_formatting_function,
        ),
        RFModelConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            model_type="causal_lm",
            model_kwargs=common_kwargs,
            peft_config=lora(32, 0.05),
            training_args=sft(1e-4, max_steps_med),
            formatting_func=schema_link_formatting_function,
        ),
        RFModelConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            model_type="causal_lm",
            model_kwargs=common_kwargs,
            peft_config=lora(32, 0.05),
            training_args=sft(2e-4, max_steps_med),
            formatting_func=schema_link_formatting_function,
        ),
        RFModelConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            model_type="causal_lm",
            model_kwargs=common_kwargs,
            peft_config=lora(64, 0.05),
            training_args=sft(8e-5, max_steps_long),
            formatting_func=schema_link_formatting_function,
        ),
        RFModelConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            model_type="causal_lm",
            model_kwargs=common_kwargs,
            peft_config=lora(32, 0.03),
            training_args=sft(5e-5, max_steps_long),
            formatting_func=schema_link_formatting_function,
        ),
        RFModelConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            model_type="causal_lm",
            model_kwargs=common_kwargs,
            peft_config=lora(48, 0.05),
            training_args=sft(1.5e-4, max_steps_long),
            formatting_func=schema_link_formatting_function,
        ),
    ])
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.json")
    parser.add_argument("--validation", default="validation.json")
    parser.add_argument("--schemas_dir", default="schemas")
    parser.add_argument("--experiment_name", default="schema_linking_qwen_lora8")
    parser.add_argument("--experiment_path", default="rf_experiments")
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps_short", type=int, default=180)
    parser.add_argument("--max_steps_med", type=int, default=260)
    parser.add_argument("--max_steps_long", type=int, default=340)
    args = parser.parse_args()

    from datasets import Dataset
    from rapidfireai import Experiment
    from rapidfireai.automl import RFGridSearch

    retrieval = base_retrieval(12, 20)
    output_format = "ids"
    run_config = {
        "base_model": None,
        "output_format": output_format,
        "retrieval": retrieval,
        "note": "RapidFire run config. Base model is read from adapter_config.json for each checkpoint.",
    }
    exp_dir = Path(args.experiment_path) / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / "default_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    train_rows = make_rf_rows(load_json(args.train), args.schemas_dir, retrieval, output_format)
    val_rows = make_rf_rows(load_json(args.validation), args.schemas_dir, retrieval, output_format)
    train_dataset = Dataset.from_list(train_rows).shuffle(seed=args.seed)
    eval_dataset = Dataset.from_list(val_rows)

    config_set = build_config_set(args.max_steps_short, args.max_steps_med, args.max_steps_long)
    config_group = RFGridSearch(configs=config_set, trainer_type="SFT")

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

    print("RapidFire training complete.")
    print(f"Default run config written to {exp_dir / 'default_run_config.json'}")
    print("Next: python evaluate_experiment_grid.py --mode rapidfire --experiment_dir " + str(exp_dir))


if __name__ == "__main__":
    main()
