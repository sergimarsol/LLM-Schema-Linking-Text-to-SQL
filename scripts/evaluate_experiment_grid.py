#!/usr/bin/env python3
"""Evaluate trained adapters and copy the best one to ./adapter.

Supports both:
  --mode local      adapters from run_8_configs.py: runs_lora8/*/adapter
  --mode rapidfire  adapters from RapidFire: rf_experiments/<exp>/runs/*/checkpoints/final_checkpoint
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from schema_linking_core import RetrievalConfig, write_run_config

SCORE_RE = re.compile(r"Leaderboard Score\s*:\s*([0-9.]+)")
METRIC_RES = {
    "precision_t": re.compile(r"Precision_T\s*:\s*([0-9.]+)"),
    "recall_t": re.compile(r"Recall_T\s*:\s*([0-9.]+)"),
    "f1_t": re.compile(r"F1_T\s*:\s*([0-9.]+)"),
    "table_score": re.compile(r"Table Score\s*:\s*([0-9.]+)"),
    "precision_c": re.compile(r"Precision_C\s*:\s*([0-9.]+)"),
    "recall_c": re.compile(r"Recall_C\s*:\s*([0-9.]+)"),
    "f1_c": re.compile(r"F1_C\s*:\s*([0-9.]+)"),
    "column_score": re.compile(r"Column Score\s*:\s*([0-9.]+)"),
}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_base_model(adapter_dir: Path) -> Optional[str]:
    path = adapter_dir / "adapter_config.json"
    if not path.exists():
        return None
    try:
        cfg = read_json(path)
        value = cfg.get("base_model_name_or_path")
        return str(value) if value else None
    except Exception:
        return None


def default_run_config(default_path: Optional[Path]) -> Dict[str, Any]:
    if default_path and default_path.exists():
        return read_json(default_path)
    return {
        "output_format": "ids",
        "retrieval": RetrievalConfig(max_tables=12, max_columns_per_table=20).to_dict(),
    }


def ensure_run_config(adapter_dir: Path, default_cfg: Dict[str, Any]) -> None:
    path = adapter_dir / "run_config.json"
    if path.exists():
        return
    cfg = dict(default_cfg)
    base_model = read_base_model(adapter_dir)
    if base_model:
        cfg["base_model"] = base_model
    write_run_config(adapter_dir, cfg)


def discover_adapters(mode: str, root: Path) -> List[Path]:
    if mode == "local":
        return sorted(p for p in root.glob("*/adapter") if (p / "adapter_config.json").exists())
    if mode == "rapidfire":
        patterns = [
            "runs/*/checkpoints/final_checkpoint",
            "*/runs/*/checkpoints/final_checkpoint",
        ]
        found: List[Path] = []
        for pat in patterns:
            found.extend(p for p in root.glob(pat) if (p / "adapter_config.json").exists())
        return sorted(set(found))
    raise ValueError("mode must be local or rapidfire")


def adapter_label(adapter: Path, mode: str, root: Path, idx: int) -> str:
    if mode == "local":
        return adapter.parent.name
    parts = adapter.resolve().parts
    if "runs" in parts:
        run_pos = len(parts) - 1 - list(reversed(parts)).index("runs")
        run_id = parts[run_pos + 1] if run_pos + 1 < len(parts) else str(idx)
        exp_name = parts[run_pos - 1] if run_pos > 0 else root.name
        return f"{exp_name}_run{run_id}"
    return f"adapter_{idx}"


def run_command(cmd: List[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    print("$ " + " ".join(cmd), flush=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=out, stderr=err, text=True)
    return int(proc.returncode)


def parse_metrics(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    m = SCORE_RE.search(text)
    out["leaderboard_score"] = float(m.group(1)) if m else -1.0
    for key, regex in METRIC_RES.items():
        mm = regex.search(text)
        out[key] = float(mm.group(1)) if mm else -1.0
    return out


def copy_adapter_for_inference(src: Path, dst: Path, keep_training_state: bool = False) -> None:
    """Copy only files needed by main.py unless training state is requested."""
    if dst.exists():
        shutil.rmtree(dst)
    ignore = None
    if not keep_training_state:
        ignore = shutil.ignore_patterns(
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "training_args.bin",
            "trainer_state.json",
        )
    shutil.copytree(src, dst, ignore=ignore)


def maybe_save_tokenizer(adapter_dir: Path) -> None:
    # RapidFire final checkpoints may not contain tokenizer files. Save them for
    # the final packaged adapter when possible.
    if any((adapter_dir / name).exists() for name in ["tokenizer.json", "tokenizer.model", "vocab.json"]):
        return
    base = read_base_model(adapter_dir)
    if not base:
        return
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.save_pretrained(adapter_dir)
        print(f"Saved tokenizer files into {adapter_dir}", flush=True)
    except Exception as exc:
        print(f"Warning: could not save tokenizer into {adapter_dir}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "rapidfire"], default="local")
    parser.add_argument("--project_dir", default=".")
    parser.add_argument("--experiment_dir", default="runs_lora8")
    parser.add_argument("--default_run_config", default=None)
    parser.add_argument("--validation_input", default="validation_input.json")
    parser.add_argument("--validation_gold", default="validation_gold_schema_links.json")
    parser.add_argument("--schemas_dir", default="schemas")
    parser.add_argument("--eval_dir", default="adapter_eval")
    parser.add_argument(
        "--inference_script",
        default="main.py",
        help="Inference entrypoint to use for validation, e.g. sample_main.py when main.py has experimental edits.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--max_input_length", type=int, default=None)
    parser.add_argument("--no_copy_best", action="store_true", help="Evaluate adapters but do not update ./adapter.")
    parser.add_argument("--min_score_to_copy", type=float, default=-1.0, help="Only update ./adapter if the best score is at least this value.")
    parser.add_argument("--keep_training_state", action="store_true", help="When copying best adapter, keep optimizer/scheduler/trainer files too.")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    root = (project_dir / args.experiment_dir).resolve()
    eval_root = (project_dir / args.eval_dir).resolve()
    eval_root.mkdir(parents=True, exist_ok=True)
    default_cfg = default_run_config(Path(args.default_run_config) if args.default_run_config else root / "default_run_config.json")

    adapters = discover_adapters(args.mode, root)
    if not adapters:
        raise SystemExit(f"No adapters found under {root} for mode={args.mode}")
    print(f"Found {len(adapters)} adapters.", flush=True)

    rows: List[Dict[str, Any]] = []
    for idx, adapter in enumerate(adapters, start=1):
        name = adapter_label(adapter, args.mode, root, idx)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        out_dir = eval_root / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)
        ensure_run_config(adapter, default_cfg)
        base_model = read_base_model(adapter)

        preds = out_dir / "preds.json"
        inference_script = project_dir / args.inference_script
        if not inference_script.exists():
            raise SystemExit(f"Inference script not found: {inference_script}")

        cmd = [
            sys.executable, str(inference_script),
            "--input", args.validation_input,
            "--output", str(preds),
            "--schemas_dir", args.schemas_dir,
            "--adapter_dir", str(adapter),
            "--max_new_tokens", str(args.max_new_tokens),
        ]
        if args.max_input_length is not None:
            cmd.extend(["--max_input_length", str(args.max_input_length)])
        if base_model:
            cmd.extend(["--base_model", base_model])
        code = run_command(cmd, project_dir, out_dir / "infer_stdout.log", out_dir / "infer_stderr.log")
        if code != 0:
            rows.append({"run_name": safe_name, "status": "infer_failed", "adapter_dir": str(adapter), "leaderboard_score": -1.0})
            continue

        eval_stdout = out_dir / "eval_stdout.log"
        code = run_command(
            [
                sys.executable, "eval.py",
                "--predictions", str(preds),
                "--gold", args.validation_gold,
                "--schemas_dir", args.schemas_dir,
                "--questions_input", args.validation_input,
                "--per_question_out", str(out_dir / "per_question.csv"),
            ],
            project_dir,
            eval_stdout,
            out_dir / "eval_stderr.log",
        )
        text = eval_stdout.read_text(encoding="utf-8") if eval_stdout.exists() else ""
        metrics = parse_metrics(text)
        row = {
            "run_name": safe_name,
            "status": "ok" if code == 0 else "eval_failed",
            "adapter_dir": str(adapter),
            "base_model": base_model or "",
            **metrics,
        }
        rows.append(row)
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)
        print(f"{safe_name}: score={row['leaderboard_score']:.4f}", flush=True)

    rows_sorted = sorted(rows, key=lambda r: float(r.get("leaderboard_score", -1.0)), reverse=True)
    fieldnames = sorted({k for r in rows_sorted for k in r})
    leaderboard = eval_root / "leaderboard.csv"
    with leaderboard.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    print(f"Wrote {leaderboard}", flush=True)

    best = next((r for r in rows_sorted if r.get("status") == "ok" and float(r.get("leaderboard_score", -1.0)) >= 0), None)
    if best:
        best_src = Path(str(best["adapter_dir"]))
        best_dst = project_dir / "adapter"
        best_score = float(best.get("leaderboard_score", -1.0))
        with (eval_root / "BEST_RUN.json").open("w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
        print(f"Best run: {best['run_name']} score={best_score:.4f}", flush=True)
        if args.no_copy_best:
            print("Did not update ./adapter because --no_copy_best was set.", flush=True)
        elif best_score < args.min_score_to_copy:
            print(
                f"Did not update ./adapter because best score {best_score:.4f} is below "
                f"--min_score_to_copy {args.min_score_to_copy:.4f}.",
                flush=True,
            )
        else:
            copy_adapter_for_inference(best_src, best_dst, keep_training_state=args.keep_training_state)
            maybe_save_tokenizer(best_dst)
            print(f"Copied best adapter to {best_dst}", flush=True)
    else:
        print("No successful adapter found.", flush=True)


if __name__ == "__main__":
    main()
