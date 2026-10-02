#!/usr/bin/env python3
"""
Select high-value training examples for manual/LLM paraphrase augmentation.

Input:
  train.json

Outputs:
  examples_to_augment.json
  examples_to_augment_review.csv

Usage:
  python select_examples_for_augmentation.py \
    --train train.json \
    --out_json examples_to_augment.json \
    --out_csv examples_to_augment_review.csv \
    --max_examples 90
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from typing import Any, Dict, List, Tuple


def count_tables(schema_links: Dict[str, List[str]]) -> int:
    return len(schema_links)


def count_columns(schema_links: Dict[str, List[str]]) -> int:
    return sum(len(cols) for cols in schema_links.values())


def has_empty_column_table(schema_links: Dict[str, List[str]]) -> bool:
    return any(len(cols) == 0 for cols in schema_links.values())


def sql_contains(sql: str, patterns: List[str]) -> bool:
    sql_l = sql.lower()
    return any(p in sql_l for p in patterns)


def difficulty_score(
    example: Dict[str, Any],
    db_counts: Counter,
    underrepresented_threshold: int,
) -> Tuple[int, List[str]]:
    db_id = example["db_id"]
    question = example["question"]
    sql = example["gold_sql"]
    schema_links = example["schema_links"]

    n_tables = count_tables(schema_links)
    n_columns = count_columns(schema_links)
    db_count = db_counts[db_id]

    score = 0
    reasons: List[str] = []

    # 1. Underrepresented databases matter most.
    if db_count < underrepresented_threshold:
        score += 5
        reasons.append(f"underrepresented_db:{db_count}")
    elif db_count <= 12:
        score += 2
        reasons.append(f"low_resource_db:{db_count}")

    # 2. Multi-table schema linking is harder and useful.
    if n_tables >= 3:
        score += 5
        reasons.append(f"many_tables:{n_tables}")
    elif n_tables == 2:
        score += 3
        reasons.append("two_tables")

    # 3. Column-heavy examples help column-level F1.
    if n_columns >= 8:
        score += 4
        reasons.append(f"many_columns:{n_columns}")
    elif n_columns >= 5:
        score += 2
        reasons.append(f"medium_columns:{n_columns}")

    # 4. Empty column lists are special evaluation boundary cases.
    if has_empty_column_table(schema_links):
        score += 4
        reasons.append("empty_column_table")

    # 5. SQL structure difficulty.
    sql_l = sql.lower()

    join_count = len(re.findall(r"\bjoin\b", sql_l))
    if join_count >= 2:
        score += 3
        reasons.append(f"multi_join:{join_count}")
    elif join_count == 1:
        score += 2
        reasons.append("join")

    if sql_contains(sql, ["group by", "order by", "having"]):
        score += 2
        reasons.append("group_order_having")

    if sql_contains(sql, ["count(", "sum(", "avg(", "min(", "max(", "distinct"]):
        score += 1
        reasons.append("aggregate_or_distinct")

    if sql_contains(sql, [" exists", " in (", " not in (", "select top", "select max", "select min"]):
        score += 3
        reasons.append("subquery_or_top")

    # 6. Long questions usually contain more semantic constraints.
    if len(question.split()) >= 25:
        score += 1
        reasons.append("long_question")

    return score, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.json")
    parser.add_argument("--out_json", default="examples_to_augment.json")
    parser.add_argument("--out_csv", default="examples_to_augment_review.csv")
    parser.add_argument("--max_examples", type=int, default=90)
    parser.add_argument("--underrepresented_threshold", type=int, default=10)
    parser.add_argument(
        "--min_score",
        type=int,
        default=9,
        help="Minimum difficulty score before max_examples cutoff.",
    )
    args = parser.parse_args()

    with open(args.train, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input train file must be a JSON list.")

    required = {"question_id", "db_id", "question", "gold_sql", "schema_links"}
    for i, ex in enumerate(data):
        missing = required - set(ex)
        if missing:
            raise ValueError(f"Example {i} missing fields: {missing}")

    db_counts = Counter(ex["db_id"] for ex in data)

    scored = []
    for ex in data:
        score, reasons = difficulty_score(
            ex,
            db_counts=db_counts,
            underrepresented_threshold=args.underrepresented_threshold,
        )

        if score >= args.min_score:
            scored.append(
                {
                    "example": ex,
                    "score": score,
                    "reasons": reasons,
                    "num_tables": count_tables(ex["schema_links"]),
                    "num_columns": count_columns(ex["schema_links"]),
                    "db_count": db_counts[ex["db_id"]],
                }
            )

    # Sort strongest first.
    scored.sort(
        key=lambda x: (
            x["score"],
            x["num_tables"],
            x["num_columns"],
            -x["db_count"],
        ),
        reverse=True,
    )

    selected = scored[: args.max_examples]

    # Keep JSON in the same base format as train.json, with extra metadata allowed.
    json_out = []
    for item in selected:
        ex = dict(item["example"])
        ex["selection_score"] = item["score"]
        ex["selection_reasons"] = item["reasons"]
        ex["num_linked_tables"] = item["num_tables"]
        ex["num_linked_columns"] = item["num_columns"]
        ex["original_db_example_count"] = item["db_count"]
        json_out.append(ex)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "db_id",
                "db_count",
                "score",
                "reasons",
                "num_tables",
                "num_columns",
                "question",
                "gold_sql",
            ],
        )
        writer.writeheader()
        for item in selected:
            ex = item["example"]
            writer.writerow(
                {
                    "question_id": ex["question_id"],
                    "db_id": ex["db_id"],
                    "db_count": item["db_count"],
                    "score": item["score"],
                    "reasons": ";".join(item["reasons"]),
                    "num_tables": item["num_tables"],
                    "num_columns": item["num_columns"],
                    "question": ex["question"],
                    "gold_sql": ex["gold_sql"],
                }
            )

    print(f"Loaded original examples: {len(data)}")
    print(f"Selected examples: {len(selected)}")
    print(f"Wrote JSON: {args.out_json}")
    print(f"Wrote review CSV: {args.out_csv}")

    print("\nSelected examples by database:")
    selected_db_counts = Counter(item["example"]["db_id"] for item in selected)
    for db_id, count in sorted(selected_db_counts.items()):
        print(f"  {db_id}: {count}")


if __name__ == "__main__":
    main()