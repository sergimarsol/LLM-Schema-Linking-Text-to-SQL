#!/usr/bin/env python3
"""
Prepare SFT training data for schema linking task.

Converts raw train.json and validation.json into chat-format JSONL files
suitable for fine-tuning. Includes schema serialization, output normalization,
validation, and optional balancing.

Usage:
  python prepare_data.py \
    --train train.json \
    --validation validation.json \
    --schemas schemas \
    --out_dir data/processed \
    --include_types \
    --include_keys \
    --balance
"""

import json
import argparse
import os
import copy
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import sys


class SchemaLoader:
    """Load and parse database schemas."""
    
    def __init__(self, schema_dir: str):
        self.schema_dir = Path(schema_dir)
        self.schemas = {}
    
    def load_schema(self, db_id: str) -> Dict[str, Any]:
        """
        Load a schema by db_id.
        Handles db_id with spaces -> filename with underscores mapping.
        """
        if db_id in self.schemas:
            return self.schemas[db_id]
        
        # Convert spaces to underscores for filename
        filename = db_id.replace(" ", "_") + ".json"
        filepath = self.schema_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Schema file not found: {filepath}")
        
        with open(filepath) as f:
            schema = json.load(f)
        
        self.schemas[db_id] = schema
        return schema
    
    def get_col_by_table(self, schema: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """
        Build table -> columns mapping with types and key info.
        
        Returns:
          {table_name: [{"name": col_name, "type": col_type, "col_idx": idx}, ...]}
        """
        col_by_table = defaultdict(list)
        
        # Build PK and FK sets for quick lookup
        pk_set = set(schema.get("primary_keys", []))
        fk_from_set = set(fk[0] for fk in schema.get("foreign_keys", []))
        
        # Iterate through column_names
        for col_idx, (table_idx, col_name) in enumerate(schema["column_names"]):
            if table_idx == -1:  # Skip wildcard
                continue
            
            # Handle column_types offset (1 fewer than column_names)
            if col_idx >= len(schema["column_types"]):
                continue
            
            table_name = schema["table_names"][table_idx]
            col_type = schema["column_types"][col_idx]
            
            col_by_table[table_name].append({
                "name": col_name,
                "type": col_type,
                "col_idx": col_idx,
                "is_pk": col_idx in pk_set,
                "is_fk": col_idx in fk_from_set
            })
        
        return dict(col_by_table)


def serialize_schema(schema: Dict[str, Any], col_by_table: Dict[str, List[Dict]], 
                     include_types: bool = True, include_keys: bool = True) -> str:
    """
    Serialize a schema into readable text format.
    
    Format:
      Database schema:
      Table: TableName
        Columns:
        - column_name [type] [key_info]
        - ...
      Table: ...
    """
    lines = ["Database schema:"]
    
    # Sort tables alphabetically
    for table_name in sorted(col_by_table.keys()):
        lines.append(f"Table: {table_name}")
        lines.append("  Columns:")
        
        # Sort columns alphabetically
        cols = sorted(col_by_table[table_name], key=lambda c: c["name"])
        
        for col in cols:
            col_str = f"  - {col['name']}"
            
            if include_types:
                col_str += f" [{col['type']}]"
            
            if include_keys:
                key_info = []
                if col["is_pk"]:
                    key_info.append("PRIMARY KEY")
                if col["is_fk"]:
                    key_info.append("FOREIGN KEY")
                if key_info:
                    col_str += " " + " ".join(key_info)
            
            lines.append(col_str)
    
    return "\n".join(lines)


def create_prompt(schema_text: str, question: str) -> str:
    """Create the user prompt for schema linking task."""
    return f"""You are given a database schema and a natural language question.
Return ONLY valid JSON mapping table names to lists of column names.
Include tables even if no specific columns are used.
Do not include explanations.

Schema:
{schema_text}

Question:
{question}

Answer:"""


def normalize_schema_links(schema_links: Dict[str, List[str]]) -> str:
    """
    Normalize schema_links to stable JSON.
    - Sort table names alphabetically
    - Sort column names alphabetically within each table
    - Preserve original casing
    """
    normalized = {
        table: sorted(cols)
        for table, cols in sorted(schema_links.items())
    }
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def validate_schema_links(schema_links: Dict[str, List[str]], col_by_table: Dict[str, List[Dict]], 
                         question_id: int, db_id: str) -> Tuple[bool, List[str]]:
    """
    Validate that schema_links matches actual schema.
    
    Returns:
      (is_valid, warnings)
    """
    warnings = []
    
    for table_name, columns in schema_links.items():
        # Check table exists
        if table_name not in col_by_table:
            warnings.append(f"question_id={question_id} db_id={db_id}: table '{table_name}' not in schema")
            continue
        
        # Check columns exist
        valid_cols = {col["name"] for col in col_by_table[table_name]}
        for col_name in columns:
            if col_name not in valid_cols:
                warnings.append(f"question_id={question_id} db_id={db_id}: column '{col_name}' not in table '{table_name}'")
    
    return len(warnings) == 0, warnings


def load_and_process_examples(data_file: str, schema_loader: SchemaLoader, 
                              include_types: bool, include_keys: bool) -> List[Dict[str, Any]]:
    """
    Load examples and process them into SFT records.
    """
    with open(data_file) as f:
        raw_data = json.load(f)
    
    records = []
    all_warnings = []
    
    for example in raw_data:
        question_id = example["question_id"]
        db_id = example["db_id"]
        question = example["question"]
        schema_links = example["schema_links"]
        
        # Load schema
        try:
            schema = schema_loader.load_schema(db_id)
        except FileNotFoundError as e:
            all_warnings.append(str(e))
            continue
        
        col_by_table = schema_loader.get_col_by_table(schema)
        
        # Validate
        is_valid, val_warnings = validate_schema_links(schema_links, col_by_table, question_id, db_id)
        all_warnings.extend(val_warnings)
        
        # Serialize schema
        schema_text = serialize_schema(schema, col_by_table, include_types, include_keys)
        
        # Create prompt
        user_content = create_prompt(schema_text, question)
        
        # Normalize output
        assistant_content = normalize_schema_links(schema_links)
        
        # Compute difficulty metrics
        num_tables = len(schema_links)
        num_columns = sum(len(cols) for cols in schema_links.values())
        is_multitable = num_tables > 1
        has_empty_columns = any(len(cols) == 0 for cols in schema_links.values())
        
        # Create SFT record with metadata
        record = {
            "messages": [
                {
                    "role": "system",
                    "content": "You output only valid JSON schema links."
                },
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ],
            "question_id": question_id,
            "db_id": db_id,
            "source": "original",
            "num_tables": num_tables,
            "num_columns": num_columns,
            "is_multitable": is_multitable,
            "has_empty_columns": has_empty_columns,
            "is_oversampled": False
        }
        records.append(record)
    
    # Print warnings
    if all_warnings:
        print(f"\n[WARNINGS] Found {len(all_warnings)} validation issues:")
        for warning in all_warnings[:10]:  # Print first 10
            print(f"  {warning}")
        if len(all_warnings) > 10:
            print(f"  ... and {len(all_warnings) - 10} more")
    
    return records


def collect_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collect dataset statistics including difficulty-aware metrics."""
    stats = {
        "total_examples": len(records),
        "unique_databases": len(set(r["db_id"] for r in records)),
        "examples_per_db": dict(Counter(r["db_id"] for r in records)),
        "multi_table_examples": 0,
        "single_table_examples": 0,
        "empty_column_examples": 0,
        "many_column_examples": 0,  # >= 3 columns
        "oversampled_examples": 0,
        "original_examples": 0,
        "avg_tables_per_example": 0,
        "avg_columns_per_example": 0,
        "max_tables": 0,
        "max_columns": 0,
    }
    
    num_tables_list = []
    num_cols_list = []
    
    for record in records:
        # Count source types
        if record.get("is_oversampled", False):
            stats["oversampled_examples"] += 1
        else:
            stats["original_examples"] += 1
        
        # Parse assistant content to count tables/columns
        try:
            schema_links = json.loads(record["messages"][2]["content"])
            n_tables = len(schema_links)
            n_cols = sum(len(cols) for cols in schema_links.values())
            
            num_tables_list.append(n_tables)
            num_cols_list.append(n_cols)
            
            if n_tables > 1:
                stats["multi_table_examples"] += 1
            else:
                stats["single_table_examples"] += 1
            
            if any(len(cols) == 0 for cols in schema_links.values()):
                stats["empty_column_examples"] += 1
            
            if n_cols >= 3:
                stats["many_column_examples"] += 1
            
        except json.JSONDecodeError:
            pass
    
    if num_tables_list:
        stats["avg_tables_per_example"] = sum(num_tables_list) / len(num_tables_list)
        stats["max_tables"] = max(num_tables_list)
    
    if num_cols_list:
        stats["avg_columns_per_example"] = sum(num_cols_list) / len(num_cols_list)
        stats["max_columns"] = max(num_cols_list)
    
    return stats


def balance_examples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Light balancing: oversample underrepresented databases.
    - Cap at 2x max (avoid over-repeating)
    - Only oversample if db has < 10 examples
    - Mark oversampled records with metadata
    """
    db_counts = Counter(r["db_id"] for r in records)
    
    # Identify underrepresented databases and target count
    balanced = list(records)
    THRESHOLD = 10  # Only oversample if below this
    MAX_DUPLICATIONS = 2  # Cap at 2x
    
    balancing_log = {}
    
    for db_id, count in db_counts.items():
        if count < THRESHOLD:
            # Get examples for this db
            db_examples = [r for r in records if r["db_id"] == db_id]
            
            # Calculate duplications needed to reach 2x
            target_count = min(count * MAX_DUPLICATIONS, THRESHOLD)
            duplications_needed = target_count - count
            
            if duplications_needed > 0:
                # Add duplicated examples with oversampled marker
                for i in range(duplications_needed):
                    example = db_examples[i % len(db_examples)]
                    # Deep copy to preserve nested structures (messages, etc.)
                    dup_record = copy.deepcopy(example)
                    dup_record["is_oversampled"] = True
                    dup_record["source"] = "oversampled"
                    balanced.append(dup_record)
                
                balancing_log[db_id] = {
                    "original_count": count,
                    "target_count": target_count,
                    "duplications_added": duplications_needed
                }
    
    # Print balancing summary
    if balancing_log:
        print("\n  Balancing log:")
        for db_id, log in balancing_log.items():
            print(f"    {db_id}: {log['original_count']} → {log['target_count']} (+{log['duplications_added']})")
    
    return balanced


def save_jsonl(records: List[Dict[str, Any]], filepath: str):
    """Save records to JSONL file."""
    with open(filepath, 'w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data for schema linking")
    parser.add_argument("--train", required=True, help="Path to train.json")
    parser.add_argument("--validation", required=True, help="Path to validation.json")
    parser.add_argument("--schemas", required=True, help="Path to schemas directory")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory")
    parser.add_argument("--include_types", action="store_true", 
                        help="Include column types in schema")
    parser.add_argument("--include_keys", action="store_true",
                        help="Include PK/FK info in schema")
    parser.add_argument("--balance", action="store_true",
                        help="Apply light balancing to underrepresented databases (cap at 2x)")
    
    args = parser.parse_args()
    
    # Create output directory
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PREPARE DATA FOR SCHEMA LINKING")
    print("=" * 60)
    
    schema_loader = SchemaLoader(args.schemas)
    
    # Load and process training data
    print("\n[1/3] Processing training data...")
    train_records = load_and_process_examples(
        args.train, schema_loader, args.include_types, args.include_keys
    )
    print(f"  Loaded {len(train_records)} training examples")
    
    # Save original for comparison if balancing will be applied
    train_records_original = copy.deepcopy(train_records) if args.balance else None
    
    # Collect stats before balancing
    train_stats_before = collect_stats(train_records)
    print(f"  Distribution before balancing: {dict(train_stats_before['examples_per_db'])}")
    
    # Balance training data
    if args.balance:
        print("  Balancing underrepresented databases...")
        train_records = balance_examples(train_records)
        train_stats_after = collect_stats(train_records)
        print(f"  After balancing: {len(train_records)} examples")
    
    train_stats = collect_stats(train_records)
    
    # Load and process validation data
    print("\n[2/3] Processing validation data...")
    val_records = load_and_process_examples(
        args.validation, schema_loader, args.include_types, args.include_keys
    )
    print(f"  Loaded {len(val_records)} validation examples")
    
    val_stats = collect_stats(val_records)
    
    # Save outputs
    print("\n[3/3] Saving outputs...")
    
    train_out = out_path / "train_sft.jsonl"
    val_out = out_path / "val_sft.jsonl"
    stats_out = out_path / "data_stats.json"
    config_out = out_path / "preprocessing_config.json"
    
    # If balancing was applied, save both original and balanced versions
    if args.balance and len(train_records) > len(train_records_original):
        train_original_out = out_path / "train_sft_original.jsonl"
        train_balanced_out = out_path / "train_sft_balanced.jsonl"
        
        save_jsonl(train_records_original, str(train_original_out))
        save_jsonl(train_records, str(train_balanced_out))
        
        print(f"  Saved {len(train_records_original)} original training records to {train_original_out}")
        print(f"  Saved {len(train_records)} balanced training records to {train_balanced_out}")
        print(f"  (Also saving balanced version as default train_sft.jsonl)")
    
    save_jsonl(train_records, str(train_out))
    save_jsonl(val_records, str(val_out))
    
    print(f"  Saved {len(train_records)} training records to {train_out}")
    print(f"  Saved {len(val_records)} validation records to {val_out}")
    
    # Save statistics
    combined_stats = {
        "training": train_stats,
        "validation": val_stats
    }
    
    with open(stats_out, 'w') as f:
        json.dump(combined_stats, f, indent=2)
    
    print(f"  Saved statistics to {stats_out}")
    
    # Save config
    config = {
        "schema_format": "text with table/column/type/key_info",
        "include_types": args.include_types,
        "include_keys": args.include_keys,
        "balancing_applied": args.balance,
        "balancing_strategy": "light oversampling (cap at 2x) for databases with < 10 examples",
        "output_format": "chat-format JSONL with system/user/assistant roles + metadata",
        "normalization": "alphabetically sorted tables and columns",
        "metadata_fields": [
            "source: 'original' or 'oversampled'",
            "num_tables: count of tables in schema_links",
            "num_columns: count of columns in schema_links",
            "is_multitable: boolean",
            "has_empty_columns: boolean",
            "is_oversampled: boolean"
        ]
    }
    
    with open(config_out, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  Saved config to {config_out}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTraining data:")
    print(f"  Total examples: {train_stats['total_examples']}")
    print(f"  Original examples: {train_stats['original_examples']}")
    print(f"  Oversampled examples: {train_stats['oversampled_examples']}")
    print(f"  Unique databases: {train_stats['unique_databases']}")
    print(f"  Multi-table examples: {train_stats['multi_table_examples']} ({100*train_stats['multi_table_examples']/max(train_stats['total_examples'],1):.1f}%)")
    print(f"  Empty-column examples: {train_stats['empty_column_examples']}")
    print(f"  Examples with >=3 columns: {train_stats['many_column_examples']}")
    print(f"  Avg tables per example: {train_stats['avg_tables_per_example']:.2f}")
    print(f"  Avg columns per example: {train_stats['avg_columns_per_example']:.2f}")
    
    print(f"\nValidation data:")
    print(f"  Total examples: {val_stats['total_examples']}")
    print(f"  Unique databases: {val_stats['unique_databases']}")
    print(f"  Multi-table examples: {val_stats['multi_table_examples']} ({100*val_stats['multi_table_examples']/max(val_stats['total_examples'],1):.1f}%)")
    
    print(f"\nOutputs saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
