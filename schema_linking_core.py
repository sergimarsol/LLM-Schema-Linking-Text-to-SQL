#!/usr/bin/env python3
"""Shared schema-linking utilities for CSE/DSC 234 Project 2.

This module is intentionally dependency-light.  It is used by training,
RapidFire data formatting, and final inference so that the prompt seen during
training is the same prompt used by main.py.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SYSTEM_PROMPT_IDS = (
    "You are a schema linking assistant. Given a database schema candidate list "
    "and a natural language question, output ONLY a JSON object. The JSON keys "
    "must be table IDs like T1 and the values must be lists of column IDs like "
    "C1. Use an empty list [] for a referenced table with no specific columns. "
    "Do not explain. Do not output table or column names."
)

SYSTEM_PROMPT_NAMES = (
    "You are a schema linking assistant. Given a database schema candidate list "
    "and a natural language question, output ONLY a JSON object mapping each "
    "referenced table name to a list of referenced column names. Use an empty "
    "list [] for a referenced table with no specific columns. Use only provided "
    "schema identifiers. Do not explain."
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "had", "how", "in", "include", "into", "is", "it", "its", "list",
    "many", "me", "more", "most", "no", "not", "number", "of", "on", "one",
    "only", "or", "per", "show", "that", "the", "their", "them", "there",
    "to", "value", "values", "was", "were", "what", "where", "which", "with",
    "without", "all", "each", "display", "give", "get", "make", "return",
}

# Domain-specific hints that help lexical retrieval on the released schemas.
# These do not predict answers; they only help candidate schema recall.
SYNONYM_GROUPS = [
    {"x", "east", "easting", "utm", "utme", "utmx", "longitude", "long", "lon"},
    {"y", "north", "northing", "utm", "utmn", "utmy", "latitude", "lat"},
    {"date", "year", "month", "day", "time"},
    {"school", "entity", "institution"},
    {"student", "subgroup", "sub", "group"},
    {"absence", "absent", "absenteeism", "chronic"},
    {"rate", "percent", "percentage", "pct"},
    {"count", "total", "number", "num", "cnt"},
    {"name", "title", "description", "text", "descr"},
    {"species", "sp", "spcode", "common", "scientific", "genus"},
    {"location", "loc", "site", "station", "place", "plot"},
    {"event", "record", "entry", "observation", "obs"},
    {"vehicle", "veh", "vehno", "vin", "make", "model"},
    {"crash", "case", "caseid", "accident"},
    {"injury", "injuries", "ais", "severity", "region"},
    {"tire", "tiresize", "tiremodel", "pressure"},
    {"employee", "emp", "empid", "staff"},
    {"account", "acct", "acctcode", "acctnum"},
    {"contact", "observer", "first", "last", "organization"},
]

SYNONYMS: Dict[str, set[str]] = {}
for group in SYNONYM_GROUPS:
    for token in group:
        SYNONYMS.setdefault(token, set()).update(group)

CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ColumnInfo:
    name: str
    col_type: str = "unknown"
    is_pk: bool = False
    is_fk: bool = False


@dataclass
class TableInfo:
    name: str
    columns: List[ColumnInfo]


@dataclass
class RetrievalConfig:
    max_tables: int = 10
    max_columns_per_table: int = 18
    min_columns_per_table: int = 4
    include_types: bool = True
    include_keys: bool = True
    include_join_neighbors: bool = True
    join_neighbor_limit: int = 2
    always_include_table_columns_for_gold: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "RetrievalConfig":
        if not d:
            return cls()
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def schema_filename_for_db(db_id: str) -> str:
    return db_id.replace(" ", "_").replace("/", "_") + ".json"


def load_raw_schema(db_id: str, schemas_dir: str | os.PathLike[str]) -> Dict[str, Any]:
    path = Path(schemas_dir) / schema_filename_for_db(db_id)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def column_type_for(raw_schema: Mapping[str, Any], col_idx: int) -> str:
    column_types = list(raw_schema.get("column_types", []))
    column_names = list(raw_schema.get("column_names_original") or raw_schema.get("column_names") or [])
    # Spider schemas often have the synthetic [-1, "*"] column but no matching type.
    type_idx = col_idx - 1 if len(column_types) == len(column_names) - 1 else col_idx
    if 0 <= type_idx < len(column_types):
        return str(column_types[type_idx] or "unknown")
    return "unknown"


def schema_to_tables(raw_schema: Mapping[str, Any]) -> List[TableInfo]:
    table_names = list(raw_schema.get("table_names_original") or raw_schema["table_names"])
    column_names = list(raw_schema.get("column_names_original") or raw_schema["column_names"])
    primary_keys = set(raw_schema.get("primary_keys", []))
    fk_cols = set()
    for pair in raw_schema.get("foreign_keys", []) or []:
        if isinstance(pair, list) and len(pair) == 2:
            fk_cols.add(pair[0])
            fk_cols.add(pair[1])

    by_table: Dict[int, List[ColumnInfo]] = {i: [] for i in range(len(table_names))}
    for col_idx, pair in enumerate(column_names):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        table_idx, col_name = pair
        if table_idx == -1:
            continue
        by_table[int(table_idx)].append(
            ColumnInfo(
                name=str(col_name),
                col_type=column_type_for(raw_schema, col_idx),
                is_pk=col_idx in primary_keys,
                is_fk=col_idx in fk_cols,
            )
        )

    return [TableInfo(name=table_names[i], columns=by_table[i]) for i in range(len(table_names))]


def load_schema_as_dict(db_id: str, schemas_dir: str | os.PathLike[str]) -> Dict[str, List[str]]:
    raw = load_raw_schema(db_id, schemas_dir)
    return {t.name: [c.name for c in t.columns] for t in schema_to_tables(raw)}


def split_identifier(text: str) -> List[str]:
    text = CAMEL_RE.sub(" ", str(text))
    text = text.replace("_", " ").replace("/", " ").replace("#", " number ")
    out: List[str] = []
    for token in TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS:
            continue
        out.append(token)
        if token.endswith("s") and len(token) > 3:
            out.append(token[:-1])
    return out


def question_tokens(question: str) -> List[str]:
    toks = split_identifier(question)
    expanded: List[str] = []
    for t in toks:
        expanded.append(t)
        expanded.extend(sorted(SYNONYMS.get(t, set())))
    # Preserve order while deduplicating.
    seen = set()
    out = []
    for t in expanded:
        if t not in seen and t not in STOPWORDS:
            out.append(t)
            seen.add(t)
    return out


def token_overlap_score(q_tokens: Sequence[str], name_tokens: Sequence[str]) -> float:
    if not name_tokens:
        return 0.0
    q = set(q_tokens)
    n = set(name_tokens)
    inter = q & n
    if not inter:
        return 0.0
    return len(inter) / math.sqrt(len(n))


def substring_score(question_lc: str, identifier: str) -> float:
    ident = identifier.lower()
    if not ident:
        return 0.0
    plain = re.sub(r"[^a-z0-9]+", " ", ident).strip()
    compact = re.sub(r"[^a-z0-9]+", "", ident)
    score = 0.0
    if plain and plain in question_lc:
        score += 3.0
    if compact and len(compact) >= 4 and compact in re.sub(r"[^a-z0-9]+", "", question_lc):
        score += 1.5
    return score


def build_fk_adjacency(raw_schema: Mapping[str, Any]) -> Dict[str, set[str]]:
    tables = list(raw_schema.get("table_names_original") or raw_schema["table_names"])
    col_to_table: Dict[int, str] = {}
    for col_idx, pair in enumerate(raw_schema.get("column_names_original") or raw_schema["column_names"]):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        tidx, _ = pair
        if tidx != -1:
            col_to_table[col_idx] = tables[int(tidx)]
    adj: Dict[str, set[str]] = {t: set() for t in tables}
    for pair in raw_schema.get("foreign_keys", []) or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = pair
        ta, tb = col_to_table.get(int(a)), col_to_table.get(int(b))
        if ta and tb and ta != tb:
            adj.setdefault(ta, set()).add(tb)
            adj.setdefault(tb, set()).add(ta)
    return adj


def score_schema(question: str, tables: Sequence[TableInfo]) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    q_tokens = question_tokens(question)
    question_lc = question.lower()
    table_scores: Dict[str, float] = {}
    column_scores: Dict[Tuple[str, str], float] = {}

    for table in tables:
        t_tokens = split_identifier(table.name)
        t_score = 0.75 * token_overlap_score(q_tokens, t_tokens) + substring_score(question_lc, table.name)
        best_col = 0.0
        sum_top_cols: List[float] = []
        for col in table.columns:
            c_tokens = split_identifier(col.name)
            score = 1.35 * token_overlap_score(q_tokens, c_tokens) + substring_score(question_lc, col.name)
            # Common natural-language operator hints.
            c_lc = col.name.lower()
            if any(tok in q_tokens for tok in ["year", "month", "day", "date"]) and any(x in c_lc for x in ["year", "month", "date", "time"]):
                score += 0.35
            if any(tok in q_tokens for tok in ["count", "number", "num", "total"]) and any(x in c_lc for x in ["count", "num", "total", "number"]):
                score += 0.25
            if any(tok in q_tokens for tok in ["average", "avg", "mean"]) and any(x in c_lc for x in ["avg", "mean", "rate", "percent"]):
                score += 0.15
            if col.is_pk or col.is_fk:
                # Join/key columns often need to be linked even if only implied.
                score += 0.05
            column_scores[(table.name, col.name)] = score
            best_col = max(best_col, score)
            sum_top_cols.append(score)
        sum_top_cols.sort(reverse=True)
        table_scores[table.name] = t_score + 0.70 * best_col + 0.15 * sum(sum_top_cols[:4])
    return table_scores, column_scores


def normalize_links_to_schema(links: Mapping[str, Any], schema: Mapping[str, List[str]]) -> Dict[str, List[str]]:
    table_map = {t.lower(): t for t in schema}
    col_maps = {t.lower(): {c.lower(): c for c in cols} for t, cols in schema.items()}
    out: Dict[str, List[str]] = {}
    if not isinstance(links, Mapping):
        return out
    for t, raw_cols in links.items():
        if not isinstance(t, str):
            continue
        real_t = table_map.get(t.lower())
        if real_t is None:
            continue
        if isinstance(raw_cols, list):
            cols_iter = raw_cols
        elif isinstance(raw_cols, dict):
            cols_iter = list(raw_cols.keys())
        elif isinstance(raw_cols, str):
            cols_iter = [raw_cols]
        else:
            cols_iter = []
        seen = set()
        real_cols: List[str] = []
        for c in cols_iter:
            if not isinstance(c, str):
                continue
            real_c = col_maps[real_t.lower()].get(c.lower())
            if real_c is not None and real_c.lower() not in seen:
                real_cols.append(real_c)
                seen.add(real_c.lower())
        out[real_t] = real_cols
    # Preserve schema table order and schema column order.
    ordered: Dict[str, List[str]] = {}
    for t, schema_cols in schema.items():
        if t not in out:
            continue
        selected = {c.lower() for c in out[t]}
        ordered[t] = [c for c in schema_cols if c.lower() in selected]
    return ordered


def build_candidate_schema(
    question: str,
    raw_schema: Mapping[str, Any],
    cfg: RetrievalConfig,
    force_links: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[TableInfo], Dict[str, str], Dict[Tuple[str, str], str]]:
    """Return candidate tables plus stable IDs.

    force_links is used for training/eval-loss data construction only so that a
    gold target never references a hidden identifier.  Final inference passes
    force_links=None.
    """
    all_tables = schema_to_tables(raw_schema)
    schema_dict = {t.name: [c.name for c in t.columns] for t in all_tables}
    normalized_force = normalize_links_to_schema(force_links or {}, schema_dict) if force_links else {}
    table_scores, column_scores = score_schema(question, all_tables)
    fk_adj = build_fk_adjacency(raw_schema)

    forced_tables = set(normalized_force.keys())
    sorted_tables = sorted(all_tables, key=lambda t: (table_scores.get(t.name, 0.0), t.name.lower()), reverse=True)

    chosen_names: List[str] = []
    for t in sorted_tables:
        if len(chosen_names) >= cfg.max_tables:
            break
        chosen_names.append(t.name)

    for t in all_tables:
        if t.name in forced_tables and t.name not in chosen_names:
            chosen_names.append(t.name)

    if cfg.include_join_neighbors:
        # Add a small number of FK-neighbor tables for selected high-score tables.
        current = list(chosen_names)
        for tname in current[: cfg.max_tables]:
            neighbors = sorted(fk_adj.get(tname, set()), key=lambda n: table_scores.get(n, 0.0), reverse=True)
            added = 0
            for nb in neighbors:
                if nb not in chosen_names:
                    chosen_names.append(nb)
                    added += 1
                if added >= cfg.join_neighbor_limit:
                    break

    by_name = {t.name: t for t in all_tables}
    candidates: List[TableInfo] = []
    for tname in chosen_names:
        table = by_name[tname]
        forced_cols = set(normalized_force.get(tname, []))
        sorted_cols = sorted(
            table.columns,
            key=lambda c: (
                c.name in forced_cols,
                column_scores.get((table.name, c.name), 0.0),
                c.is_pk or c.is_fk,
                c.name.lower(),
            ),
            reverse=True,
        )
        keep_n = max(cfg.min_columns_per_table, cfg.max_columns_per_table)
        selected = sorted_cols[:keep_n]
        # Force any gold columns into the candidate set during training.
        selected_names = {c.name for c in selected}
        for c in table.columns:
            if c.name in forced_cols and c.name not in selected_names:
                selected.append(c)
                selected_names.add(c.name)
        # Restore original schema column order for readability.
        selected_set = {c.name for c in selected}
        ordered_cols = [c for c in table.columns if c.name in selected_set]
        candidates.append(TableInfo(name=table.name, columns=ordered_cols))

    # Stable IDs by candidate order.  Tables are sorted by retrieval score with forced append.
    table_id = {table.name: f"T{i+1}" for i, table in enumerate(candidates)}
    column_id: Dict[Tuple[str, str], str] = {}
    for table in candidates:
        for j, col in enumerate(table.columns):
            column_id[(table.name, col.name)] = f"C{j+1}"
    return candidates, table_id, column_id


def serialize_candidate_schema(
    candidates: Sequence[TableInfo],
    table_id: Mapping[str, str],
    column_id: Mapping[Tuple[str, str], str],
    cfg: RetrievalConfig,
    output_format: str = "ids",
) -> str:
    lines: List[str] = []
    if output_format == "ids":
        lines.append("Candidate schema. Use the IDs, not names, in your JSON answer.")
    else:
        lines.append("Candidate schema. Use the exact table and column names in your JSON answer.")
    for table in candidates:
        tid = table_id[table.name]
        if output_format == "ids":
            lines.append(f"{tid} = {table.name}")
        else:
            lines.append(f"Table: {table.name}")
        for col in table.columns:
            cid = column_id[(table.name, col.name)]
            bits = []
            if cfg.include_types and col.col_type:
                bits.append(str(col.col_type))
            if cfg.include_keys:
                if col.is_pk:
                    bits.append("PK")
                if col.is_fk:
                    bits.append("FK")
            suffix = f" [{' '.join(bits)}]" if bits else ""
            if output_format == "ids":
                lines.append(f"  {cid} = {col.name}{suffix}")
            else:
                lines.append(f"  - {col.name}{suffix}")
    return "\n".join(lines)


def build_user_content(
    question: str,
    candidates: Sequence[TableInfo],
    table_id: Mapping[str, str],
    column_id: Mapping[Tuple[str, str], str],
    retrieval_cfg: RetrievalConfig,
    output_format: str = "ids",
) -> str:
    schema_text = serialize_candidate_schema(candidates, table_id, column_id, retrieval_cfg, output_format)
    if output_format == "ids":
        instruction = (
            "Return ONLY JSON of table IDs to column-ID lists. Example: {\"T1\":[\"C2\"],\"T3\":[]} . "
            "Include a table ID with [] when the table is referenced but no specific column is used, such as COUNT(*). "
            "Do not include columns merely because they appear in the candidate schema."
        )
    else:
        instruction = (
            "Return ONLY JSON of table names to column-name lists. Include a table with [] when the table is referenced "
            "but no specific column is used, such as COUNT(*). Do not copy whole tables."
        )
    return f"{instruction}\n\n{schema_text}\n\nQuestion:\n{question}\n\nJSON answer:"


def target_json_from_links(
    links: Mapping[str, Any],
    schema: Mapping[str, List[str]],
    candidates: Sequence[TableInfo],
    table_id: Mapping[str, str],
    column_id: Mapping[Tuple[str, str], str],
    output_format: str = "ids",
) -> str:
    clean = normalize_links_to_schema(links, schema)
    candidate_tables = {t.name for t in candidates}
    out: Dict[str, List[str]] = {}
    for t, schema_cols in schema.items():
        if t not in clean or t not in candidate_tables:
            continue
        if output_format == "ids":
            key = table_id[t]
            vals = []
            for c in schema_cols:
                if c in clean[t] and (t, c) in column_id:
                    vals.append(column_id[(t, c)])
            out[key] = vals
        else:
            vals = [c for c in schema_cols if c in clean[t]]
            out[t] = vals
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def build_prompt_text(system_prompt: str, user_content: str, tokenizer: Any) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system_prompt}\n\nUser: {user_content}\n\nAssistant:"


def first_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    s = text.strip()
    try:
        obj, _ = decoder.raw_decode(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    m = JSON_OBJECT_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def coerce_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, str):
        return [value]
    return []


def parse_prediction_to_schema_links(
    response: str,
    schema: Mapping[str, List[str]],
    candidates: Sequence[TableInfo],
    table_id: Mapping[str, str],
    column_id: Mapping[Tuple[str, str], str],
    output_format: str = "ids",
) -> Dict[str, List[str]]:
    obj = first_json_object(response)
    if isinstance(obj.get("schema_links"), dict):
        obj = obj["schema_links"]
    if not isinstance(obj, dict):
        return {}

    id_to_table = {v.lower(): k for k, v in table_id.items()}
    id_to_col: Dict[Tuple[str, str], str] = {}
    for (t, c), cid in column_id.items():
        id_to_col[(t.lower(), cid.lower())] = c
    table_name_map = {t.lower(): t for t in schema}
    col_name_maps = {t.lower(): {c.lower(): c for c in cols} for t, cols in schema.items()}

    raw_links: Dict[str, List[str]] = {}
    for raw_t, raw_cols in obj.items():
        if not isinstance(raw_t, str):
            continue
        t_lc = raw_t.strip().lower()
        real_t = id_to_table.get(t_lc) or table_name_map.get(t_lc)
        if real_t is None:
            continue
        raw_links.setdefault(real_t, [])
        seen = {c.lower() for c in raw_links[real_t]}
        for raw_c in coerce_list(raw_cols):
            if not isinstance(raw_c, str):
                continue
            c_lc = raw_c.strip().lower()
            real_c = id_to_col.get((real_t.lower(), c_lc)) or col_name_maps.get(real_t.lower(), {}).get(c_lc)
            if real_c is not None and real_c.lower() not in seen:
                raw_links[real_t].append(real_c)
                seen.add(real_c.lower())
    return normalize_links_to_schema(raw_links, schema)


def make_example_payload(
    item: Mapping[str, Any],
    schemas_dir: str | os.PathLike[str],
    retrieval_cfg: RetrievalConfig,
    output_format: str = "ids",
    force_gold: bool = True,
) -> Dict[str, Any]:
    raw_schema = load_raw_schema(str(item["db_id"]), schemas_dir)
    schema = {t.name: [c.name for c in t.columns] for t in schema_to_tables(raw_schema)}
    candidates, tid, cid = build_candidate_schema(
        str(item["question"]),
        raw_schema,
        retrieval_cfg,
        force_links=item.get("schema_links") if force_gold else None,
    )
    system_prompt = SYSTEM_PROMPT_IDS if output_format == "ids" else SYSTEM_PROMPT_NAMES
    user_content = build_user_content(str(item["question"]), candidates, tid, cid, retrieval_cfg, output_format)
    assistant = ""
    if "schema_links" in item:
        assistant = target_json_from_links(item["schema_links"], schema, candidates, tid, cid, output_format)
    return {
        "question_id": item.get("question_id"),
        "db_id": item.get("db_id"),
        "question": item.get("question"),
        "system": system_prompt,
        "user": user_content,
        "assistant": assistant,
        "output_format": output_format,
    }


def load_run_config(adapter_dir: str | os.PathLike[str]) -> Dict[str, Any]:
    path = Path(adapter_dir) / "run_config.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_run_config(adapter_dir: str | os.PathLike[str], config: Mapping[str, Any]) -> None:
    path = Path(adapter_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(dict(config), f, indent=2, ensure_ascii=False)
