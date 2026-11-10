# -*- coding: utf-8 -*-
"""Rebuild the translation SFT datasets from the cleaned gold corpus."""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


TASKS = (
    "darija_ar_to_english",
    "darija_latn_to_english",
    "english_to_darija_ar",
    "english_to_darija_latn",
)


def parse_args():
    parser = argparse.ArgumentParser(description="ETL Darija gold corpus -> SFT JSONL.")
    parser.add_argument("--gold-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-per-task", type=int, default=5000)
    parser.add_argument("--validation-per-task", type=int, default=500)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize(value):
    return re.sub(r"\s+", " ", clean(value).lower())


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def make_examples(records):
    examples = []
    for row in records:
        arabic = clean(row.get("darija_ar"))
        franco = clean(row.get("darija_latn"))
        english = clean(row.get("english"))
        source = clean(row.get("dataset_source")) or "unknown"
        if not arabic or not english:
            continue

        examples.append(
            {
                "messages": [
                    {"role": "user", "content": f"ترجم هاد العبارة للإنجليزية: {arabic}"},
                    {"role": "assistant", "content": english},
                ],
                "task_type": "darija_ar_to_english",
                "dataset_source": source,
            }
        )
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": f"ترجم هاد الجملة للدارجة المغربية: {english}"},
                    {"role": "assistant", "content": arabic},
                ],
                "task_type": "english_to_darija_ar",
                "dataset_source": source,
            }
        )
        if franco:
            examples.extend(
                [
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Trjem had l3ibara l English: {franco}",
                            },
                            {"role": "assistant", "content": english},
                        ],
                        "task_type": "darija_latn_to_english",
                        "dataset_source": source,
                    },
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Trjem had sentence l Darija b Franco: {english}",
                            },
                            {"role": "assistant", "content": franco},
                        ],
                        "task_type": "english_to_darija_latn",
                        "dataset_source": source,
                    },
                ]
            )
    return examples


def example_key(example):
    return json.dumps(example["messages"], ensure_ascii=False, sort_keys=True)


def group_key(example):
    user = example["messages"][0]["content"]
    assistant = example["messages"][1]["content"]
    if example["task_type"] in {"darija_ar_to_english", "darija_latn_to_english"}:
        return normalize(assistant)
    return normalize(user.split(":", 1)[-1])


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    args.gold_file = args.gold_file or project_root / "data/gold/darija_unified_gold_clean.jsonl"
    args.output_dir = args.output_dir or project_root / "data/gold/sft"
    if not args.gold_file.is_absolute():
        args.gold_file = project_root / args.gold_file
    if not args.output_dir.is_absolute():
        args.output_dir = project_root / args.output_dir
    if not args.gold_file.exists():
        raise FileNotFoundError(f"Gold corpus not found: {args.gold_file}")
    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = make_examples(read_jsonl(args.gold_file))
    unique = {}
    source_map = defaultdict(set)
    for example in candidates:
        key = example_key(example)
        source_map[key].add(example["dataset_source"])
        unique.setdefault(key, example)

    deduplicated = list(unique.values())
    for example in deduplicated:
        example["dataset_sources"] = sorted(source_map[example_key(example)])

    candidates_path = args.output_dir / "darija_translation_sft_candidates.jsonl"
    dedup_path = args.output_dir / "darija_translation_sft_deduplicated.jsonl"
    write_jsonl(candidates_path, candidates)
    write_jsonl(dedup_path, deduplicated)

    groups = defaultdict(list)
    for example in deduplicated:
        groups[group_key(example)].append(example)
    group_names = list(groups)
    random.Random(args.seed).shuffle(group_names)
    validation_group_count = max(1, round(len(group_names) * args.validation_ratio))
    validation_groups = set(group_names[:validation_group_count])

    train_by_task = defaultdict(list)
    validation_by_task = defaultdict(list)
    for example in deduplicated:
        target = validation_by_task if group_key(example) in validation_groups else train_by_task
        target[example["task_type"]].append(example)

    rng = random.Random(args.seed)
    train = []
    validation = []
    for task in TASKS:
        train_rows = train_by_task[task][:]
        validation_rows = validation_by_task[task][:]
        rng.shuffle(train_rows)
        rng.shuffle(validation_rows)
        train.extend(train_rows[: args.train_per_task])
        validation.extend(validation_rows[: args.validation_per_task])
    rng.shuffle(train)
    rng.shuffle(validation)

    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    report = {
        "gold_records": sum(1 for _ in read_jsonl(args.gold_file)),
        "candidate_examples": len(candidates),
        "deduplicated_examples": len(deduplicated),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_file": str(train_path),
        "validation_file": str(validation_path),
        "seed": args.seed,
    }
    report_path = project_root / "reports" / "etl_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
