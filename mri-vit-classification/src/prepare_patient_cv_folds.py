"""Create leakage-free patient-level cross-validation folds from an ImageFolder dataset.

The source dataset is expected to contain ``train`` and ``val`` directories.
Those existing split labels are ignored: all patients are pooled, stratified by
patient label, and reassigned to exactly one validation fold. Images are
hard-linked by default so that the fold datasets do not duplicate MRI contents.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    patient_id: str
    class_name: str
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/repro_fl_t1_all_axial_patient_split_3class"),
        help="Existing patient-split ImageFolder dataset containing train/ and val/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cv5_all_axial_3class"),
        help="Destination root; fold1/ ... foldN/ will be created below it.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Delete and rebuild an existing output root.")
    return parser.parse_args()


def extract_patient_id(path: Path) -> str:
    patient_id = path.name.split("_", 1)[0].strip()
    if not patient_id:
        raise ValueError(f"Could not extract patient ID from: {path}")
    return patient_id


def parse_label(class_name: str) -> int:
    if not class_name.startswith("grade"):
        raise ValueError(f"Unexpected class directory: {class_name}")
    try:
        return int(class_name.removeprefix("grade"))
    except ValueError as exc:
        raise ValueError(f"Unexpected class directory: {class_name}") from exc


def scan_source(source_root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    seen_paths: set[Path] = set()
    class_names: set[str] = set()

    for split in ("train", "val"):
        split_root = source_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Required split directory not found: {split_root}")
        for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            class_name = class_dir.name
            label = parse_label(class_name)
            class_names.add(class_name)
            for path in sorted(class_dir.iterdir()):
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved in seen_paths:
                    raise ValueError(f"Duplicate source path: {path}")
                seen_paths.add(resolved)
                records.append(
                    ImageRecord(
                        path=resolved,
                        patient_id=extract_patient_id(path),
                        class_name=class_name,
                        label=label,
                    )
                )

    if not records:
        raise RuntimeError(f"No images found below: {source_root}")
    expected_classes = {f"grade{label}" for label in sorted({record.label for record in records})}
    if class_names != expected_classes:
        raise ValueError(f"Non-contiguous or inconsistent classes: {sorted(class_names)}")
    return records


def group_patients(records: Iterable[ImageRecord]) -> dict[str, list[ImageRecord]]:
    patient_records: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        patient_records[record.patient_id].append(record)

    for patient_id, rows in patient_records.items():
        labels = {row.label for row in rows}
        class_names = {row.class_name for row in rows}
        if len(labels) != 1 or len(class_names) != 1:
            raise ValueError(
                f"Patient {patient_id} has inconsistent labels/classes: "
                f"labels={sorted(labels)}, classes={sorted(class_names)}"
            )
    return dict(patient_records)


def assign_validation_folds(
    patient_records: dict[str, list[ImageRecord]],
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    if n_splits < 2:
        raise ValueError("--n-splits must be at least 2")

    patient_ids = sorted(patient_records)
    labels = [patient_records[patient_id][0].label for patient_id in patient_ids]
    class_counts = Counter(labels)
    if min(class_counts.values()) < n_splits:
        raise ValueError(
            f"Each class must contain at least n_splits patients: counts={dict(class_counts)}, "
            f"n_splits={n_splits}"
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignments: dict[str, int] = {}
    for fold_index, (_train_indices, val_indices) in enumerate(splitter.split(patient_ids, labels), start=1):
        for index in val_indices:
            patient_id = patient_ids[int(index)]
            if patient_id in assignments:
                raise AssertionError(f"Patient assigned to multiple validation folds: {patient_id}")
            assignments[patient_id] = fold_index

    if set(assignments) != set(patient_ids):
        raise AssertionError("Not every patient was assigned to exactly one validation fold")
    return assignments


def create_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Destination collision: {destination}")
    try:
        os.link(source, destination)
    except OSError as exc:
        raise OSError(
            f"Could not hard-link {source} -> {destination}. "
            "Source and destination must be on the same filesystem."
        ) from exc


def build_fold_tree(
    build_root: Path,
    patient_records: dict[str, list[ImageRecord]],
    assignments: dict[str, int],
    n_splits: int,
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    patient_ids = set(patient_records)

    for fold_index in range(1, n_splits + 1):
        val_patients = {pid for pid, assigned_fold in assignments.items() if assigned_fold == fold_index}
        train_patients = patient_ids - val_patients
        if train_patients & val_patients:
            raise AssertionError(f"Patient leakage detected before writing fold {fold_index}")
        if train_patients | val_patients != patient_ids:
            raise AssertionError(f"Patient coverage error in fold {fold_index}")

        for split_name, split_patients in (("train", train_patients), ("val", val_patients)):
            per_class_patients: Counter[str] = Counter()
            per_class_images: Counter[str] = Counter()
            for patient_id in sorted(split_patients):
                rows = patient_records[patient_id]
                class_name = rows[0].class_name
                per_class_patients[class_name] += 1
                for row in rows:
                    destination = build_root / f"fold{fold_index}" / split_name / class_name / row.path.name
                    create_hardlink(row.path, destination)
                    per_class_images[class_name] += 1

            for class_name in sorted(per_class_patients):
                summary.append(
                    {
                        "fold": fold_index,
                        "split": split_name,
                        "class_name": class_name,
                        "num_patients": per_class_patients[class_name],
                        "num_images": per_class_images[class_name],
                    }
                )
    return summary


def validate_fold_tree(
    output_root: Path,
    patient_records: dict[str, list[ImageRecord]],
    assignments: dict[str, int],
    n_splits: int,
) -> None:
    all_patient_ids = set(patient_records)
    validation_occurrences: Counter[str] = Counter()

    for fold_index in range(1, n_splits + 1):
        fold_root = output_root / f"fold{fold_index}"
        split_patients: dict[str, set[str]] = {}
        split_images: dict[str, int] = {}
        for split_name in ("train", "val"):
            root = fold_root / split_name
            if not root.is_dir():
                raise FileNotFoundError(f"Missing generated split: {root}")
            paths = [path for class_dir in root.iterdir() if class_dir.is_dir() for path in class_dir.iterdir() if path.is_file()]
            patients = {extract_patient_id(path) for path in paths}
            split_patients[split_name] = patients
            split_images[split_name] = len(paths)

        overlap = split_patients["train"] & split_patients["val"]
        if overlap:
            raise AssertionError(f"Fold {fold_index} patient leakage: {sorted(overlap)[:5]}")
        if split_patients["train"] | split_patients["val"] != all_patient_ids:
            raise AssertionError(f"Fold {fold_index} does not cover all patients")
        expected_val = {pid for pid, fold in assignments.items() if fold == fold_index}
        if split_patients["val"] != expected_val:
            raise AssertionError(f"Fold {fold_index} validation assignment mismatch")
        for patient_id in split_patients["val"]:
            validation_occurrences[patient_id] += 1

        expected_total_images = sum(len(rows) for rows in patient_records.values())
        if split_images["train"] + split_images["val"] != expected_total_images:
            raise AssertionError(f"Fold {fold_index} image coverage mismatch")

    invalid_occurrences = {pid: count for pid, count in validation_occurrences.items() if count != 1}
    if invalid_occurrences or set(validation_occurrences) != all_patient_ids:
        raise AssertionError(
            "Each patient must occur in validation exactly once: "
            f"invalid={list(invalid_occurrences.items())[:5]}"
        )


def write_metadata(
    output_root: Path,
    final_output_root: Path,
    source_root: Path,
    patient_records: dict[str, list[ImageRecord]],
    assignments: dict[str, int],
    summary: list[dict[str, object]],
    n_splits: int,
    seed: int,
) -> None:
    with (output_root / "patient_fold_assignments.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "class_name", "label", "validation_fold"])
        writer.writeheader()
        for patient_id in sorted(patient_records):
            row = patient_records[patient_id][0]
            writer.writerow(
                {
                    "patient_id": patient_id,
                    "class_name": row.class_name,
                    "label": row.label,
                    "validation_fold": assignments[patient_id],
                }
            )

    with (output_root / "summary_folds.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "split", "class_name", "num_patients", "num_images"],
        )
        writer.writeheader()
        writer.writerows(summary)

    patient_class_counts = Counter(rows[0].class_name for rows in patient_records.values())
    metadata = {
        "source_root": str(source_root.resolve()),
        "output_root": str(final_output_root.resolve()),
        "split_unit": "patient",
        "splitter": "StratifiedKFold",
        "n_splits": n_splits,
        "shuffle": True,
        "seed": seed,
        "num_patients": len(patient_records),
        "num_images": sum(len(rows) for rows in patient_records.values()),
        "patient_class_counts": dict(sorted(patient_class_counts.items())),
        "hardlinks": True,
        "all_axial": True,
        "validation_guarantees": {
            "train_val_patient_overlap_per_fold": 0,
            "each_patient_validation_fold_count": 1,
        },
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    build_root = output_root.with_name(f".{output_root.name}.building")

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists; use --force to rebuild: {output_root}")
        shutil.rmtree(output_root)
    if build_root.exists():
        if not args.force:
            raise FileExistsError(f"Incomplete build exists; use --force to remove it: {build_root}")
        shutil.rmtree(build_root)

    records = scan_source(source_root)
    patient_records = group_patients(records)
    assignments = assign_validation_folds(patient_records, n_splits=args.n_splits, seed=args.seed)

    try:
        summary = build_fold_tree(
            build_root=build_root,
            patient_records=patient_records,
            assignments=assignments,
            n_splits=args.n_splits,
        )
        write_metadata(
            output_root=build_root,
            final_output_root=output_root,
            source_root=source_root,
            patient_records=patient_records,
            assignments=assignments,
            summary=summary,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        validate_fold_tree(
            output_root=build_root,
            patient_records=patient_records,
            assignments=assignments,
            n_splits=args.n_splits,
        )
        build_root.replace(output_root)
    except Exception:
        print(f"[ERROR] Incomplete build retained for inspection: {build_root}")
        raise

    print(f"Created {args.n_splits} patient-level folds at: {output_root}")
    print(f"Patients: {len(patient_records)}")
    print(f"Images per fold (train + val): {len(records)}")
    print("Verified: zero train/val patient overlap in every fold")
    print("Verified: every patient appears in validation exactly once")


if __name__ == "__main__":
    main()
