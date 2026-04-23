import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sklearn.model_selection import train_test_split


def _can_stratify(labels: Sequence[int]) -> bool:
    counts = Counter(labels)
    return len(counts) >= 2 and min(counts.values()) >= 2


def _majority_label(labels: Sequence[int]) -> int:
    return Counter(labels).most_common(1)[0][0]


def _resolve_modality(filename: str, source: str, modality_col_value: str) -> str:
    if source == "filename_prefix":
        # Example: FL_0000.jpg -> FL
        return filename.split("_", 1)[0].upper()
    return modality_col_value.strip().upper()


def _resolve_source_path(image_root: Path, modality: str, filename: str) -> Optional[Path]:
    candidates = [
        image_root / modality / filename,
        image_root / filename,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _safe_split(
    items: List[str],
    labels: List[int],
    test_size: float,
    seed: int,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    if not items or test_size <= 0:
        return items, [], labels, []
    if test_size >= 1:
        return [], items, [], labels

    stratify = labels if _can_stratify(labels) else None
    try:
        train_items, test_items, train_labels, test_labels = train_test_split(
            items,
            labels,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        return list(train_items), list(test_items), list(train_labels), list(test_labels)
    except ValueError:
        train_items, test_items, train_labels, test_labels = train_test_split(
            items,
            labels,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
        return list(train_items), list(test_items), list(train_labels), list(test_labels)


def _split_patient_ids(
    patient_to_label: Dict[str, int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, set]:
    patient_ids = sorted(patient_to_label.keys())
    labels = [patient_to_label[pid] for pid in patient_ids]

    if len(patient_ids) <= 1:
        return {"train": set(patient_ids), "val": set(), "test": set()}

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    train_ids, temp_ids, _train_labels, temp_labels = _safe_split(
        items=patient_ids,
        labels=labels,
        test_size=1.0 - train_ratio,
        seed=seed,
    )

    if not temp_ids:
        return {"train": set(train_ids), "val": set(), "test": set()}

    if val_ratio == 0 and test_ratio > 0:
        return {"train": set(train_ids), "val": set(), "test": set(temp_ids)}

    if test_ratio == 0 and val_ratio > 0:
        return {"train": set(train_ids), "val": set(temp_ids), "test": set()}

    remain_ratio = val_ratio + test_ratio
    if remain_ratio == 0:
        return {"train": set(train_ids), "val": set(), "test": set()}

    # Within temp_ids, keep the original val:test proportion.
    temp_val_share = val_ratio / remain_ratio
    val_ids, test_ids, _val_labels, _test_labels = _safe_split(
        items=temp_ids,
        labels=temp_labels,
        test_size=1.0 - temp_val_share,
        seed=seed,
    )

    return {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build modality-wise grade classification dataset from CSV metadata.",
    )
    parser.add_argument("--csv-path", type=str, required=True, help="Path to metadata CSV file.")
    parser.add_argument("--image-root", type=str, required=True, help="Root directory for raw images.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/grade_by_modality",
        help="Output root for ImageFolder-style train/val/test directories.",
    )
    parser.add_argument("--name-col", type=str, default="name", help="CSV column for image filename.")
    parser.add_argument("--id-col", type=str, default="ID", help="CSV column for patient ID.")
    parser.add_argument("--label-col", type=str, default="wm", help="CSV column for grade label (0-4).")
    parser.add_argument(
        "--modality-source",
        type=str,
        choices=["filename_prefix", "column"],
        default="filename_prefix",
        help="How to resolve modality (FL/T1/T2).",
    )
    parser.add_argument(
        "--modality-col",
        type=str,
        default="",
        help="CSV column for modality when --modality-source column is used.",
    )
    parser.add_argument(
        "--include-modalities",
        nargs="+",
        default=["FL", "T1", "T2"],
        help="Modalities to export.",
    )
    parser.add_argument(
        "--fixed-modality",
        type=str,
        default="",
        help="Force all rows to this modality (e.g., FL, T1, T2).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-mode",
        type=str,
        choices=["copy", "move"],
        default="copy",
        help="Whether to copy or move files when constructing dataset.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing modality output directories before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    csv_path = Path(args.csv_path)

    if not image_root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")
    if not csv_path.exists():
        raise FileNotFoundError(f"csv_path not found: {csv_path}")
    if args.modality_source == "column" and not args.modality_col:
        raise ValueError("--modality-col is required when --modality-source column is used")
    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio must be in (0, 1)")
    if not (0 <= args.val_ratio < 1):
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.train_ratio + args.val_ratio > 1:
        raise ValueError("train_ratio + val_ratio must be <= 1")
    fixed_modality = args.fixed_modality.strip().upper()
    if fixed_modality and fixed_modality not in {"FL", "T1", "T2"}:
        raise ValueError("--fixed-modality must be one of FL, T1, T2")

    modalities = {m.upper() for m in args.include_modalities}
    entries_by_modality: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    missing_files: List[Tuple[int, str]] = []
    invalid_rows: List[Tuple[int, str]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            filename = (row.get(args.name_col) or "").strip()
            patient_id = str((row.get(args.id_col) or "").strip())
            raw_label = (row.get(args.label_col) or "").strip()

            if not filename or not patient_id or not raw_label:
                invalid_rows.append((line_no, "missing required columns"))
                continue

            try:
                label = int(float(raw_label))
            except ValueError:
                invalid_rows.append((line_no, f"invalid label: {raw_label}"))
                continue

            if label < 0:
                invalid_rows.append((line_no, f"negative label: {label}"))
                continue

            if fixed_modality:
                modality = fixed_modality
            else:
                modality_col_value = row.get(args.modality_col, "") if args.modality_col else ""
                modality = _resolve_modality(
                    filename=filename,
                    source=args.modality_source,
                    modality_col_value=modality_col_value,
                )

            if modality not in modalities:
                continue

            src_path = _resolve_source_path(image_root=image_root, modality=modality, filename=filename)
            if src_path is None:
                missing_files.append((line_no, f"{modality}/{filename}"))
                continue

            entries_by_modality[modality].append(
                {
                    "patient_id": patient_id,
                    "label": str(label),
                    "filename": filename,
                    "src_path": str(src_path),
                }
            )

    if not entries_by_modality:
        raise RuntimeError("No valid entries found. Check CSV columns, labels, and image paths.")

    summary_rows: List[Dict[str, str]] = []

    for modality in sorted(entries_by_modality.keys()):
        modality_entries = entries_by_modality[modality]

        if args.clean_output:
            modality_out = output_root / modality
            if modality_out.exists():
                shutil.rmtree(modality_out)

        patient_to_labels: Dict[str, List[int]] = defaultdict(list)
        for item in modality_entries:
            patient_to_labels[item["patient_id"]].append(int(item["label"]))

        patient_to_label = {pid: _majority_label(labels) for pid, labels in patient_to_labels.items()}
        split_patients = _split_patient_ids(
            patient_to_label=patient_to_label,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        split_counts: Dict[str, Counter] = {
            "train": Counter(),
            "val": Counter(),
            "test": Counter(),
        }

        for item in modality_entries:
            patient_id = item["patient_id"]
            split_name = "train"
            if patient_id in split_patients["val"]:
                split_name = "val"
            elif patient_id in split_patients["test"]:
                split_name = "test"

            label = int(item["label"])
            src_path = Path(item["src_path"])
            class_dir = output_root / modality / split_name / f"grade{label}"
            class_dir.mkdir(parents=True, exist_ok=True)

            # Prefix with patient ID to reduce filename collision risk.
            dest_name = f"{patient_id}_{src_path.name}"
            dst_path = _next_available_path(class_dir / dest_name)

            if args.copy_mode == "move":
                shutil.move(str(src_path), str(dst_path))
            else:
                shutil.copy2(src_path, dst_path)

            split_counts[split_name][label] += 1

        for split_name in ["train", "val", "test"]:
            for label, count in sorted(split_counts[split_name].items()):
                summary_rows.append(
                    {
                        "modality": modality,
                        "split": split_name,
                        "label": str(label),
                        "count": str(count),
                    }
                )

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary_counts.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["modality", "split", "label", "count"])
        writer.writeheader()
        writer.writerows(summary_rows)

    if invalid_rows:
        print(f"[WARN] Skipped invalid rows: {len(invalid_rows)}")
        for line_no, reason in invalid_rows[:10]:
            print(f"  line {line_no}: {reason}")

    if missing_files:
        print(f"[WARN] Missing files: {len(missing_files)}")
        for line_no, rel_path in missing_files[:10]:
            print(f"  line {line_no}: {rel_path}")

    print("Done. Summary saved to:", summary_path)


if __name__ == "__main__":
    main()
