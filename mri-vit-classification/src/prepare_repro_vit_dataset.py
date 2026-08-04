from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from sklearn.model_selection import train_test_split


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _resolve_modality(filename: str) -> str:
    return filename.split("_", 1)[0].upper()


def _can_stratify(labels: Sequence[int]) -> bool:
    counts = Counter(labels)
    return len(counts) >= 2 and min(counts.values()) >= 2


def _safe_split(
    items: List[str],
    labels: List[int],
    test_size: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if not items or test_size <= 0:
        return items, []
    if test_size >= 1:
        return [], items

    stratify = labels if _can_stratify(labels) else None
    try:
        train_items, val_items = train_test_split(
            items,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        train_items, val_items = train_test_split(
            items,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    return list(train_items), list(val_items)


def _split_entries(
    entries: List[Dict[str, str]],
    val_ratio: float,
    seed: int,
    split_mode: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if split_mode == "image":
        train_entries, val_entries = train_test_split(
            entries,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
        return list(train_entries), list(val_entries)

    # Patient-level split keeps slices from one patient in a single split.
    patient_to_entries: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for item in entries:
        patient_to_entries[item["patient_id"]].append(item)

    patient_ids = sorted(patient_to_entries.keys())
    patient_labels: List[int] = []
    for pid in patient_ids:
        labels = [int(sample["label"]) for sample in patient_to_entries[pid]]
        majority_label = Counter(labels).most_common(1)[0][0]
        patient_labels.append(majority_label)

    train_patient_ids, val_patient_ids = _safe_split(
        items=patient_ids,
        labels=patient_labels,
        test_size=val_ratio,
        seed=seed,
    )

    train_set = set(train_patient_ids)
    val_set = set(val_patient_ids)

    train_entries = [item for item in entries if item["patient_id"] in train_set]
    val_entries = [item for item in entries if item["patient_id"] in val_set]
    return train_entries, val_entries


def _load_entries(
    csv_path: Path,
    image_root: Path,
    name_col: str,
    id_col: str,
    label_col: str,
    axial_col: str,
    axial_min: int | None,
    axial_max: int | None,
    missing_files: List[str],
    invalid_rows: List[str],
) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            filename = (row.get(name_col) or "").strip()
            patient_id = str((row.get(id_col) or "").strip())
            raw_label = (row.get(label_col) or "").strip()

            if not filename or not patient_id or not raw_label:
                invalid_rows.append(f"{csv_path.name}:{line_no} missing required columns")
                continue

            try:
                label = int(float(raw_label))
            except ValueError:
                invalid_rows.append(f"{csv_path.name}:{line_no} invalid label={raw_label}")
                continue

            axial_raw = row.get(axial_col)
            axial_val = _parse_int(axial_raw)
            if axial_val is None:
                invalid_rows.append(f"{csv_path.name}:{line_no} invalid axial={axial_raw}")
                continue
            if axial_min is not None and axial_val < axial_min:
                continue
            if axial_max is not None and axial_val > axial_max:
                continue

            modality = _resolve_modality(filename)
            src_path = image_root / modality / filename
            if not src_path.exists():
                missing_files.append(f"{csv_path.name}:{line_no} {modality}/{filename}")
                continue

            entries.append(
                {
                    "patient_id": patient_id,
                    "label": str(label),
                    "filename": filename,
                    "src_path": str(src_path),
                }
            )

    return entries


def _copy_entries(entries: List[Dict[str, str]], split_name: str, output_root: Path) -> None:
    for item in entries:
        label = item["label"]
        src_path = Path(item["src_path"])
        class_dir = output_root / split_name / f"grade{label}"
        class_dir.mkdir(parents=True, exist_ok=True)

        dest_name = f"{item['patient_id']}_{src_path.name}"
        dst_path = class_dir / dest_name
        if dst_path.exists():
            dst_path = class_dir / f"{item['patient_id']}_{src_path.stem}_{random.randint(0, 9999)}{src_path.suffix}"

        shutil.copy2(src_path, dst_path)
        

def _count_split(output_root: Path, split_name: str) -> Counter:
    counts: Counter = Counter()
    split_root = output_root / split_name
    if not split_root.exists():
        return counts
    for class_dir in split_root.iterdir():
        if not class_dir.is_dir() or not class_dir.name.startswith("grade"):
            continue
        label_str = class_dir.name.replace("grade", "", 1)
        try:
            label = int(label_str)
        except ValueError:
            continue
        counts[label] = sum(1 for p in class_dir.iterdir() if p.is_file())
    return counts


def _count_unique_patients(entries: List[Dict[str, str]]) -> int:
    return len({item["patient_id"] for item in entries})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FL+T1 grade dataset for ViT reproduction experiment.",
    )
    parser.add_argument("--csv-paths", nargs="+", required=True, help="CSV files to merge (e.g., FL and T1).")
    parser.add_argument("--image-root", type=str, required=True, help="Root directory of labeled images.")
    parser.add_argument("--output-root", type=str, required=True, help="Output directory for ImageFolder dataset.")
    parser.add_argument("--name-col", type=str, default="name")
    parser.add_argument("--id-col", type=str, default="ID")
    parser.add_argument("--label-col", type=str, default="wm")
    parser.add_argument("--axial-col", type=str, default="axial")
    parser.add_argument("--axial-min", type=int, default=None)
    parser.add_argument("--axial-max", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["image", "patient"],
        default="image",
        help="Split unit: image-level or patient-level.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Skip copying train split.")
    parser.add_argument("--skip-val", action="store_true", help="Skip copying val split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_paths = [Path(p) for p in args.csv_paths]
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)

    if not image_root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"csv_path not found: {csv_path}")

    if args.val_ratio <= 0 or args.val_ratio >= 1:
        raise ValueError("val_ratio must be in (0, 1)")

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)

    random.seed(args.seed)

    missing_files: List[str] = []
    invalid_rows: List[str] = []
    entries: List[Dict[str, str]] = []
    for csv_path in csv_paths:
        entries.extend(
            _load_entries(
                csv_path=csv_path,
                image_root=image_root,
                name_col=args.name_col,
                id_col=args.id_col,
                label_col=args.label_col,
                axial_col=args.axial_col,
                axial_min=args.axial_min,
                axial_max=args.axial_max,
                missing_files=missing_files,
                invalid_rows=invalid_rows,
            )
        )

    if not entries:
        raise RuntimeError("No valid entries found. Check CSVs and image paths.")

    train_entries, val_entries = _split_entries(
        entries=entries,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_mode=args.split_mode,
    )

    if not args.skip_train:
        _copy_entries(train_entries, "train", output_root)
    if not args.skip_val:
        _copy_entries(val_entries, "val", output_root)

    train_counts = _count_split(output_root, "train")
    val_counts = _count_split(output_root, "val")

    summary_path = output_root / "summary_counts.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label", "count"])
        writer.writeheader()
        for split_name, counts in [("train", train_counts), ("val", val_counts)]:
            for label, count in sorted(counts.items()):
                writer.writerow({"split": split_name, "label": str(label), "count": str(count)})

    split_summary_path = output_root / "summary_splits.csv"
    with split_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "num_images", "num_patients"])
        writer.writeheader()
        rows = [
            {
                "split": "train",
                "num_images": str(len(train_entries)),
                "num_patients": str(_count_unique_patients(train_entries)),
            },
            {
                "split": "val",
                "num_images": str(len(val_entries)),
                "num_patients": str(_count_unique_patients(val_entries)),
            },
        ]
        writer.writerows(rows)

    if invalid_rows:
        print(f"[WARN] Skipped invalid rows: {len(invalid_rows)}")
        for msg in invalid_rows[:10]:
            print(f"  {msg}")

    if missing_files:
        print(f"[WARN] Missing files: {len(missing_files)}")
        for msg in missing_files[:10]:
            print(f"  {msg}")

    print("Split mode:", args.split_mode)
    print("Train: images=", len(train_entries), "patients=", _count_unique_patients(train_entries))
    print("Val: images=", len(val_entries), "patients=", _count_unique_patients(val_entries))
    print("Done. Summary saved to:", summary_path)
    print("Split summary saved to:", split_summary_path)


if __name__ == "__main__":
    main()
