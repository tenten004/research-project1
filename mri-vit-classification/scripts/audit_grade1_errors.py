"""Technical grade-1 OOF error audit and blinded clinical review package.

The script performs only deterministic data/model-output checks. It does not
change teacher labels or make a medical judgement. Clinical re-grading must be
performed independently by qualified reviewers using the blinded materials.

All patient-level and image-derived outputs are local-only and must not be
staged or pushed to Git.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
CV_ROOT = PROJECT_ROOT / "data/cv5_all_axial_3class"
OOF_ROOT = PROJECT_ROOT / "outputs/cv5_all_axial_3class/oof_summary"
DETAIL_ROOT = PROJECT_ROOT / "outputs/cv5_all_axial_3class/oof_detailed_analysis"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/cv5_all_axial_3class/grade1_audit"
DEFAULT_TEACHER_CSVS = [
    WORKSPACE_ROOT / "教師データ/labeled_image_list_FL_preprocess.csv",
    WORKSPACE_ROOT / "教師データ/labeled_image_list_T1_preprocess.csv",
]
CLASS_LABELS = {0: "grade0", 1: "grade1", 2: "grade2+"}
ERROR_CATEGORIES = ["both_correct", "vit_only_correct", "cnn_only_correct", "both_wrong"]


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    fold: int
    target: int
    patient_id: str
    modality: str
    axial: int | None
    source_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", type=Path, default=CV_ROOT)
    parser.add_argument("--cnn-csv", type=Path, default=OOF_ROOT / "resnet18_oof_patients.csv")
    parser.add_argument("--vit-csv", type=Path, default=OOF_ROOT / "vit_oof_patients.csv")
    parser.add_argument("--error-csv", type=Path, default=DETAIL_ROOT / "patient_error_analysis.csv")
    parser.add_argument("--teacher-csvs", nargs="+", type=Path, default=DEFAULT_TEACHER_CSVS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--high-confidence", type=float, default=0.80)
    parser.add_argument("--review-cases", type=int, default=200)
    parser.add_argument("--skip-contact-sheets", action="store_true")
    return parser.parse_args()


def normalize_patient_id(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("Empty patient ID")
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return value


def split_semicolon(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def extract_patient_id(path: Path) -> str:
    patient_id = path.name.split("_", 1)[0].strip()
    if not patient_id:
        raise ValueError(f"Cannot extract patient ID: {path}")
    return normalize_patient_id(patient_id)


def extract_modality(path: Path) -> str:
    parts = path.name.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def extract_axial(path: Path) -> int | None:
    original_stem = path.stem.rsplit("-", 1)[0]
    token = original_stem.rsplit("_", 1)[-1]
    return int(token) if token.isdigit() else None


def original_source_name(path: Path) -> str:
    parts = path.name.split("_", 1)
    return parts[1] if len(parts) == 2 else path.name


def parse_target(class_name: str) -> int:
    if not class_name.startswith("grade"):
        raise ValueError(f"Unexpected class directory: {class_name}")
    return int(class_name.removeprefix("grade"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_oof(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for raw in load_csv_rows(path):
        patient_id = normalize_patient_id(raw["patient_id"])
        if patient_id in rows:
            raise ValueError(f"Duplicate patient in {path}: {patient_id}")
        probs = np.asarray(
            [float(raw["prob_class0"]), float(raw["prob_class1"]), float(raw["prob_class2"])],
            dtype=np.float64,
        )
        if not np.isclose(probs.sum(), 1.0, atol=1e-4):
            raise ValueError(f"Probabilities do not sum to one: {path}, patient={patient_id}")
        rows[patient_id] = {
            "fold": int(raw["cv_fold"]),
            "target": int(raw["target"]),
            "pred": int(raw["pred"]),
            "probs": probs,
            "num_slices": int(raw["num_slices"]),
            "selected_paths": split_semicolon(raw.get("selected_paths", "")),
            "selected_modalities": split_semicolon(raw.get("selected_modalities", "")),
            "selected_slice_indices": split_semicolon(raw.get("selected_slice_indices", "")),
            "selected_confidences": [
                float(value) for value in split_semicolon(raw.get("selected_confidences", ""))
            ],
        }
    return rows


def scan_cv_validation(cv_root: Path) -> tuple[dict[str, list[ImageRecord]], list[str]]:
    records: dict[str, list[ImageRecord]] = defaultdict(list)
    errors: list[str] = []
    seen_paths: set[Path] = set()
    for fold in range(1, 6):
        val_root = cv_root / f"fold{fold}" / "val"
        if not val_root.is_dir():
            raise FileNotFoundError(val_root)
        for class_dir in sorted(path for path in val_root.iterdir() if path.is_dir()):
            target = parse_target(class_dir.name)
            for path in sorted(class_dir.iterdir()):
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved in seen_paths:
                    errors.append(f"duplicate_validation_path:{path}")
                    continue
                seen_paths.add(resolved)
                patient_id = extract_patient_id(path)
                records[patient_id].append(
                    ImageRecord(
                        path=resolved,
                        fold=fold,
                        target=target,
                        patient_id=patient_id,
                        modality=extract_modality(path),
                        axial=extract_axial(path),
                        source_name=original_source_name(path),
                    )
                )
    return dict(records), errors


def load_teacher_data(paths: list[Path]) -> dict[str, dict[str, Any]]:
    patients: dict[str, dict[str, Any]] = {}
    for path in paths:
        modality = "FL" if "_FL_" in path.name else "T1" if "_T1_" in path.name else path.stem
        for row in load_csv_rows(path):
            patient_id = normalize_patient_id(row["ID"])
            patient = patients.setdefault(
                patient_id,
                {"grades": set(), "names": set(), "by_modality": defaultdict(list)},
            )
            grade = int(float(row["wm"]))
            axial = int(float(row["axial"]))
            name = row["name"].strip()
            patient["grades"].add(grade)
            patient["names"].add(name)
            patient["by_modality"][modality].append({"name": name, "grade": grade, "axial": axial})
    return patients


def prediction_margin(probs: np.ndarray) -> float:
    ordered = np.sort(probs)
    return float(ordered[-1] - ordered[-2])


def normalized_entropy(probs: np.ndarray) -> float:
    safe = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(safe * np.log(safe)) / math.log(len(probs)))


def axial_gap_count(indices: Iterable[int | None]) -> int:
    known = sorted({value for value in indices if value is not None})
    if len(known) < 2:
        return 0
    return len(set(range(known[0], known[-1] + 1)) - set(known))


def inspect_image_files(records: list[ImageRecord]) -> tuple[int, str]:
    unreadable = 0
    sizes: Counter[tuple[int, int]] = Counter()
    for record in records:
        try:
            with Image.open(record.path) as image:
                image.verify()
            with Image.open(record.path) as image:
                sizes[image.size] += 1
        except Exception:
            unreadable += 1
    size_text = ";".join(f"{width}x{height}:{count}" for (width, height), count in sorted(sizes.items()))
    return unreadable, size_text


def validate_selected(
    row: dict[str, Any],
    patient_records: list[ImageRecord],
) -> dict[str, Any]:
    paths = row["selected_paths"]
    modalities = row["selected_modalities"]
    indices = row["selected_slice_indices"]
    confidences = row["selected_confidences"]
    lengths = [len(paths), len(modalities), len(indices), len(confidences)]
    patient_names = {record.path.name for record in patient_records}
    missing_path_count = 0
    wrong_patient_count = 0
    metadata_mismatch_count = 0
    for position, raw_path in enumerate(paths):
        selected_path = Path(raw_path.replace("\\", "/"))
        if not selected_path.is_absolute():
            selected_path = PROJECT_ROOT / selected_path
        if not selected_path.exists():
            missing_path_count += 1
        if extract_patient_id(selected_path) != patient_records[0].patient_id:
            wrong_patient_count += 1
        if selected_path.name not in patient_names:
            metadata_mismatch_count += 1
        if position < len(modalities) and extract_modality(selected_path) != modalities[position]:
            metadata_mismatch_count += 1
        if position < len(indices):
            actual_index = extract_axial(selected_path)
            expected = None if indices[position] == "unknown" else int(indices[position])
            if actual_index != expected:
                metadata_mismatch_count += 1
    return {
        "selected_length_consistent": len(set(lengths)) == 1,
        "selected_count": len(paths),
        "selected_missing_path_count": missing_path_count,
        "selected_wrong_patient_count": wrong_patient_count,
        "selected_metadata_mismatch_count": metadata_mismatch_count,
        "selected_confidence_mean": float(np.mean(confidences)) if confidences else float("nan"),
        "selected_confidence_min": float(np.min(confidences)) if confidences else float("nan"),
        "selected_fl_fraction": modalities.count("FL") / len(modalities) if modalities else float("nan"),
        "selected_axial_9_15_fraction": (
            sum(value != "unknown" and 9 <= int(value) <= 15 for value in indices) / len(indices)
            if indices
            else float("nan")
        ),
    }


def classify_error(target: int, cnn_pred: int, vit_pred: int) -> tuple[str, str]:
    cnn_correct = cnn_pred == target
    vit_correct = vit_pred == target
    if cnn_correct and vit_correct:
        return "both_correct", "both_correct"
    if not cnn_correct and vit_correct:
        return "vit_only_correct", f"cnn_to_{CLASS_LABELS[cnn_pred]}"
    if cnn_correct and not vit_correct:
        return "cnn_only_correct", f"vit_to_{CLASS_LABELS[vit_pred]}"
    if cnn_pred == vit_pred:
        return "both_wrong", f"both_to_{CLASS_LABELS[cnn_pred]}"
    return "both_wrong", f"opposite_cnn_{CLASS_LABELS[cnn_pred]}_vit_{CLASS_LABELS[vit_pred]}"


def audit_patients(
    cnn: dict[str, dict[str, Any]],
    vit: dict[str, dict[str, Any]],
    cv_records: dict[str, list[ImageRecord]],
    teachers: dict[str, dict[str, Any]],
    high_confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if set(cnn) != set(vit) or set(cnn) != set(cv_records):
        raise ValueError(
            "Patient sets differ: "
            f"cnn={len(cnn)}, vit={len(vit)}, cv={len(cv_records)}, "
            f"cnn-vs-vit={len(set(cnn) ^ set(vit))}, cnn-vs-cv={len(set(cnn) ^ set(cv_records))}"
        )
    results: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    for patient_id in sorted(cnn):
        cnn_row = cnn[patient_id]
        vit_row = vit[patient_id]
        records = cv_records[patient_id]
        targets = {record.target for record in records}
        folds = {record.fold for record in records}
        if len(targets) != 1 or len(folds) != 1:
            raise ValueError(f"Inconsistent CV records for patient {patient_id}")
        target = next(iter(targets))
        fold = next(iter(folds))
        if (cnn_row["target"], vit_row["target"]) != (target, target):
            raise ValueError(f"OOF/CV target mismatch for patient {patient_id}")
        if (cnn_row["fold"], vit_row["fold"]) != (fold, fold):
            raise ValueError(f"OOF/CV fold mismatch for patient {patient_id}")

        by_modality: dict[str, list[ImageRecord]] = defaultdict(list)
        for record in records:
            by_modality[record.modality].append(record)
        teacher = teachers.get(patient_id)
        teacher_grades = sorted(teacher["grades"]) if teacher else []
        original_grade = teacher_grades[0] if len(teacher_grades) == 1 else None
        teacher_merged = min(original_grade, 2) if original_grade is not None else None
        teacher_names = teacher["names"] if teacher else set()
        cv_source_names = {record.source_name for record in records}
        unreadable, image_size_counts = inspect_image_files(records) if target == 1 else (0, "not_checked")
        category, direction = classify_error(target, cnn_row["pred"], vit_row["pred"])
        cnn_selected = validate_selected(cnn_row, records)
        vit_selected = validate_selected(vit_row, records)

        duplicate_counts: dict[str, int] = {}
        gap_counts: dict[str, int] = {}
        unknown_axial_counts: dict[str, int] = {}
        for modality in ("FL", "T1"):
            axial_values = [record.axial for record in by_modality.get(modality, [])]
            known_counts = Counter(value for value in axial_values if value is not None)
            duplicate_counts[modality] = sum(count - 1 for count in known_counts.values() if count > 1)
            gap_counts[modality] = axial_gap_count(axial_values)
            unknown_axial_counts[modality] = sum(value is None for value in axial_values)

        flags: list[str] = []
        if not by_modality.get("FL"):
            flags.append("missing_FL")
        if not by_modality.get("T1"):
            flags.append("missing_T1")
        if len(records) < 40 or len(records) > 52:
            flags.append("unusual_total_image_count")
        for modality in ("FL", "T1"):
            if len(by_modality.get(modality, [])) < 20 or len(by_modality.get(modality, [])) > 26:
                flags.append(f"unusual_{modality}_image_count")
            if duplicate_counts[modality]:
                flags.append(f"duplicate_{modality}_axial")
            if gap_counts[modality]:
                flags.append(f"gap_{modality}_axial")
            if unknown_axial_counts[modality]:
                flags.append(f"unknown_{modality}_axial")
        if teacher is None:
            flags.append("teacher_patient_missing")
        elif len(teacher_grades) != 1:
            flags.append("teacher_grade_inconsistent")
        elif teacher_merged != target:
            flags.append("teacher_target_mismatch")
        if teacher is not None and cv_source_names != teacher_names:
            flags.append("cv_teacher_file_set_mismatch")
        if unreadable:
            flags.append("unreadable_image")
        for model, selected in (("cnn", cnn_selected), ("vit", vit_selected)):
            if not selected["selected_length_consistent"] or selected["selected_count"] != 5:
                flags.append(f"{model}_selected_length_error")
            if selected["selected_missing_path_count"]:
                flags.append(f"{model}_selected_path_missing")
            if selected["selected_wrong_patient_count"]:
                flags.append(f"{model}_selected_wrong_patient")
            if selected["selected_metadata_mismatch_count"]:
                flags.append(f"{model}_selected_metadata_mismatch")

        row: dict[str, Any] = {
            "patient_id": patient_id,
            "cv_fold": fold,
            "merged_target": target,
            "original_grade": original_grade,
            "error_category": category,
            "error_direction": direction,
            "num_images": len(records),
            "fl_count": len(by_modality.get("FL", [])),
            "t1_count": len(by_modality.get("T1", [])),
            "fl_axial_min": min((record.axial for record in by_modality.get("FL", []) if record.axial is not None), default=None),
            "fl_axial_max": max((record.axial for record in by_modality.get("FL", []) if record.axial is not None), default=None),
            "t1_axial_min": min((record.axial for record in by_modality.get("T1", []) if record.axial is not None), default=None),
            "t1_axial_max": max((record.axial for record in by_modality.get("T1", []) if record.axial is not None), default=None),
            "fl_axial_gaps": gap_counts["FL"],
            "t1_axial_gaps": gap_counts["T1"],
            "fl_axial_duplicates": duplicate_counts["FL"],
            "t1_axial_duplicates": duplicate_counts["T1"],
            "teacher_grade_values": ";".join(map(str, teacher_grades)),
            "teacher_merged_target": teacher_merged,
            "teacher_file_count": len(teacher_names),
            "cv_teacher_missing_files": len(teacher_names - cv_source_names) if teacher else None,
            "cv_teacher_extra_files": len(cv_source_names - teacher_names) if teacher else None,
            "unreadable_images": unreadable,
            "image_size_counts": image_size_counts,
            "cnn_pred": cnn_row["pred"],
            "cnn_prob_grade0": cnn_row["probs"][0],
            "cnn_prob_grade1": cnn_row["probs"][1],
            "cnn_prob_grade2plus": cnn_row["probs"][2],
            "cnn_max_prob": float(np.max(cnn_row["probs"])),
            "cnn_margin": prediction_margin(cnn_row["probs"]),
            "cnn_entropy": normalized_entropy(cnn_row["probs"]),
            "cnn_high_confidence_wrong": bool(cnn_row["pred"] != target and np.max(cnn_row["probs"]) >= high_confidence),
            "vit_pred": vit_row["pred"],
            "vit_prob_grade0": vit_row["probs"][0],
            "vit_prob_grade1": vit_row["probs"][1],
            "vit_prob_grade2plus": vit_row["probs"][2],
            "vit_max_prob": float(np.max(vit_row["probs"])),
            "vit_margin": prediction_margin(vit_row["probs"]),
            "vit_entropy": normalized_entropy(vit_row["probs"]),
            "vit_high_confidence_wrong": bool(vit_row["pred"] != target and np.max(vit_row["probs"]) >= high_confidence),
            "technical_flag_count": len(set(flags)),
            "technical_flags": ";".join(sorted(set(flags))),
        }
        for model, selected in (("cnn", cnn_selected), ("vit", vit_selected)):
            for key, value in selected.items():
                row[f"{model}_{key}"] = value
        results.append(row)
        for flag in sorted(set(flags)):
            anomalies.append(
                {
                    "patient_id": patient_id,
                    "cv_fold": fold,
                    "merged_target": target,
                    "original_grade": original_grade,
                    "error_category": category,
                    "technical_flag": flag,
                }
            )
    return results, anomalies


def mean_or_nan(values: Iterable[Any]) -> float:
    numeric = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(np.mean(numeric)) if numeric else float("nan")


def summarize_grade1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grade1 = [row for row in rows if row["merged_target"] == 1]
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("all", "all", grade1)]
    for category in ERROR_CATEGORIES:
        groups.append(("error_category", category, [row for row in grade1 if row["error_category"] == category]))
    for direction in sorted({row["error_direction"] for row in grade1}):
        groups.append(("error_direction", direction, [row for row in grade1 if row["error_direction"] == direction]))

    output: list[dict[str, Any]] = []
    for group_type, group, subset in groups:
        if not subset:
            continue
        output.append(
            {
                "group_type": group_type,
                "group": group,
                "num_patients": len(subset),
                "fraction_of_grade1": len(subset) / len(grade1),
                "technical_flag_patients": sum(bool(row["technical_flag_count"]) for row in subset),
                "cnn_high_confidence_wrong": sum(bool(row["cnn_high_confidence_wrong"]) for row in subset),
                "vit_high_confidence_wrong": sum(bool(row["vit_high_confidence_wrong"]) for row in subset),
                "mean_num_images": mean_or_nan(row["num_images"] for row in subset),
                "mean_cnn_prob_grade1": mean_or_nan(row["cnn_prob_grade1"] for row in subset),
                "mean_vit_prob_grade1": mean_or_nan(row["vit_prob_grade1"] for row in subset),
                "mean_cnn_margin": mean_or_nan(row["cnn_margin"] for row in subset),
                "mean_vit_margin": mean_or_nan(row["vit_margin"] for row in subset),
                "cnn_selected_fl_fraction": mean_or_nan(row["cnn_selected_fl_fraction"] for row in subset),
                "vit_selected_fl_fraction": mean_or_nan(row["vit_selected_fl_fraction"] for row in subset),
                "cnn_selected_axial_9_15_fraction": mean_or_nan(
                    row["cnn_selected_axial_9_15_fraction"] for row in subset
                ),
                "vit_selected_axial_9_15_fraction": mean_or_nan(
                    row["vit_selected_axial_9_15_fraction"] for row in subset
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_review_cases(rows: list[dict[str, Any]], total_cases: int, seed: int) -> list[dict[str, Any]]:
    if total_cases != 200:
        raise ValueError("The predeclared blinded review design currently requires --review-cases 200")
    rng = random.Random(seed)
    grade1 = [row for row in rows if row["merged_target"] == 1]
    grade1_both_wrong = [row for row in grade1 if row["error_category"] == "both_wrong"]
    dual_high_confidence = [
        row
        for row in grade1_both_wrong
        if row["cnn_high_confidence_wrong"] and row["vit_high_confidence_wrong"]
    ]
    if len(dual_high_confidence) > 60:
        raise ValueError(
            "More than 60 dual-high-confidence grade1 errors; revise the predeclared review design"
        )
    remaining_both_wrong = [row for row in grade1_both_wrong if row not in dual_high_confidence]
    selected_both_wrong = [
        *dual_high_confidence,
        *rng.sample(remaining_both_wrong, 60 - len(dual_high_confidence)),
    ]
    pools: list[tuple[str, list[dict[str, Any]], int]] = [
        ("grade1_both_wrong", selected_both_wrong, 60),
        ("grade1_vit_only", [row for row in grade1 if row["error_category"] == "vit_only_correct"], 20),
        ("grade1_cnn_only", [row for row in grade1 if row["error_category"] == "cnn_only_correct"], 20),
        ("grade1_both_correct", [row for row in grade1 if row["error_category"] == "both_correct"], 20),
        (
            "grade0_both_correct_control",
            [row for row in rows if row["merged_target"] == 0 and row["error_category"] == "both_correct"],
            20,
        ),
        (
            "grade0_error_control",
            [row for row in rows if row["merged_target"] == 0 and row["error_category"] != "both_correct"],
            20,
        ),
        (
            "grade2plus_both_correct_control",
            [row for row in rows if row["merged_target"] == 2 and row["error_category"] == "both_correct"],
            20,
        ),
        (
            "grade2plus_error_control",
            [row for row in rows if row["merged_target"] == 2 and row["error_category"] != "both_correct"],
            20,
        ),
    ]
    selected: list[dict[str, Any]] = []
    for stratum, pool, count in pools:
        if len(pool) < count:
            raise ValueError(f"Insufficient review pool for {stratum}: need={count}, available={len(pool)}")
        chosen = list(pool) if len(pool) == count else rng.sample(pool, count)
        for row in chosen:
            item = dict(row)
            item["review_sampling_stratum"] = stratum
            selected.append(item)
    rng.shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["case_id"] = f"AUD{index:03d}"
    return selected


def make_contact_sheet(case_id: str, records: list[ImageRecord], output_path: Path) -> None:
    tile_width, tile_height = 160, 176
    image_width, image_height = 150, 150
    columns = 8
    sections: list[tuple[str, list[ImageRecord]]] = []
    for modality in ("FL", "T1"):
        modality_records = sorted(
            [record for record in records if record.modality == modality],
            key=lambda record: (record.axial is None, record.axial or 10_000, record.path.name),
        )
        sections.append((modality, modality_records))
    rows_per_section = [max(1, math.ceil(len(modality_records) / columns)) for _, modality_records in sections]
    header_height = 42
    section_title_height = 28
    canvas_height = header_height + sum(
        section_title_height + rows * tile_height for rows in rows_per_section
    ) + 12
    canvas = Image.new("RGB", (columns * tile_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 12), f"CASE {case_id} - blinded full FL/T1 series", fill="black", font=font)
    y_offset = header_height
    for (modality, modality_records), row_count in zip(sections, rows_per_section):
        draw.rectangle((0, y_offset, columns * tile_width, y_offset + section_title_height), fill="#EAF2F8")
        draw.text((12, y_offset + 8), f"{modality} ({len(modality_records)} slices)", fill="black", font=font)
        y_offset += section_title_height
        for index, record in enumerate(modality_records):
            column = index % columns
            row = index // columns
            x = column * tile_width + 5
            y = y_offset + row * tile_height + 2
            try:
                with Image.open(record.path) as source:
                    image = source.convert("L")
                    image.thumbnail((image_width, image_height), Image.Resampling.LANCZOS)
                    rgb = Image.new("RGB", (image_width, image_height), "black")
                    paste_x = (image_width - image.width) // 2
                    paste_y = (image_height - image.height) // 2
                    rgb.paste(image.convert("RGB"), (paste_x, paste_y))
                    canvas.paste(rgb, (x, y))
            except Exception:
                draw.rectangle((x, y, x + image_width, y + image_height), fill="#F4CCCC")
                draw.text((x + 10, y + 60), "UNREADABLE", fill="#C00000", font=font)
            axial_label = "?" if record.axial is None else str(record.axial)
            draw.text((x + 3, y + image_height + 4), f"axial {axial_label}", fill="black", font=font)
        y_offset += row_count * tile_height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=90, optimize=True)


def create_review_package(
    output_dir: Path,
    review_rows: list[dict[str, Any]],
    cv_records: dict[str, list[ImageRecord]],
    generate_contact_sheets: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contact_dir = output_dir / "blinded_review" / "contact_sheets"
    blinded_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for row in review_rows:
        case_id = row["case_id"]
        contact_name = f"contact_sheets/{case_id}.jpg"
        if generate_contact_sheets:
            make_contact_sheet(case_id, cv_records[row["patient_id"]], output_dir / "blinded_review" / contact_name)
        blinded_rows.append(
            {
                "case_id": case_id,
                "contact_sheet": contact_name,
                "reviewer_id": "",
                "review_date": "",
                "reviewed_original_grade_0_to_4": "",
                "review_confidence_1_to_5": "",
                "borderline_grade0_grade1_YN": "",
                "borderline_grade1_grade2_YN": "",
                "image_quality_good_fair_poor": "",
                "motion_artifact_YN": "",
                "coverage_issue_YN": "",
                "FL_sufficient_YN": "",
                "T1_sufficient_YN": "",
                "lesion_visible_YN_uncertain": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "case_id": case_id,
                "patient_id": row["patient_id"],
                "review_sampling_stratum": row["review_sampling_stratum"],
                "cv_fold": row["cv_fold"],
                "original_grade": row["original_grade"],
                "merged_target": row["merged_target"],
                "error_category": row["error_category"],
                "error_direction": row["error_direction"],
                "cnn_pred": row["cnn_pred"],
                "cnn_prob_grade0": row["cnn_prob_grade0"],
                "cnn_prob_grade1": row["cnn_prob_grade1"],
                "cnn_prob_grade2plus": row["cnn_prob_grade2plus"],
                "vit_pred": row["vit_pred"],
                "vit_prob_grade0": row["vit_prob_grade0"],
                "vit_prob_grade1": row["vit_prob_grade1"],
                "vit_prob_grade2plus": row["vit_prob_grade2plus"],
                "technical_flags": row["technical_flags"],
            }
        )
    write_csv(output_dir / "blinded_review" / "blinded_review_form.csv", blinded_rows)
    for reviewer_id in ("Reviewer_A", "Reviewer_B"):
        reviewer_rows = []
        for row in blinded_rows:
            reviewer_row = dict(row)
            reviewer_row["reviewer_id"] = reviewer_id
            reviewer_rows.append(reviewer_row)
        write_csv(
            output_dir / "blinded_review" / f"blinded_review_form_{reviewer_id}.csv",
            reviewer_rows,
        )
    write_csv(output_dir / "blinded_review" / "PRIVATE_case_key_do_not_give_reviewer.csv", key_rows)
    (output_dir / "blinded_review" / "REVIEWER_INSTRUCTIONS.md").write_text(
        """# Blinded MRI grade review instructions

## Purpose
Independently review each patient's complete FL and T1 axial series. Do not consult the original teacher grade, CNN/ViT predictions, confidence values, or the private case key before completing the form.

## Procedure
1. Open the contact sheet named in `blinded_review_form.csv`.
2. Record the original 0–4 grade you judge most appropriate.
3. Record confidence from 1 (very uncertain) to 5 (very certain).
4. Mark whether the case is borderline between grades 0/1 or 1/2.
5. Record image quality, motion, coverage, modality sufficiency, and notes.
6. Do not change a grade to agree with a model. Model outputs are intentionally hidden.

## Recommended review design
- Two qualified reviewers should assess cases independently.
- Resolve disagreements only after both first-pass forms are locked.
- Reviewer A and Reviewer B should use their separate pre-generated CSV forms; keep both first-pass forms locked for inter-rater agreement analysis.
- Do not share contact sheets or patient-level files outside the approved research environment.

## Important
These contact sheets are for structured research review and may not preserve every diagnostic display property of the original images. If a case is uncertain, inspect the approved original image series in a clinical-grade viewer and record that additional review in `notes`.
""",
        encoding="utf-8",
    )
    (output_dir / "blinded_review" / "COORDINATOR_README_PRIVATE.md").write_text(
        """# Private coordinator notes

- Keep `PRIVATE_case_key_do_not_give_reviewer.csv` hidden until all independent reviews are locked.
- The 200 cases are model-error-enriched and include grade0/grade2+ controls. They are not a prevalence-representative sample.
- Do not recompute whole-cohort Accuracy by selectively replacing labels in this sample.
- Use the sample to identify error mechanisms and estimate category-specific disagreement.
- A revised primary ground truth requires a prospectively defined consensus process covering all patients or a representative random sample.
- This directory contains patient-derived medical images and identifiers. Never stage or push it to Git.
""",
        encoding="utf-8",
    )
    return blinded_rows, key_rows


def plot_grade1(rows: list[dict[str, Any]], output_dir: Path) -> None:
    grade1 = [row for row in rows if row["merged_target"] == 1]
    colors = {
        "both_correct": "#70AD47",
        "vit_only_correct": "#2E74B5",
        "cnn_only_correct": "#C55A11",
        "both_wrong": "#A5A5A5",
    }
    fig, axis = plt.subplots(figsize=(6.4, 5.6))
    for category in ERROR_CATEGORIES:
        subset = [row for row in grade1 if row["error_category"] == category]
        axis.scatter(
            [row["cnn_prob_grade1"] for row in subset],
            [row["vit_prob_grade1"] for row in subset],
            s=22,
            alpha=0.65,
            color=colors[category],
            label=f"{category} (n={len(subset)})",
        )
    axis.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    axis.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axis.plot([0, 1], [0, 1], color="#555555", linestyle=":", linewidth=1)
    axis.set_xlabel("ResNet18 probability for grade1")
    axis.set_ylabel("DeiT-small probability for grade1")
    axis.set_title("Grade1 OOF probability audit")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "grade1_probability_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    direction_counts = Counter(row["error_direction"] for row in grade1)
    labels = list(direction_counts)
    values = [direction_counts[label] for label in labels]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    bars = axis.barh(labels, values, color="#5B9BD5")
    axis.bar_label(bars)
    axis.set_xlabel("Patients")
    axis.set_title("Grade1 error directions")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "grade1_error_directions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    scan_errors: list[str],
    review_count: int,
    high_confidence: float,
) -> None:
    grade1 = [row for row in rows if row["merged_target"] == 1]
    category_counts = Counter(row["error_category"] for row in grade1)
    direction_counts = Counter(row["error_direction"] for row in grade1)
    anomaly_counts = Counter(row["technical_flag"] for row in anomalies if row["merged_target"] == 1)
    all_summary = next(row for row in summaries if row["group_type"] == "all")
    both_wrong = [row for row in grade1 if row["error_category"] == "both_wrong"]
    high_cnn = sum(bool(row["cnn_high_confidence_wrong"]) for row in grade1)
    high_vit = sum(bool(row["vit_high_confidence_wrong"]) for row in grade1)
    both_high = sum(
        bool(row["cnn_high_confidence_wrong"] and row["vit_high_confidence_wrong"])
        for row in grade1
    )

    lines = [
        "# Grade1 OOF technical audit",
        "",
        "## Scope",
        "",
        f"- Grade1 patients audited: {len(grade1)}",
        "- Source: frozen all-axial patient-level 5-fold OOF predictions",
        "- Models: ResNet18 and DeiT-small, best-loss checkpoint, patient top-5",
        "- This is a technical audit only; no medical re-grading was performed.",
        "- Patient IDs and derived contact sheets are local-only and must never be pushed to Git.",
        "",
        "## Paired error categories",
        "",
        "| Category | Patients | Fraction |",
        "|---|---:|---:|",
    ]
    for category in ERROR_CATEGORIES:
        count = category_counts.get(category, 0)
        lines.append(f"| {category} | {count} | {count / len(grade1):.1%} |")
    lines.extend(
        [
            "",
            "## Error directions",
            "",
            "| Direction | Patients |",
            "|---|---:|",
        ]
    )
    for direction, count in direction_counts.most_common():
        lines.append(f"| {direction} | {count} |")
    lines.extend(
        [
            "",
            "## High-confidence wrong predictions",
            "",
            f"High confidence was predeclared as final patient max probability ≥ {high_confidence:.2f}.",
            "",
            f"- CNN high-confidence wrong grade1 patients: {high_cnn}",
            f"- ViT high-confidence wrong grade1 patients: {high_vit}",
            f"- Both models high-confidence wrong on the same grade1 patient: {both_high}",
            "",
            "High-confidence errors are priority cases for blinded medical review because they may indicate a label discrepancy, a systematic visual pattern, or overconfident model failure.",
            "",
            "## Data and selection integrity",
            "",
            f"- Global validation scan errors: {len(scan_errors)}",
            f"- Grade1 patients with at least one technical flag: {all_summary['technical_flag_patients']} / {len(grade1)}",
            f"- Mean images per grade1 patient: {fmt(all_summary['mean_num_images'])}",
            f"- CNN top-5 FL fraction: {fmt(all_summary['cnn_selected_fl_fraction'])}",
            f"- ViT top-5 FL fraction: {fmt(all_summary['vit_selected_fl_fraction'])}",
            f"- CNN top-5 axial 9–15 fraction: {fmt(all_summary['cnn_selected_axial_9_15_fraction'])}",
            f"- ViT top-5 axial 9–15 fraction: {fmt(all_summary['vit_selected_axial_9_15_fraction'])}",
            "",
            "### Grade1 technical flags",
            "",
            "| Flag | Patient occurrences |",
            "|---|---:|",
        ]
    )
    if anomaly_counts:
        for flag, count in anomaly_counts.most_common():
            lines.append(f"| {flag} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Blinded clinical review package",
            "",
            f"A fixed-seed, model-error-enriched review set of {review_count} cases was created.",
            "It includes grade1 error categories, all grade1 cases misclassified with ≥0.80 confidence by both models, and grade0/grade2+ controls. Reviewers receive case IDs and complete FL/T1 contact sheets but not original labels, predictions, confidences, or sampling strata.",
            "",
            "The review should answer:",
            "",
            "1. How often are original grade1 cases independently re-graded as grade0 or grade2–4?",
            "2. What is inter-rater agreement (raw agreement and weighted kappa)?",
            "3. Are both-wrong cases enriched for borderline grade0/1, poor quality, incomplete coverage, or subtle lesions?",
            "4. Are high-confidence model errors plausible label discrepancies or true overconfident failures?",
            "5. Do top-5 failures reflect missed informative slices, requiring patient-level MIL/attention aggregation?",
            "",
            "## Interpretation rule",
            "",
            "- A high rate of blinded reviewer disagreement supports label ambiguity and ordinal/soft-label modelling.",
            "- Clear original-label errors support a prospectively defined consensus relabelling process followed by retraining.",
            "- Adequate labels but missed informative slices support a learned patient-level aggregator.",
            "- Adequate labels and appropriate selected slices support improving the encoder/loss rather than changing labels.",
            "",
            "## Important limitation",
            "",
            f"The {review_count}-case package is deliberately enriched by model error category and is not prevalence representative. It cannot be used to replace labels selectively and then report a revised whole-cohort Accuracy. Medical review remains pending until qualified reviewers complete the blinded forms.",
            "",
        ]
    )
    (output_dir / "grade1_technical_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_privacy_files(output_dir: Path) -> None:
    (output_dir / ".gitignore").write_text(
        "# Local patient-level audit package. Never stage or push.\n*\n!.gitignore\n",
        encoding="utf-8",
    )
    (output_dir / "DO_NOT_PUSH_MEDICAL_AUDIT.txt").write_text(
        "This directory contains patient identifiers and derived medical image review materials. "
        "Do not stage, commit, upload, email, or share outside the approved research environment.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_privacy_files(args.output_dir)

    cnn = load_oof(args.cnn_csv)
    vit = load_oof(args.vit_csv)
    cv_records, scan_errors = scan_cv_validation(args.cv_root)
    teachers = load_teacher_data(args.teacher_csvs)
    rows, anomalies = audit_patients(
        cnn=cnn,
        vit=vit,
        cv_records=cv_records,
        teachers=teachers,
        high_confidence=args.high_confidence,
    )
    if len(rows) != 1154:
        raise ValueError(f"Expected 1,154 audited OOF patients, got {len(rows)}")
    grade1_rows = [row for row in rows if row["merged_target"] == 1]
    if len(grade1_rows) != 444:
        raise ValueError(f"Expected 444 grade1 patients, got {len(grade1_rows)}")

    error_reference = {
        normalize_patient_id(row["patient_id"]): row for row in load_csv_rows(args.error_csv)
    }
    if set(error_reference) != set(cnn):
        raise ValueError("Detailed error CSV patient set does not match OOF patient set")
    for row in rows:
        reference = error_reference[row["patient_id"]]
        if row["error_category"] != reference["error_category"]:
            raise ValueError(f"Error-category mismatch for patient {row['patient_id']}")
        if int(reference["original_grade"]) != row["original_grade"]:
            raise ValueError(f"Original-grade mismatch for patient {row['patient_id']}")

    summaries = summarize_grade1(rows)
    write_csv(args.output_dir / "grade1_patient_technical_audit.csv", grade1_rows)
    write_csv(args.output_dir / "grade1_group_summary.csv", summaries)
    write_csv(args.output_dir / "technical_anomalies.csv", anomalies)
    if scan_errors:
        write_csv(args.output_dir / "validation_scan_errors.csv", [{"error": value} for value in scan_errors])
    plot_grade1(rows, args.output_dir)

    review_rows = select_review_cases(rows, total_cases=args.review_cases, seed=args.seed)
    create_review_package(
        output_dir=args.output_dir,
        review_rows=review_rows,
        cv_records=cv_records,
        generate_contact_sheets=not args.skip_contact_sheets,
    )
    write_report(
        output_dir=args.output_dir,
        rows=rows,
        summaries=summaries,
        anomalies=anomalies,
        scan_errors=scan_errors,
        review_count=len(review_rows),
        high_confidence=args.high_confidence,
    )

    result = {
        "num_oof_patients": len(rows),
        "num_grade1_patients": len(grade1_rows),
        "grade1_error_categories": dict(Counter(row["error_category"] for row in grade1_rows)),
        "grade1_error_directions": dict(Counter(row["error_direction"] for row in grade1_rows)),
        "grade1_patients_with_technical_flags": sum(bool(row["technical_flag_count"]) for row in grade1_rows),
        "review_cases": len(review_rows),
        "review_seed": args.seed,
        "contact_sheets_generated": not args.skip_contact_sheets,
        "medical_regrading_completed": False,
    }
    (args.output_dir / "audit_run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved grade1 audit to: {args.output_dir}")


if __name__ == "__main__":
    main()
