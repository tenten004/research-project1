# Frozen protocol: all-axial patient-level 5-fold CV

Frozen on 2026-07-14 before inspecting any cross-validation fold results.

## Data and split

- Source: `data/repro_fl_t1_all_axial_patient_split_3class`
- Modalities: FL + T1
- Slice scope: all axial slices; no manual axial-range restriction
- Classes: grade0 / grade1 / grade2+ (original grades 2, 3 and 4 merged)
- Unit of splitting: patient
- Splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=1)`
- Patients: 1,154 (grade0=557, grade1=444, grade2+=153)
- Images: 53,194
- Leakage requirement: zero train/validation patient overlap in every fold
- Coverage requirement: each patient appears in validation exactly once

## Frozen training conditions

Both models use the same fold, 224-pixel input, batch size 16, 30 epochs, AdamW,
learning rate 3e-5, weight decay 0.1, cosine scheduling, label smoothing 0.1,
and the same augmentation settings. The ViT-specific dropout and stochastic
depth are fixed at 0.1.

Models:

- DeiT-small patch16 224
- ResNet18

## Frozen evaluation

- Checkpoint: best validation loss
- Evaluation unit: patient
- Pooling: top-5 confidence
- Primary descriptive metric: macro-F1
- Secondary metrics: macro ROC-AUC, balanced accuracy, accuracy and per-class recall
- Final comparison: out-of-fold predictions from all 1,154 patients
- Uncertainty: paired patient bootstrap, 10,000 resamples, seed 20260714

No axial range, pooling value, checkpoint rule, optimizer, augmentation or model
hyperparameter will be selected again from the five fold results. Any later model
change constitutes a new exploratory experiment and must not be mixed with this
frozen comparison.

## Limitation

The protocol was frozen after exploratory work on the earlier validation cohort.
Five-fold out-of-fold evaluation reduces dependence on a single split, but it
does not create a truly external untouched cohort. External validation remains
necessary for the strongest generalization claim.
