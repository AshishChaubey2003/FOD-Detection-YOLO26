# Evaluation artefacts

- `metrics/` metric summaries (CSV/JSON) from the corrected custom evaluation.
- `confusion_matrix/` and `curves/`: produced by the **official Ultralytics test evaluation**. They reflect that run, not the custom conf 0.25 / IoU 0.50 matching.
- `error_analysis/` class-wise errors, report and improvement roadmap (custom evaluation).
- Figures from the custom evaluation are in `../reports/figures/`.
- Ultralytics `val_batch*.jpg` preview images are intentionally not included.
