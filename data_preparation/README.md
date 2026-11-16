# Data preparation scripts

All 18 scripts are the original files, unchanged. They use paths relative to the current working directory (`data/raw/...`, `outputs/...`), so run them from a folder that contains `data/` and `outputs/`. Docstrings mention `python src\...`; use `python data_preparation/scripts/<file>` instead.

| Stage | Script | Purpose |
|---|---|---|
| EDA | `eda.py` | Dataset statistics |
| 1 | `step1_overlap.py` | Overlap / near-duplicate checks |
| 2 | `step2_adjacency.py` | Adjacent-frame analysis |
| 3 | `step3_sequences.py` | Sequence detection |
| 4a | `step4a_buffer.py` | Buffer analysis |
| 4b | `step4b_weather.py` | Weather labels |
| 5 | `step5_group_construction.py` | Recording-session groups |
| 6 | `step6_feasibility.py`, `step6_plan_analysis.py`, `step6b_improved.py` | Split feasibility and assignment plan |
| 7 | `step7_build_split.py` | Apply the approved split |
| 8 | `step8a_inspect_annotations.py`, `step8_convert.py`, `step8g_visualize.py` | Inspect, convert VOC to YOLO, visualise |
| 9 | `step9_build_config.py`, `step9a_ultralytics_validation.py` | `data.yaml` and Ultralytics validation |
| 10 | `step10_hardware_check.py` | Hardware check |
| 12 | `step12_zip_dataset_for_kaggle.py` | Zip dataset for Kaggle (contains hard-coded `D:\FOD_project\...` default paths; adjust before use) |
