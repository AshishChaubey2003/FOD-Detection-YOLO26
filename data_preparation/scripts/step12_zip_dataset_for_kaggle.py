"""
step12_zip_dataset_for_kaggle.py

Rebuilds fod_yolo_dataset.zip with forward-slash ("/") path separators
inside the archive, which is what Kaggle (and zip/Linux tooling in
general) requires. PowerShell's Compress-Archive stored Windows-style
backslash ("\\") separators in the entry names, which Kaggle's upload
validator rejects ("contains a forbidden character in name").

This script is READ-ONLY on the source dataset folder (outputs\\step9_config
\\dataset) -- it only (re)writes the zip file. The old zip (if present) is
removed and rebuilt from scratch so there is no ambiguity about leftover
bad entries.
"""
import argparse
import os
import zipfile


def main(a):
    src_dir = a.src
    zip_path = a.out

    if not os.path.isdir(src_dir):
        raise SystemExit(f"[ABORT] source directory not found: {src_dir}")

    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"Removed old zip: {zip_path}")

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(src_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, src_dir)
                arcname = rel_path.replace(os.sep, "/")  # force forward slashes
                zf.write(full_path, arcname)
                file_count += 1

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Wrote {file_count} files into {zip_path} ({size_mb:.1f} MB)")

    # Self-check: confirm no backslashes survived in any stored entry name
    bad = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if "\\" in info.filename:
                bad.append(info.filename)

    if bad:
        print(f"[WARNING] {len(bad)} entries still contain backslashes, e.g.: {bad[:3]}")
    else:
        print("[OK] All entries use forward-slash separators -- Kaggle-compatible.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=r"D:\FOD_project\outputs\step9_config\dataset")
    p.add_argument("--out", default=r"D:\FOD_project\outputs\step9_config\fod_yolo_dataset.zip")
    main(p.parse_args())