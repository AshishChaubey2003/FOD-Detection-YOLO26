#!/usr/bin/env python3
"""
Step 9A: FINAL verification using the ACTUAL installed Ultralytics library.

This script does NOT trust the Step 9 report's numbers -- it independently
re-derives everything from the real source files on disk right now:
  - outputs/step6_final/{train,val,test}.csv  (Step 7, read-only)
  - outputs/step9_config/data.yaml            (Step 9, read-only)
  - outputs/step9_config/dataset/{images,labels}/{train,val,test}  (read-only)
  - outputs/step9_config/dataset/path_manifest.csv  (to know each link's source,
    for the hard-link verification)

It writes ONLY new files:
  outputs/step9_config/step9a_validation_report.txt

It never: modifies Step 7/8/9 outputs, modifies raw images/XML, installs
packages, downloads weights, or starts training/inference. If Ultralytics is
not installed, it reports that plainly and stops -- it does not pip install
on its own, per the explicit "ask for permission before installing" rule.
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter


def say(s=""):
    print(s)
    sys.stdout.flush()


def main(a):
    data_yaml = Path(a.data_yaml)
    dataset_dir = Path(a.dataset_dir)
    classes_path = Path(a.classes)
    split_dir = Path(a.split_dir)
    manifest_path = dataset_dir / "path_manifest.csv"
    out = Path(a.out)
    report_path = out / "step9a_validation_report.txt"
    if report_path.exists():
        say(f"[ABORT] {report_path} already exists -- purani report overwrite nahi karunga. "
            f"Rename/move karke dobara chalayein.")
        sys.exit(1)

    report = []

    def rsay(s=""):
        say(s)
        report.append(s)

    rsay("=" * 74)
    rsay("STEP 9A: FINAL ULTRALYTICS CONFIGURATION VERIFICATION")
    rsay("=" * 74)

    # ================= SECTION 4: CHECK ULTRALYTICS INSTALLATION =================
    rsay("\n--- 4. Ultralytics installation check ---")
    try:
        import ultralytics
        ultra_version = ultralytics.__version__
        rsay(f"ultralytics INSTALLED, version = {ultra_version}")
        ultra_available = True
    except ImportError:
        rsay("ultralytics installed NAHI hai is Python environment mein.")
        rsay("Maine KHUD install NAHI kiya -- yeh aapki permission se hona chahiye.")
        rsay("Agar aap chahen, terminal mein chalayein: pip install ultralytics")
        rsay("Uske baad yeh script dobara chalayein. Is run mein Section 5 ke "
             "'actual Ultralytics library' checks SKIP kiye ja rahe hain -- "
             "inka result PASS claim NAHI kiya ja raha, kyunki library use nahi hui.")
        ultra_available = False
        ultra_version = None

    # ================= SECTION 3: INSPECT EXISTING CONFIGURATION (fresh, no trust) =================
    rsay("\n--- 3. Inspecting existing configuration (read-only, fresh re-derivation) ---")

    if not data_yaml.exists():
        rsay(f"[ABORT] {data_yaml} nahi mili.")
        sys.exit(1)
    yaml_text = data_yaml.read_text(encoding="utf-8")
    rsay(f"data.yaml ({data_yaml}) ka poora content:\n{'-'*40}\n{yaml_text}{'-'*40}")

    # parse with the REAL ultralytics loader if available, else a minimal manual parse
    parsed = None
    yaml_api_used = None
    if ultra_available:
        try:
            try:
                from ultralytics.utils import yaml_load as _yl
                parsed = _yl(str(data_yaml))
                yaml_api_used = "yaml_load()"
            except ImportError:
                from ultralytics.utils import YAML as _Y
                parsed = _Y.load(str(data_yaml))
                yaml_api_used = "YAML.load()"
            rsay(f"[PASS] data.yaml Ultralytics ke apne {yaml_api_used} se bina error parse hui.")
        except Exception as e:
            rsay(f"[FAIL] data.yaml Ultralytics ke {yaml_api_used or 'yaml loader'} se parse NAHI hui: {e}")
    if parsed is None:
        # minimal manual fallback parse (only used if ultralytics unavailable or its loader failed)
        import ast
        parsed = {}
        names = {}
        in_names = False
        for ln in yaml_text.splitlines():
            if ln.strip().startswith("#") or not ln.strip():
                continue
            if ln.startswith("names:"):
                in_names = True
                continue
            if in_names and ln.startswith("  "):
                k, v = ln.strip().split(":", 1)
                names[int(k)] = v.strip()
                continue
            in_names = False
            if ":" in ln:
                k, v = ln.split(":", 1)
                parsed[k.strip()] = v.strip()
        parsed["names"] = names
        if "nc" in parsed:
            parsed["nc"] = int(parsed["nc"])
        rsay("[INFO] data.yaml manually parse ki gayi (ultralytics loader available/successful nahi tha).")

    # ---- verify nc / names against classes.txt ----
    if not classes_path.exists():
        rsay(f"[ABORT] {classes_path} nahi mili.")
        sys.exit(1)
    class_list = classes_path.read_text(encoding="utf-8").splitlines()
    names = parsed.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): v for k, v in names.items()}

    nc_ok = parsed.get("nc") == len(class_list)
    rsay(f"[{'PASS' if nc_ok else 'FAIL'}] data.yaml 'nc' == {len(class_list)}  (got {parsed.get('nc')})")

    order_mismatch = [(i, class_list[i], names.get(i)) for i in range(len(class_list)) if names.get(i) != class_list[i]]
    rsay(f"[{'PASS' if not order_mismatch else 'FAIL'}] data.yaml 'names' classes.txt ke EXACT order se match "
         f"karte hain" + ("" if not order_mismatch else f" -- mismatches: {order_mismatch[:5]}"))

    # ---- use the REAL ultralytics check_class_names, if available ----
    class_names_check_ok = None
    if ultra_available:
        try:
            from ultralytics.data.utils import check_class_names
            checked = check_class_names(names)
            class_names_check_ok = (checked == names) or (len(checked) == len(names))
            rsay(f"[{'PASS' if class_names_check_ok else 'FAIL'}] Ultralytics ke apne check_class_names() se "
                 f"names dictionary valid hai.")
        except Exception as e:
            rsay(f"[WARNING] check_class_names() call nahi ho paya: {e}")

    # ---- resolve path/train/val/test exactly like data.yaml says ----
    base_path = Path(parsed.get("path", data_yaml.parent))
    split_paths = {}
    for split in ("train", "val", "test"):
        rel = parsed.get(split)
        if rel is None:
            rsay(f"[ABORT] data.yaml mein '{split}:' key missing hai.")
            sys.exit(1)
        p = Path(rel)
        split_paths[split] = p if p.is_absolute() else (base_path / p)
    rsay(f"Resolved paths -> train: {split_paths['train']} | val: {split_paths['val']} | test: {split_paths['test']}")

    # ---- inspect mirrored directory structure ----
    rsay("\nMirrored dataset directory structure:")
    disk_images = {}
    for split in ("train", "val", "test"):
        img_dir = split_paths[split]
        lbl_dir = dataset_dir / "labels" / split
        imgs = sorted(p for p in img_dir.iterdir()) if img_dir.exists() else []
        lbls = sorted(p for p in lbl_dir.iterdir()) if lbl_dir.exists() else []
        disk_images[split] = {p.stem: p for p in imgs}
        rsay(f"  {split}: images dir exists={img_dir.exists()} ({len(imgs)} files) | "
             f"labels dir exists={lbl_dir.exists()} ({len(lbls)} files)")

    # ================= SECTION 5: ACTUAL DATASET VALIDATION =================
    rsay("\n" + "=" * 74)
    rsay("SECTION 5: ACTUAL DATASET VALIDATION")
    rsay("=" * 74)
    results = []

    def chk(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        rsay(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))

    chk("data.yaml parses without error", parsed is not None)
    chk(f"nc == {len(class_list)} (actual class count, from classes.txt)", nc_ok, f"got {parsed.get('nc')}")
    chk("names match classes.txt in exact order", not order_mismatch, f"{order_mismatch[:5]}")
    if class_names_check_ok is not None:
        chk("Ultralytics check_class_names() accepts the names dict", class_names_check_ok)

    # fresh, independent re-derivation of expected split membership from Step 7 CSVs
    rows_by_split = {}
    expected_ids = {}
    for split in ("train", "val", "test"):
        p = split_dir / f"{split}.csv"
        if not p.exists():
            rsay(f"[ABORT] {p} nahi mili -- Step 7 ka source-of-truth chahiye.")
            sys.exit(1)
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows_by_split[split] = rows
        expected_ids[split] = {r["image_id"] for r in rows}

    for split in ("train", "val", "test"):
        missing_on_disk = expected_ids[split] - set(disk_images[split].keys())
        extra_on_disk = set(disk_images[split].keys()) - expected_ids[split]
        chk(f"{split}: image paths resolve (Step 7 list == mirrored dataset dir)",
            not missing_on_disk and not extra_on_disk,
            f"missing={len(missing_on_disk)} sample={list(missing_on_disk)[:5]} | "
            f"extra={len(extra_on_disk)} sample={list(extra_on_disk)[:5]}")

    # labels resolvable using Ultralytics' OWN path-resolution logic (not a re-implementation)
    img2label_used = None
    label_resolve_fail = []
    if ultra_available:
        try:
            from ultralytics.data.utils import img2label_paths
            img2label_used = "ultralytics.data.utils.img2label_paths (REAL library function)"
            for split in ("train", "val", "test"):
                img_paths = [str(p) for p in disk_images[split].values()]
                derived_labels = img2label_paths(img_paths)
                for ip, lp in zip(img_paths, derived_labels):
                    if not Path(lp).exists():
                        label_resolve_fail.append((split, ip, lp))
        except Exception as e:
            rsay(f"[WARNING] img2label_paths() call nahi ho paya: {e}")
            img2label_used = None
    if img2label_used:
        chk(f"Corresponding labels resolve via {img2label_used}",
            not label_resolve_fail, f"{len(label_resolve_fail)} failures, sample {label_resolve_fail[:5]}")
    else:
        chk("Corresponding labels resolve via Ultralytics' actual path-resolution logic",
            False, "ultralytics not installed or img2label_paths import failed -- "
                   "yeh check ACTUAL library ke bina skip/FAIL mark ki gayi")

    # no duplicate images across splits (fresh recheck)
    all_ids = []
    for split in ("train", "val", "test"):
        all_ids += list(expected_ids[split])
    dup = [iid for iid, c in Counter(all_ids).items() if c > 1]
    chk("No duplicate images across splits", not dup, f"{dup[:5]}")

    # split counts unchanged
    chk("Split counts: train/val/test match --expect-* values",
        len(expected_ids["train"]) == a.expect_train
        and len(expected_ids["val"]) == a.expect_val
        and len(expected_ids["test"]) == a.expect_test,
        f"got train={len(expected_ids['train'])} val={len(expected_ids['val'])} test={len(expected_ids['test'])}; "
        f"expected train={a.expect_train} val={a.expect_val} test={a.expect_test}")

    # ================= SECTION 6: HARD LINK VERIFICATION =================
    rsay("\n" + "=" * 74)
    rsay("SECTION 6: HARD LINK VERIFICATION")
    rsay("=" * 74)

    if not manifest_path.exists():
        rsay(f"[ABORT] {manifest_path} nahi mili -- link-source jaankari ke bina verify nahi kar sakta.")
        sys.exit(1)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    by_split = defaultdict(list)
    for r in manifest_rows:
        by_split[r["split"]].append(r)

    link_checks = []
    for split in ("train", "val", "test"):
        sample = by_split[split][: a.sample_per_split]
        for r in sample:
            iid = r["image_id"]
            src_img = Path(r["source_image_path"])
            link_img = Path(r["linked_image_path"])
            label_path = Path(r["label_path"])
            entry = dict(image_id=iid, split=split)

            entry["image_exists"] = link_img.exists()
            entry["image_readable"] = False
            if entry["image_exists"]:
                try:
                    with open(link_img, "rb") as fh:
                        fh.read(16)
                    entry["image_readable"] = True
                except OSError:
                    pass

            entry["label_exists"] = label_path.exists()
            entry["label_readable"] = False
            if entry["label_exists"]:
                try:
                    label_path.read_text(encoding="utf-8")
                    entry["label_readable"] = True
                except OSError:
                    pass

            entry["source_exists"] = src_img.exists()
            same_file = False
            if entry["image_exists"] and entry["source_exists"]:
                try:
                    st_link = os.stat(link_img)
                    st_src = os.stat(src_img)
                    same_file = (st_link.st_dev == st_src.st_dev) and (st_link.st_ino == st_src.st_ino)
                except OSError:
                    same_file = False
            entry["same_inode_as_source"] = same_file
            link_checks.append(entry)

    bad = [e for e in link_checks
           if not (e["image_exists"] and e["image_readable"] and e["label_exists"]
                   and e["label_readable"] and e["source_exists"] and e["same_inode_as_source"])]
    rsay(f"Checked {len(link_checks)} representative files across train/val/test.")
    for e in link_checks:
        ok = (e["image_exists"] and e["image_readable"] and e["label_exists"]
              and e["label_readable"] and e["source_exists"] and e["same_inode_as_source"])
        rsay(f"  [{'OK' if ok else 'ISSUE'}] {e['split']}/{e['image_id']}: "
             f"image_exists={e['image_exists']} readable={e['image_readable']} "
             f"label_exists={e['label_exists']} readable={e['label_readable']} "
             f"source_exists={e['source_exists']} same_inode_as_source={e['same_inode_as_source']}")
    chk("Hard-link sample: images+labels exist, readable, and share the source's inode "
        "(confirms a true link, not an accidental copy)", not bad, f"{len(bad)} issues: {bad[:3]}")

    rsay("\n[Known Windows/NTFS limitations -- general info, not a per-run result]")
    rsay("- Hard links (CreateHardLink) only work WITHIN the same NTFS volume/drive; they fail across "
         "different drive letters. If D:\\FOD_project's raw data and outputs ever end up on different "
         "drives, linking would fail outright (the Step 9 script aborts rather than silently copying).")
    rsay("- ReFS and exFAT do not support hard links at all; FAT32 doesn't either.")
    rsay("- If this project folder is inside a cloud-sync directory (OneDrive, Dropbox, Google Drive), the "
         "sync client can sometimes treat hard-linked copies as separate files, which could duplicate data "
         "or desync the link relationship. Keeping the project outside synced folders avoids this.")

    all_pass = all(r[1] for r in results)
    rsay("\n" + "=" * 74)
    rsay(f"OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED -- USE NA KAREIN, review karein'}")
    if not ultra_available:
        rsay("NOTE: ultralytics installed nahi tha, isliye 'actual Ultralytics library' validation (Section 5 "
             "ka library-dependent hissa) genuinely PASS claim nahi ki ja sakti -- upar jo FAIL/SKIP dikh "
             "raha hai, wahi sahi reflect karta hai.")
    rsay("=" * 74)

    out.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    say(f"\n{report_path} likhi gayi.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", default="outputs/step9_config/data.yaml")
    ap.add_argument("--dataset-dir", default="outputs/step9_config/dataset")
    ap.add_argument("--classes", default="outputs/step8_yolo/classes.txt")
    ap.add_argument("--split-dir", default="outputs/step6_final")
    ap.add_argument("--out", default="outputs/step9_config")
    ap.add_argument("--sample-per-split", type=int, default=5)
    ap.add_argument("--expect-train", type=int, default=23688)
    ap.add_argument("--expect-val", type=int, default=3384)
    ap.add_argument("--expect-test", type=int, default=6721)
    main(ap.parse_args())