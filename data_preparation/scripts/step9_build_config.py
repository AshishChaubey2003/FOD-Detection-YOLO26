#!/usr/bin/env python3
"""
Step 9: YOLO dataset configuration (data.yaml) + compatibility verification.

READS (never modifies):
  - Raw images under --raw-root (read-only; located recursively by stem)
  - Step 7 split files: <split-dir>/train.csv, val.csv, test.csv, split_manifest.csv
  - Step 8 outputs: <yolo-dir>/classes.txt, class_mapping.csv, labels/{train,val,test}/*.txt
    (the class mapping is REUSED exactly as Step 8 made it -- never recomputed)

WHY A NEW "dataset/" TREE IS BUILT (the compatibility problem):
  Ultralytics finds an image's label by substituting the path segment
  "{sep}images{sep}" with "{sep}labels{sep}" in the image's path and swapping
  the extension for ".txt" (this is the exact algorithm Ultralytics uses
  internally -- img2label_paths in ultralytics/data/utils.py). Our raw images
  live in a flat "JPEGImages" folder (no "images" path segment, no train/val/
  test split) and our labels live in a *separate* tree
  (outputs/step8_yolo/labels/{train,val,test}). Neither folder name nor the
  split structure lines up, so Ultralytics' own substitution would NOT find
  our labels if we pointed it straight at JPEGImages.

  Smallest safe fix that touches neither the raw dataset nor the Step 8
  labels: build a small mirrored "dataset/images/{train,val,test}" and
  "dataset/labels/{train,val,test}" tree under outputs/step9_config, where
  each file is a HARD LINK (same inode, zero extra disk space, not a copy,
  original untouched) to the real raw image / Step 8 label file. This gives
  Ultralytics' own path convention something that actually resolves, while
  "do not copy or move the raw dataset" and "do not modify ... Step 8 labels"
  both still hold -- a hard link does not duplicate or alter the source file.
  (Symlink is tried as a fallback if hard-linking fails, e.g. across volumes;
  if neither works for a file, that file is reported as a FAILURE, never
  silently copied or skipped.)

WRITES (new files only, under --out):
  dataset/images/{train,val,test}/<image_id>.<ext>   (hard link / symlink)
  dataset/labels/{train,val,test}/<image_id>.txt      (hard link / symlink)
  dataset/path_manifest.csv   image_id,split,source_image_path,linked_image_path,label_path,link_type
  data.yaml
  config_validation_report.txt
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


def find_files(root: Path, exts):
    out = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            out[p.stem] = p
    return out


def load_csv(path, required_cols=()):
    if not path.exists():
        say(f"[ABORT] {path} nahi mili.")
        sys.exit(1)
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        for c in required_cols:
            if c not in cols:
                say(f"[ABORT] {path} mein '{c}' column nahi mili. Columns: {cols}")
                sys.exit(1)
        return list(r), cols


def link_file(src: Path, dst: Path):
    """Try hard link, then symlink. Never copies. Returns link type or None on failure."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:
        return None


def img2label_path(img_path: str):
    """Replicates Ultralytics' own img2label_paths substitution exactly, so we
    can verify -- using their actual convention -- that our tree resolves."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa not in img_path:
        return None
    base = sb.join(img_path.rsplit(sa, 1))
    return str(Path(base).with_suffix(".txt"))


def main(a):
    raw_root = Path(a.raw_root)
    split_dir = Path(a.split_dir)
    yolo_dir = Path(a.yolo_dir)
    out = Path(a.out)
    dataset = out / "dataset"
    report_lines = []

    def rsay(s=""):
        say(s)
        report_lines.append(s)

    rsay("=" * 74)
    rsay("STEP 9: YOLO DATASET CONFIGURATION + COMPATIBILITY VERIFICATION")
    rsay("=" * 74)

    # ================= SECTION 3: INSPECT EXISTING STRUCTURE =================
    rsay("\n--- 3. Inspecting existing structure (read-only) ---")

    train_rows, train_cols = load_csv(split_dir / "train.csv", ["image_id", "image_path"])
    val_rows, val_cols = load_csv(split_dir / "val.csv", ["image_id", "image_path"])
    test_rows, test_cols = load_csv(split_dir / "test.csv", ["image_id", "image_path"])
    rsay(f"train.csv: {len(train_rows)} rows, columns={train_cols}")
    rsay(f"val.csv:   {len(val_rows)} rows, columns={val_cols}")
    rsay(f"test.csv:  {len(test_rows)} rows, columns={test_cols}")

    manifest_rows, manifest_cols = load_csv(split_dir / "split_manifest.csv", ["image_id", "assignment_unit"])
    rsay(f"split_manifest.csv: {len(manifest_rows)} rows, columns={manifest_cols}")

    classes_path = yolo_dir / "classes.txt"
    mapping_path = yolo_dir / "class_mapping.csv"
    if not classes_path.exists() or not mapping_path.exists():
        rsay(f"[ABORT] {classes_path} ya {mapping_path} nahi mili.")
        sys.exit(1)
    class_list = classes_path.read_text(encoding="utf-8").splitlines()
    mapping_rows, _ = load_csv(mapping_path, ["class_name", "class_id"])
    mapping_dict = {r["class_name"]: int(r["class_id"]) for r in mapping_rows}
    rsay(f"classes.txt: {len(class_list)} classes | class_mapping.csv: {len(mapping_dict)} entries")

    mismatch = [c for i, c in enumerate(class_list) if mapping_dict.get(c) != i]
    if mismatch:
        rsay(f"[ABORT] classes.txt aur class_mapping.csv ka order/ID match nahi karta: {mismatch[:5]}")
        sys.exit(1)
    rsay("classes.txt aur class_mapping.csv ek doosre se consistent hain (same order, same IDs). "
         "Class mapping Step 8 se verbatim REUSE ki ja rahi hai, dobara banayi nahi ja rahi.")

    split_by_id = {}
    rows_by_split = {"train": train_rows, "val": val_rows, "test": test_rows}
    for split, rows in rows_by_split.items():
        for row in rows:
            iid = row["image_id"]
            if iid in split_by_id:
                rsay(f"[ABORT] {iid} {split_by_id[iid]} aur {split} dono mein hai.")
                sys.exit(1)
            split_by_id[iid] = split
    counts = Counter(split_by_id.values())
    rsay(f"Verified split counts (train.csv/val.csv/test.csv se): "
         f"train={counts['train']} val={counts['val']} test={counts['test']} total={len(split_by_id)}")

    # sample label existence check for a handful from each split, per Step 8 labels
    for split in ("train", "val", "test"):
        sample = rows_by_split[split][:3]
        for row in sample:
            lp = yolo_dir / "labels" / split / f"{row['image_id']}.txt"
            if not lp.exists():
                rsay(f"[ABORT] Sample check fail: {lp} nahi mili (Step 8 output check karein).")
                sys.exit(1)
    rsay("Sample image_id -> Step 8 label-file mapping (stem match) verified.")

    raw_imgs = find_files(raw_root, {".jpg", ".jpeg", ".png"})
    rsay(f"Raw image files found (recursive, read-only scan): {len(raw_imgs)}")
    missing_raw = [iid for iid in split_by_id if iid not in raw_imgs]
    if missing_raw:
        rsay(f"[ABORT] {len(missing_raw)} split image_id ke liye raw image nahi mili. Sample: {missing_raw[:5]}")
        sys.exit(1)

    rsay("\n--- Compatibility problem (verified, not assumed) ---")
    sample_raw_path = str(next(iter(raw_imgs.values())))
    rsay(f"Sample raw image path: {sample_raw_path}")
    if f"{os.sep}images{os.sep}" in sample_raw_path.lower():
        rsay("Raw path mein literal 'images' path-segment hai -- Ultralytics shayad direct resolve kar le, "
             "phir bhi hum mirrored tree bana rahe hain taaki train/val/test split bhi correctly resolve ho.")
    else:
        rsay("Raw image path mein Ultralytics ka expected literal '" + os.sep + "images" + os.sep +
             "' path-segment NAHI hai (folder 'JPEGImages' hai, 'images' nahi), aur raw images train/val/test "
             "ke hisaab se alag folders mein bhi nahi hain. Ultralytics apne hi convention "
             "(img2label_paths: 'images'->'labels' substitution) se humari Step 8 labels is structure mein "
             "DHOOND NAHI payega. Isliye current structure Ultralytics ke saath directly INCOMPATIBLE hai.")

    # ================= BUILD MIRRORED images/labels TREE (hard link, no copy) =================
    rsay("\n--- 4. Building Ultralytics-compatible mirrored tree (hard links, NOT copies) ---")
    dataset.mkdir(parents=True, exist_ok=True)
    link_failures = []
    path_manifest_rows = []
    for split in ("train", "val", "test"):
        for row in rows_by_split[split]:
            iid = row["image_id"]
            src_img = raw_imgs[iid]
            dst_img = dataset / "images" / split / f"{iid}{src_img.suffix.lower()}"
            lt_img = link_file(src_img, dst_img)
            src_lbl = yolo_dir / "labels" / split / f"{iid}.txt"
            dst_lbl = dataset / "labels" / split / f"{iid}.txt"
            lt_lbl = link_file(src_lbl, dst_lbl)
            if lt_img is None:
                link_failures.append(("image", iid, str(src_img), str(dst_img)))
            if lt_lbl is None:
                link_failures.append(("label", iid, str(src_lbl), str(dst_lbl)))
            path_manifest_rows.append((iid, split, str(src_img), str(dst_img), str(dst_lbl), lt_img, lt_lbl))

    with open(dataset / "path_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "split", "source_image_path", "linked_image_path", "label_path",
                     "image_link_type", "label_link_type"])
        for r in path_manifest_rows:
            w.writerow(r)

    if link_failures:
        rsay(f"[ABORT] {len(link_failures)} file(s) link nahi ho payi (hardlink aur symlink dono fail) -- "
             f"copy NAHI ki gayi (strict rule). Sample: {link_failures[:5]}")
        rsay("Possible reason: images/outputs alag drives/volumes par hain, ya symlink ke liye privilege chahiye "
             "(Windows par Developer Mode on karein ya admin se chalayein). Review karein aur bataein.")
        sys.exit(1)
    link_types = Counter(lt for _, _, _, _, _, lt_img, lt_lbl in [(0,0,0,0,0,r[5],r[6]) for r in path_manifest_rows] for lt in (lt_img, lt_lbl))
    rsay(f"Mirrored tree ban gayi: {dataset/'images'} + {dataset/'labels'}  "
         f"(link types used: {dict(link_types)}; koi file copy nahi hui -- same inode/pointer hai).")

    # ================= SECTION 4: data.yaml =================
    rsay("\n--- Writing data.yaml ---")
    abs_dataset = dataset.resolve()
    yaml_lines = []
    yaml_lines.append("# Auto-generated by step9_build_config.py -- Step 9 YOLO dataset configuration")
    yaml_lines.append("# Split: Step 7 (outputs/step6_final, unchanged). Labels: Step 8 (outputs/step8_yolo, unchanged).")
    yaml_lines.append("# images/labels below are hard-link mirrors, not copies -- see dataset/path_manifest.csv")
    yaml_lines.append(f"path: {abs_dataset.as_posix()}")
    yaml_lines.append("train: images/train")
    yaml_lines.append("val: images/val")
    yaml_lines.append("test: images/test")
    yaml_lines.append("")
    yaml_lines.append(f"nc: {len(class_list)}")
    yaml_lines.append("names:")
    for i, c in enumerate(class_list):
        yaml_lines.append(f"  {i}: {c}")
    data_yaml_path = out / "data.yaml"
    data_yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    rsay(f"data.yaml likhi gayi: {data_yaml_path}")

    # ================= SECTION 5: PATH COMPATIBILITY VALIDATION =================
    rsay("\n" + "=" * 74)
    rsay("SECTION 5: PATH COMPATIBILITY VALIDATION")
    rsay("=" * 74)
    results = []

    def chk(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        rsay(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))

    # 1-3: every split image resolves to an existing image (in the new tree)
    missing_img_by_split = defaultdict(list)
    for split in ("train", "val", "test"):
        for row in rows_by_split[split]:
            iid = row["image_id"]
            p = dataset / "images" / split / f"{iid}{raw_imgs[iid].suffix.lower()}"
            if not p.exists():
                missing_img_by_split[split].append(iid)
    for split in ("train", "val", "test"):
        chk(f"Every {split} image resolves to an existing linked image",
            len(missing_img_by_split[split]) == 0,
            f"{len(missing_img_by_split[split])} missing, sample {missing_img_by_split[split][:5]}")

    # 4: every image resolves to its correct YOLO label VIA ULTRALYTICS' OWN SUBSTITUTION RULE
    sub_fail = []
    for split in ("train", "val", "test"):
        for row in rows_by_split[split]:
            iid = row["image_id"]
            img_path = str(dataset / "images" / split / f"{iid}{raw_imgs[iid].suffix.lower()}")
            derived = img2label_path(img_path)
            expected = str(dataset / "labels" / split / f"{iid}.txt")
            if derived is None or Path(derived) != Path(expected) or not Path(derived).exists():
                sub_fail.append((iid, img_path, derived, expected))
    chk("Every image resolves to its correct label via Ultralytics' own img2label_paths substitution",
        len(sub_fail) == 0, f"{len(sub_fail)} mismatches, sample {sub_fail[:3]}")

    # 5: no image appears in multiple splits
    dup = [iid for iid, cnt in Counter(split_by_id.keys()).items() if cnt > 1]
    chk("No image appears in multiple splits", len(dup) == 0, f"{dup[:5]}")

    # 6: no assignment unit divided between splits (from split_manifest.csv, authoritative)
    unit_to_splits = defaultdict(set)
    for row in manifest_rows:
        unit_to_splits[row["assignment_unit"]].add(row["split"])
    unit_issue = [u for u, ss in unit_to_splits.items() if len(ss) > 1]
    chk("No assignment unit is divided between splits", len(unit_issue) == 0, f"{unit_issue[:5]}")

    # 7: all label paths resolve correctly (already covered by check 4, restated for the required list)
    chk("All label paths resolve correctly (dataset/labels tree)", len(sub_fail) == 0)

    # 8: all 31 class names match the mapping
    xml_classes_seen = set(class_list)
    chk("All class names match class_mapping.csv (Step 8, reused unchanged)",
        set(mapping_dict.keys()) == xml_classes_seen,
        f"diff={set(mapping_dict.keys()) ^ xml_classes_seen}")

    # 9: class IDs between 0 and 30 (0..N-1)
    ids = sorted(mapping_dict.values())
    chk(f"Class IDs remain between 0 and {len(class_list)-1}",
        ids == list(range(len(class_list))), f"ids sample={ids[:5]}")

    # 10: image counts remain exactly as Step 7
    chk("Image counts remain exactly 23688 / 3384 / 6721 (or dataset's own verified Step 7 counts)",
        True if (a.expect_train is None or (counts["train"] == a.expect_train and counts["val"] == a.expect_val and counts["test"] == a.expect_test)) else False,
        f"got train={counts['train']} val={counts['val']} test={counts['test']}; "
        f"expected train={a.expect_train} val={a.expect_val} test={a.expect_test}")

    all_pass = all(r[1] for r in results)
    rsay("\n" + "=" * 74)
    rsay(f"OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED -- USE NA KAREIN, review karein'}")
    rsay("=" * 74)

    # ================= SECTION 6: ULTRALYTICS COMPATIBILITY CHECK =================
    rsay("\n--- 6. Ultralytics compatibility check ---")
    ultra_status = "not checked"
    try:
        import ultralytics  # noqa
        ultra_status = f"installed, version={ultralytics.__version__}"
        rsay(f"ultralytics {ultra_status}")
        try:
            from ultralytics.utils import yaml_load
            loaded = yaml_load(str(data_yaml_path))
            ok = loaded.get("nc") == len(class_list) and len(loaded.get("names", {})) == len(class_list)
            rsay(f"[{'PASS' if ok else 'FAIL'}] Ultralytics apne hi yaml_load() se data.yaml ko bina error "
                 f"ke parse kar paaya, aur nc/names count match karta hai.")
            ultra_status += "; yaml_load OK" if ok else "; yaml_load MISMATCH"
        except Exception as e:
            rsay(f"[WARNING] Ultralytics yaml_load() se check nahi ho paya: {e}")
            ultra_status += f"; yaml_load check failed: {e}"
    except ImportError:
        rsay("ultralytics package installed NAHI hai is Python environment mein. "
             "Agar aap chahen to 'pip install ultralytics' chala kar is check ko repeat kar sakte hain -- "
             "yeh optional hai, kyunki hum ne upar Ultralytics ka APNA hi path-substitution algorithm "
             "(img2label_paths) replicate karke directory-layout compatibility already verify kar li hai, "
             "bina library install kiye.")
        ultra_status = "not installed (directory-layout compatibility verified independently above)"

    # ---- write validation report ----
    with open(out / "config_validation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        f.write("\n\nValidation checks:\n")
        for name, ok, detail in results:
            f.write(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else "") + "\n")
        f.write(f"\nUltralytics status: {ultra_status}\n")

    say(f"\nconfig_validation_report.txt likhi gayi: {out/'config_validation_report.txt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="data/raw/FODPascalVOCFormat-V.2.1")
    ap.add_argument("--split-dir", default="outputs/step6_final")
    ap.add_argument("--yolo-dir", default="outputs/step8_yolo")
    ap.add_argument("--out", default="outputs/step9_config")
    ap.add_argument("--expect-train", type=int, default=23688)
    ap.add_argument("--expect-val", type=int, default=3384)
    ap.add_argument("--expect-test", type=int, default=6721)
    main(ap.parse_args())