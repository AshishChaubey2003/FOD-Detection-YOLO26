#!/usr/bin/env python3
"""
Step 8B-8F: Pascal VOC -> YOLO class mapping, box conversion, split-preserving
label generation, and mandatory validation.

READS (never modifies):
  - Original VOC XML/images under --root (read-only; located recursively by
    filename stem, so an unknown sub-folder depth like VOC2007\\ is fine)
  - The approved Step 7 split manifests: <split-dir>/train.csv, val.csv, test.csv
    (must have an 'image_id' column; 'image_path' from these files is carried
    through verbatim into the output manifests, never invented)

WRITES (new files only, under --out, which must end in "step8_yolo"):
  classes.txt                   one class name per line, in ID order (0..N-1)
  class_mapping.csv             class_name,class_id
  labels/train/<image_id>.txt   labels/val/..., labels/test/...  (YOLO format)
  manifests/train_labels.csv    image_id,image_path,label_path,split
  manifests/val_labels.csv
  manifests/test_labels.csv
  conversion_summary.txt
  step8_report.txt

Does NOT re-split (uses the exact Step 7 train/val/test assignment as given),
does NOT divide any assignment unit, does NOT copy/move/alter original images
or XML, does NOT touch outputs/step6_final.

Coordinate convention: Step 8A confirmed (on the real dataset) a ZERO-BASED
convention (min xmin=0.0, min ymin=0.0, boxes exactly touch the image edge at
xmax==width / ymax==height). So the standard formula is applied with NO -1 or
other adjustment:
    x_center = ((xmin+xmax)/2) / width
    y_center = ((ymin+ymax)/2) / height
    width_n  = (xmax-xmin) / width
    height_n = (ymax-ymin) / height
"""
import argparse
import csv
import sys
import xml.etree.ElementTree as ET
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


def text(node, tag, default=None):
    e = node.find(tag)
    return e.text.strip() if (e is not None and e.text is not None) else default


def parse_xml(path):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        return {"error": str(e)}
    size = root.find("size")
    w = h = None
    if size is not None:
        wt, ht = text(size, "width"), text(size, "height")
        w = float(wt) if wt is not None else None
        h = float(ht) if ht is not None else None
    objs = []
    for obj in root.findall("object"):
        name = text(obj, "name")
        bb = obj.find("bndbox")
        box = None
        if bb is not None:
            try:
                box = (
                    float(text(bb, "xmin")),
                    float(text(bb, "ymin")),
                    float(text(bb, "xmax")),
                    float(text(bb, "ymax")),
                )
            except (TypeError, ValueError):
                box = None
        objs.append(dict(name=name, box=box))
    return dict(width=w, height=h, objects=objs)


def load_split_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "image_id" not in (r.fieldnames or []):
            say(f"[ABORT] {path} mein 'image_id' column nahi mili. Columns: {r.fieldnames}")
            sys.exit(1)
        for row in r:
            rows.append(row)
    return rows


def main(a):
    root = Path(a.root)
    split_dir = Path(a.split_dir)
    out = Path(a.out)
    if out.name != "step8_yolo":
        say(f"[ABORT] --out ka last folder naam 'step8_yolo' hona chahiye, mila: {out.name}")
        sys.exit(1)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "test").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)

    say("=" * 74)
    say("STEP 8B-8F: VOC -> YOLO CLASS MAPPING + CONVERSION + SPLIT LABELS + VALIDATION")
    say("=" * 74)

    # ---- 1. Load Step 7 approved split (read-only; NO re-split) ----
    splits = {}
    split_rows = {}
    for name in ("train", "val", "test"):
        p = split_dir / f"{name}.csv"
        if not p.exists():
            say(f"[ABORT] {p} nahi mili. Step 7 approved split chahiye, naya split nahi banaya ja sakta.")
            sys.exit(1)
        rows = load_split_csv(p)
        split_rows[name] = rows
        for row in rows:
            iid = row["image_id"]
            if iid in splits:
                say(f"[ABORT] image_id {iid} {splits[iid]} aur {name} dono mein hai -- split files corrupt lagte hain.")
                sys.exit(1)
            splits[iid] = name
    counts = Counter(splits.values())
    say(f"Loaded split (read-only, Step 7 se): train={counts['train']} val={counts['val']} "
        f"test={counts['test']} total={len(splits)}")

    # ---- 2. Locate all XML/image files recursively ----
    xmls = find_files(root, {".xml"})
    imgs = find_files(root, {".jpg", ".jpeg", ".png"})
    say(f"XML files found: {len(xmls)} | Image files found: {len(imgs)}")

    missing_xml = [iid for iid in splits if iid not in xmls]
    if missing_xml:
        say(f"[ABORT] {len(missing_xml)} split image_id ke liye XML nahi mili. Sample: {missing_xml[:5]}")
        sys.exit(1)
    missing_img = [iid for iid in splits if iid not in imgs]
    if missing_img:
        say(f"[ABORT] {len(missing_img)} split image_id ke liye image nahi mili. Sample: {missing_img[:5]}")
        sys.exit(1)
    extra_xml = [s for s in xmls if s not in splits]
    if extra_xml:
        say(f"[WARNING] {len(extra_xml)} XML files mile jo split manifest mein nahi hain "
            f"(sample: {extra_xml[:5]}). Yeh skip ki jayengi.")

    # ---- 3. Parse all split XMLs once, build class list ----
    say("\nParsing XML (sirf split-listed images)...")
    parsed = {}
    class_names_found = set()
    n = 0
    for iid in splits:
        d = parse_xml(xmls[iid])
        if "error" in d:
            say(f"[ABORT] {xmls[iid]} parse nahi hui: {d['error']}")
            sys.exit(1)
        if d["width"] is None or d["height"] is None:
            say(f"[ABORT] {xmls[iid]} mein <size> missing hai.")
            sys.exit(1)
        parsed[iid] = d
        for o in d["objects"]:
            if o["name"] is None:
                say(f"[ABORT] {xmls[iid]} mein ek object ka <name> missing hai -- class invent nahi karunga.")
                sys.exit(1)
            class_names_found.add(o["name"])
        n += 1
        if n % 5000 == 0:
            say(f"  parsed {n}/{len(splits)}")
    say(f"Parsed OK: {n}/{len(splits)}")

    class_list = sorted(class_names_found)
    say(f"\nUnique classes found across split images: {len(class_list)}")
    if a.expect_classes and len(class_list) != a.expect_classes:
        say(f"[WARNING] Expected {a.expect_classes} classes (Step 8A se), mila {len(class_list)}. "
            f"Aage badh rahe hain par isse review karein.")
    class_to_id = {c: i for i, c in enumerate(class_list)}

    # ---- 8B: write class mapping ----
    with open(out / "classes.txt", "w", encoding="utf-8") as f:
        for c in class_list:
            f.write(c + "\n")
    with open(out / "class_mapping.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name", "class_id"])
        for c in class_list:
            w.writerow([c, class_to_id[c]])
    say(f"classes.txt aur class_mapping.csv likhi gayi ({len(class_list)} classes, "
        f"alphabetical order mein ID 0..{len(class_list)-1}).")

    # ---- 8C + 8D: convert + write per-split label files ----
    say("\nConverting VOC -> YOLO aur per-split label files likh rahe hain...")
    n_images = 0
    n_objects_total = 0
    n_empty_images = 0
    out_of_range = []
    nonpositive_wh = []
    per_split_obj_count = Counter()
    per_split_img_count = Counter()
    manifest_rows = defaultdict(list)
    source_obj_count_by_split = Counter()

    image_path_lookup = {}
    for name in ("train", "val", "test"):
        for row in split_rows[name]:
            image_path_lookup[row["image_id"]] = row.get("image_path", "")

    for iid, split in splits.items():
        d = parsed[iid]
        w, h = d["width"], d["height"]
        lines = []
        for o in d["objects"]:
            source_obj_count_by_split[split] += 1
            box = o["box"]
            if box is None:
                say(f"[ABORT] {iid}: <bndbox> missing/unparseable -- conversion rok rahe hain.")
                sys.exit(1)
            xmin, ymin, xmax, ymax = box
            xc = ((xmin + xmax) / 2.0) / w
            yc = ((ymin + ymax) / 2.0) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            clamped = any(v < 0.0 or v > 1.0 for v in (xc, yc, bw, bh))
            if clamped:
                out_of_range.append((iid, round(xc, 4), round(yc, 4), round(bw, 4), round(bh, 4)))
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                bw = min(max(bw, 0.0), 1.0)
                bh = min(max(bh, 0.0), 1.0)
            if bw <= 0.0 or bh <= 0.0:
                nonpositive_wh.append(iid)
            cid = class_to_id[o["name"]]
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            n_objects_total += 1
            per_split_obj_count[split] += 1

        if not lines:
            n_empty_images += 1

        label_path = out / "labels" / split / f"{iid}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        manifest_rows[split].append((iid, image_path_lookup.get(iid, ""), str(label_path)))
        per_split_img_count[split] += 1
        n_images += 1
        if n_images % 5000 == 0:
            say(f"  converted {n_images}/{len(splits)}")

    say(f"Converted images: {n_images} | Total objects written: {n_objects_total} | "
        f"Empty-label images: {n_empty_images}")
    if out_of_range:
        say(f"[WARNING] {len(out_of_range)} boxes clamp karni padi (0-1 range se bahar thi) -- "
            f"yeh Step 8A ke verified convention ke hisaab se expected nahi tha. "
            f"Sample: {out_of_range[:5]}")
    if nonpositive_wh:
        say(f"[WARNING] {len(nonpositive_wh)} boxes ki width/height <=0 hai. Sample: {nonpositive_wh[:5]}")

    # ---- manifests ----
    for split in ("train", "val", "test"):
        p = out / "manifests" / f"{split}_labels.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "image_path", "label_path", "split"])
            for iid, ip, lp in manifest_rows[split]:
                w.writerow([iid, ip, lp, split])
        say(f"manifest likhi: {p} ({len(manifest_rows[split])} rows)")

    # ================= 8F: MANDATORY VALIDATION =================
    say("\n" + "=" * 74)
    say("STEP 8F: MANDATORY VALIDATION CHECKS")
    say("=" * 74)
    results = []

    def chk(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        say(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))

    # 1. every split-listed image has exactly one label file at the right path
    missing_label = []
    for split in ("train", "val", "test"):
        for row in split_rows[split]:
            iid = row["image_id"]
            if not (out / "labels" / split / f"{iid}.txt").exists():
                missing_label.append((split, iid))
    chk("1. Har split image ke liye exactly ek label file maujood hai",
        len(missing_label) == 0, f"{len(missing_label)} missing, sample {missing_label[:5]}")

    # 2. no image appears in more than one split folder
    dupe = []
    seen = set()
    for split in ("train", "val", "test"):
        for row in split_rows[split]:
            iid = row["image_id"]
            if iid in seen:
                dupe.append(iid)
            seen.add(iid)
    chk("2. Koi image ek se zyada split folder mein nahi hai",
        len(dupe) == 0, f"{len(dupe)} dupes, sample {dupe[:5]}")

    # 3. train/val/test image counts unchanged from the approved Step 7 split
    chk("3. Train/Val/Test image counts Step 7 jaisi hi hain",
        per_split_img_count["train"] == counts["train"]
        and per_split_img_count["val"] == counts["val"]
        and per_split_img_count["test"] == counts["test"],
        f"got train={per_split_img_count['train']} val={per_split_img_count['val']} "
        f"test={per_split_img_count['test']}; expected train={counts['train']} "
        f"val={counts['val']} test={counts['test']}")

    # 4. all class IDs unique and exactly 0..N-1
    ids = sorted(class_to_id.values())
    chk("4. Class IDs unique aur 0..N-1 range mein hain",
        ids == list(range(len(ids))), f"ids sample={ids[:5]}")

    # 5. no class silently dropped -- every object's class name has an id
    unmapped = set()
    for iid, d in parsed.items():
        for o in d["objects"]:
            if o["name"] not in class_to_id:
                unmapped.add(o["name"])
    chk("5. Koi class silently drop nahi hui (har object ko ID mili)",
        len(unmapped) == 0, f"unmapped: {unmapped}")

    # 6. all normalized coordinates within [0,1] (post-clamp)
    chk("6. Saari YOLO coordinates [0,1] range mein hain",
        len(out_of_range) == 0,
        f"{len(out_of_range)} boxes range se bahar thi (clamp ki gayi) -- review karein")

    # 7. positive width/height
    chk("7. Saari boxes ki width aur height > 0 hai",
        len(nonpositive_wh) == 0, f"{len(nonpositive_wh)} boxes, sample {nonpositive_wh[:5]}")

    # 8. no malformed/non-numeric label lines (re-read the written files)
    malformed = []
    for split in ("train", "val", "test"):
        for row in split_rows[split]:
            iid = row["image_id"]
            p = out / "labels" / split / f"{iid}.txt"
            with open(p, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    parts = ln.split()
                    if len(parts) != 5:
                        malformed.append((iid, ln))
                        continue
                    try:
                        int(parts[0])
                        [float(x) for x in parts[1:]]
                    except ValueError:
                        malformed.append((iid, ln))
    chk("8. Koi label file mein malformed/non-numeric line nahi hai",
        len(malformed) == 0, f"{len(malformed)}, sample {malformed[:3]}")

    # 9. converted object count == source XML object count, per split (no exclusions)
    mism = {s: (per_split_obj_count[s], source_obj_count_by_split[s])
            for s in ("train", "val", "test") if per_split_obj_count[s] != source_obj_count_by_split[s]}
    chk("9. Converted object count = source XML object count (per split, koi exclusion nahi)",
        len(mism) == 0, f"{mism}")

    # 10. no assignment_unit divided across splits (re-checked from Step 7's own column, if present)
    unit_issue = []
    has_unit_col = all(("assignment_unit" in split_rows[s][0]) for s in ("train", "val", "test") if split_rows[s])
    if has_unit_col:
        unit_to_splits = defaultdict(set)
        for s in ("train", "val", "test"):
            for row in split_rows[s]:
                unit_to_splits[row["assignment_unit"]].add(s)
        unit_issue = [u for u, ss in unit_to_splits.items() if len(ss) > 1]
        chk("10. Koi assignment_unit do splits mein divide nahi hua (Step 7 manifest ke hisaab se)",
            len(unit_issue) == 0, f"{unit_issue[:5]}")
    else:
        chk("10. Koi assignment_unit do splits mein divide nahi hua",
            False, "split CSV mein 'assignment_unit' column nahi mili -- yeh check skip nahi ki ja sakti")

    all_pass = all(r[1] for r in results)
    say("\n" + "=" * 74)
    say(f"OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED -- USE NA KAREIN, review karein'}")
    say("=" * 74)

    # ---- class-wise object count summary ----
    class_obj_count = Counter()
    for d in parsed.values():
        for o in d["objects"]:
            class_obj_count[o["name"]] += 1

    with open(out / "conversion_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Total images converted: {n_images}\n")
        f.write(f"Total objects written: {n_objects_total}\n")
        f.write(f"Empty-label images: {n_empty_images}\n")
        f.write(f"Classes: {len(class_list)}\n\n")
        for split in ("train", "val", "test"):
            f.write(f"{split}: images={per_split_img_count[split]} objects={per_split_obj_count[split]}\n")
        f.write("\nObjects per class:\n")
        for c in class_list:
            f.write(f"  {c}: {class_obj_count[c]}\n")
        f.write("\nValidation:\n")
        for name, ok, detail in results:
            f.write(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else "") + "\n")

    with open(out / "step8_report.txt", "w", encoding="utf-8") as f:
        f.write("STEP 8B-8F REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"1. Class mapping: {len(class_list)} classes, alphabetical order, IDs 0..{len(class_list)-1}.\n")
        f.write("   Files: classes.txt, class_mapping.csv\n")
        f.write("2. Conversion formula: zero-based convention (confirmed in Step 8A) -- "
                "standard midpoint/width formula, no coordinate adjustment.\n")
        f.write(f"3. Labels generated: {n_images} files, {n_objects_total} objects, "
                f"{n_empty_images} empty-label images.\n")
        f.write(f"4. Split preserved exactly from Step 7: train={per_split_img_count['train']} "
                f"val={per_split_img_count['val']} test={per_split_img_count['test']}.\n")
        f.write(f"5. Validation checks: {'ALL PASS' if all_pass else 'SOME FAILED -- see conversion_summary.txt'}\n")
        f.write("6. Visual verification: run step8g_visualize.py separately.\n")
        f.write("7. Limitations: see WARNING lines above (if any) in the terminal output "
                "and conversion_summary.txt.\n")

    say(f"\nconversion_summary.txt aur step8_report.txt likhi gayi: {out}")
    say("\nAb Step 8G (visual verification montage) ke liye step8g_visualize.py chalayein.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/FODPascalVOCFormat-V.2.1")
    ap.add_argument("--split-dir", default="outputs/step6_final")
    ap.add_argument("--out", default="outputs/step8_yolo")
    ap.add_argument("--expect-classes", type=int, default=31)
    main(ap.parse_args())