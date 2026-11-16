#!/usr/bin/env python3
"""
Step 8G: Visual verification montage.

Selects representative images across all 3 splits (spanning different
classes, object counts, and -- where a multi-object image exists in a split
-- at least one multi-object example), draws:
    - GREEN boxes = original VOC XML annotation
    - RED boxes   = the converted YOLO label, denormalized back to pixels
side by side, and saves one montage image. This is a SAMPLE spot-check only
-- it does NOT prove every annotation is correct; that confidence comes from
the Step 8F numeric validation, which runs over every image and object.

Reads only (writes just the montage file + a text report):
  - original images/XML under --root (read-only, located recursively)
  - classes.txt and labels/<split>/*.txt written by step8_convert.py
Does not touch the original dataset, Step 7 outputs, or the label files.
"""
import argparse
import csv
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw


def say(s=""):
    print(s)
    sys.stdout.flush()


def find_files(root: Path, exts):
    out = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            out[p.stem] = p
    return out


def parse_xml_boxes(path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bb = obj.find("bndbox")
        box = (
            float(bb.find("xmin").text),
            float(bb.find("ymin").text),
            float(bb.find("xmax").text),
            float(bb.find("ymax").text),
        )
        boxes.append((name, box))
    return w, h, boxes


def read_yolo_label(path, w, h, id_to_class):
    boxes = []
    txt = Path(path).read_text(encoding="utf-8").strip()
    if not txt:
        return boxes
    for ln in txt.splitlines():
        parts = ln.split()
        cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        xmin = (xc - bw / 2) * w
        xmax = (xc + bw / 2) * w
        ymin = (yc - bh / 2) * h
        ymax = (yc + bh / 2) * h
        boxes.append((id_to_class[cid], (xmin, ymin, xmax, ymax)))
    return boxes


def main(a):
    root = Path(a.root)
    out = Path(a.out)
    classes_path = out / "classes.txt"
    if not classes_path.exists():
        say(f"[ABORT] {classes_path} nahi mili. Pehle step8_convert.py chalayein.")
        sys.exit(1)
    classes = classes_path.read_text(encoding="utf-8").splitlines()
    id_to_class = {i: c for i, c in enumerate(classes)}

    say("=" * 74)
    say("STEP 8G: VISUAL VERIFICATION MONTAGE")
    say("=" * 74)

    imgs = find_files(root, {".jpg", ".jpeg", ".png"})
    xmls = find_files(root, {".xml"})

    per_split_by_key = defaultdict(lambda: defaultdict(list))  # split -> (cls, n_obj) -> [iid]
    for split in ("train", "val", "test"):
        mp = out / "manifests" / f"{split}_labels.csv"
        if not mp.exists():
            say(f"[ABORT] {mp} nahi mili. Pehle step8_convert.py chalayein.")
            sys.exit(1)
        with open(mp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                iid = row["image_id"]
                if iid not in xmls:
                    say(f"[ABORT] {iid} ki XML nahi mili.")
                    sys.exit(1)
                _, _, xml_boxes = parse_xml_boxes(xmls[iid])
                n_obj = len(xml_boxes)
                cls = xml_boxes[0][0] if xml_boxes else "EMPTY"
                per_split_by_key[split][(cls, n_obj)].append(iid)

    random.seed(a.seed)
    selected = []
    for split in ("train", "val", "test"):
        keys = list(per_split_by_key[split].keys())
        if not keys:
            continue
        multi = [k for k in keys if k[1] > 1]
        singles = [k for k in keys if k[1] <= 1]
        chosen_keys = []
        if multi:
            chosen_keys.append(multi[0])
        random.shuffle(singles)
        chosen_keys += singles[: max(0, a.per_split - len(chosen_keys))]
        for k in chosen_keys:
            iid = per_split_by_key[split][k][0]
            selected.append((split, iid))

    if not selected:
        say("[ABORT] Koi image select nahi hui -- manifests khali lag rahe hain.")
        sys.exit(1)

    say(f"Selected {len(selected)} images across splits for visual verification:")
    for split, iid in selected:
        say(f"  {split}: {iid}")

    cell_w, cell_h = 320, 320
    n_cols = 2  # original (green) | converted-redrawn (red)
    montage = Image.new("RGB", (cell_w * n_cols, cell_h * len(selected)), "white")

    mismatches = []
    for row_i, (split, iid) in enumerate(selected):
        img_path = imgs.get(iid)
        if img_path is None:
            say(f"[ABORT] {iid} ki image file nahi mili.")
            sys.exit(1)
        xml_w, xml_h, xml_boxes = parse_xml_boxes(xmls[iid])
        label_path = out / "labels" / split / f"{iid}.txt"
        if not label_path.exists():
            say(f"[ABORT] {label_path} nahi mili.")
            sys.exit(1)
        yolo_boxes = read_yolo_label(label_path, xml_w, xml_h, id_to_class)

        im_orig = Image.open(img_path).convert("RGB").resize((cell_w, cell_h))
        im_conv = im_orig.copy()
        sx, sy = cell_w / xml_w, cell_h / xml_h

        d1 = ImageDraw.Draw(im_orig)
        for name, (x0, y0, x1, y1) in xml_boxes:
            d1.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline="lime", width=2)
            d1.text((x0 * sx + 2, y0 * sy + 2), name, fill="lime")

        d2 = ImageDraw.Draw(im_conv)
        for name, (x0, y0, x1, y1) in yolo_boxes:
            d2.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline="red", width=2)
            d2.text((x0 * sx + 2, y0 * sy + 2), name, fill="red")

        if len(xml_boxes) != len(yolo_boxes):
            mismatches.append((iid, "object count differs"))
        else:
            for (n1, b1), (n2, b2) in zip(xml_boxes, yolo_boxes):
                if n1 != n2 or any(abs(c1 - c2) > 0.5 for c1, c2 in zip(b1, b2)):
                    mismatches.append((iid, f"{n1} vs {n2}, {b1} vs {b2}"))

        montage.paste(im_orig, (0, row_i * cell_h))
        montage.paste(im_conv, (cell_w, row_i * cell_h))

    montage_path = out / "step8g_verification_montage.jpg"
    montage.save(montage_path, quality=90)
    say(f"\nMontage saved: {montage_path}")
    say("  Left column  = original XML boxes (green)")
    say("  Right column = converted YOLO label re-drawn (red)")

    if mismatches:
        say(f"\n[WARNING] {len(mismatches)} mismatch(es) mile box re-drawing mein:")
        for m in mismatches[:10]:
            say(f"  {m}")
    else:
        say(f"\n{len(selected)} sampled images mein, original XML boxes aur converted-YOLO-se-redraw-ki-gayi "
            f"boxes exactly (rounding tolerance ke andar) match hui.")
    say("\nYeh sirf ek SAMPLE spot-check hai -- isse yeh CLAIM nahi kiya ja sakta ki saari annotations "
        "sahi hain. Poora confidence Step 8F ke numeric validation se aata hai, jo saari images/objects "
        "par chalti hai.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/FODPascalVOCFormat-V.2.1")
    ap.add_argument("--out", default="outputs/step8_yolo")
    ap.add_argument("--per-split", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())