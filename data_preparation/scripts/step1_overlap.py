"""
Step 1: investigate overlapping ids between split files. READ-ONLY.

Run from D:\\FOD_project:
    python src\\step1_overlap.py --root "data\\raw\\FODPascalVOCFormat-V.2.1"
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main(root, eda_dir, out):
    root, eda_dir, out = Path(root), Path(eda_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines_out = []

    def say(msg=""):
        print(msg)
        lines_out.append(str(msg))

    # ---- 1. all images (stem -> path)
    images = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            images[p.stem] = p
    say(f"Total images on disk: {len(images)}")

    # ---- 2. every .txt under ImageSets (full path, not just stem)
    txt_files = sorted(p for p in root.rglob("*.txt") if "ImageSets" in p.parts)
    say(f"\n.txt files under ImageSets: {len(txt_files)}")
    for p in txt_files:
        say(f"  {p.relative_to(root)}")

    raw = {}  # name -> list of raw lines
    for p in txt_files:
        raw[p.stem] = [l.rstrip("\n") for l in p.read_text(errors="replace").splitlines() if l.strip()]

    # ---- 3. line format of each file
    say("\n== Raw line format (first 3 lines of each file) ==")
    for name, lines in raw.items():
        say(f"{name}: {len(lines)} lines | sample: {lines[:3]}")
    if "category_information" in raw:
        say("\ncategory_information (full content):")
        for l in raw["category_information"]:
            say(f"  {l}")

    need = {"test", "trainval"}
    if not need.issubset(raw):
        say(f"\nCould not find both 'test' and 'trainval' files. Found: {list(raw)}")
        (out / "step1_report.txt").write_text("\n".join(lines_out), encoding="utf-8")
        return

    def ids(name):
        return [l.split()[0] for l in raw[name]]

    test_ids, tv_ids = ids("test"), ids("trainval")
    test_set, tv_set = set(test_ids), set(tv_ids)

    # ---- 4. consistency checks
    say("\n== Consistency ==")
    say(f"test: {len(test_ids)} lines, {len(test_set)} unique")
    say(f"trainval: {len(tv_ids)} lines, {len(tv_set)} unique")
    dup_t = [k for k, v in Counter(test_ids).items() if v > 1]
    dup_tv = [k for k, v in Counter(tv_ids).items() if v > 1]
    say(f"ids repeated inside test: {len(dup_t)} | inside trainval: {len(dup_tv)}")

    overlap = sorted(test_set & tv_set)
    union = test_set | tv_set
    on_disk = set(images)
    in_neither = sorted(on_disk - union)
    listed_no_image = sorted(union - on_disk)
    say(f"\nOVERLAP (in both test and trainval): {len(overlap)}")
    say(f"Images on disk but in NEITHER list: {len(in_neither)}")
    say(f"Listed ids with NO image on disk: {len(listed_no_image)}")

    # ---- 5. show raw lines for overlapping ids
    say("\n== Overlapping ids: raw line numbers ==")
    for i in overlap:
        in_t = [n + 1 for n, l in enumerate(raw["test"]) if l.split()[0] == i]
        in_v = [n + 1 for n, l in enumerate(raw["trainval"]) if l.split()[0] == i]
        say(f"  id={i!r} | test.txt line {in_t} -> {[raw['test'][n-1] for n in in_t]} "
            f"| trainval.txt line {in_v} -> {[raw['trainval'][n-1] for n in in_v]}")

    say("\n== Images in neither list ==")
    for i in in_neither:
        say(f"  {i}")
    if listed_no_image:
        say("\n== Listed ids without an image ==")
        for i in listed_no_image:
            say(f"  {i}")

    # ---- 6. annotation info for the suspicious ids
    ann_path = eda_dir / "annotations.csv"
    rows = []
    if ann_path.exists():
        ann = pd.read_csv(ann_path, dtype={"stem": str})
        sus = sorted(set(overlap) | set(in_neither))
        sub = ann[ann.stem.isin(sus)][["stem", "cls", "xmin", "ymin", "xmax", "ymax"]]
        say("\n== Annotations of suspicious ids (overlap + in-neither) ==")
        say(sub.to_string(index=False) if len(sub) else "(none found in annotations.csv)")
        sub.to_csv(out / "suspicious_annotations.csv", index=False)
    else:
        say(f"\n(annotations.csv not found at {ann_path}; skipping annotation info)")

    for tag, lst in (("overlap", overlap), ("in_neither", in_neither)):
        for i in lst:
            rows.append(dict(kind=tag, id=i, image=str(images.get(i, ""))))
    pd.DataFrame(rows, columns=["kind", "id", "image"]).to_csv(out / "suspicious_ids.csv", index=False)

    # ---- 7. do suspicious ids sit next to each other numerically? (off-by-one hint)
    def num(s):
        return int(s) if s.isdigit() else None

    ov_n = sorted(n for n in map(num, overlap) if n is not None)
    ne_n = sorted(n for n in map(num, in_neither) if n is not None)
    say(f"\nNumeric overlap ids   : {ov_n}")
    say(f"Numeric in-neither ids: {ne_n}")

    (out / "step1_report.txt").write_text("\n".join(lines_out), encoding="utf-8")
    print(f"\nDone. Report saved in: {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--eda", default="outputs/eda")
    ap.add_argument("--out", default="outputs/step1")
    a = ap.parse_args()
    main(a.root, a.eda, a.out)