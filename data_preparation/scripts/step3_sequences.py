"""
Step 3: build a sequence table (runs of consecutive frames) + per-class sequence counts.
READ-ONLY on the dataset.

Run from D:\\FOD_project:
    python src\\step3_sequences.py --root "data\\raw\\FODPascalVOCFormat-V.2.1"

A new sequence starts when:
  (a) ids are not consecutive, or
  (b) consecutive images share no class, or
  (c) mean abs pixel difference between consecutive 32x32 thumbnails > --cut-mad
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def folder_scan(root, say, max_depth=4):
    say("\n== 4. Folder scan (extension counts per folder) ==")
    dirs = defaultdict(Counter)
    for p in root.rglob("*"):
        if p.is_file():
            dirs[p.parent][p.suffix.lower() or "(none)"] += 1
    for d in sorted(dirs):
        rel = d.relative_to(root)
        if len(rel.parts) <= max_depth:
            say(f"  {str(rel) or '.'}: {dict(dirs[d])}")
    say("\n  Non-image/non-xml files outside ImageSets (possible weather/light labels):")
    found = False
    for d in sorted(dirs):
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() not in IMG_EXT | {".xml"} and "ImageSets" not in p.parts:
                say(f"    {p.relative_to(root)}  ({p.stat().st_size} bytes)")
                found = True
    if not found:
        say("    (none found)")
    say("\n  Files inside any 'CategorizationData' folder:")
    found = False
    for p in sorted(root.rglob("*")):
        if p.is_file() and "CategorizationData" in p.parts:
            say(f"    {p.relative_to(root)}  ({p.stat().st_size} bytes)")
            found = True
    if not found:
        say("    (none found)")


def main(root, eda_dir, out, cut_mad):
    root, eda_dir, out = Path(root), Path(eda_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg=""):
        print(msg)
        log.append(str(msg))

    ann = pd.read_csv(eda_dir / "annotations.csv", dtype={"stem": str})
    imgs_info = pd.read_csv(eda_dir / "images.csv", dtype={"stem": str})
    paths = dict(zip(imgs_info.stem, imgs_info.path))
    stems = sorted(paths, key=int)
    nums = [int(s) for s in stems]
    cls_sets = ann.groupby("stem").cls.agg(lambda x: frozenset(x)).to_dict()
    area_ratio = ann.groupby("stem").area_ratio.mean().to_dict()
    say(f"Images: {len(stems)} | id range: {nums[0]}..{nums[-1]} | cut-mad threshold: {cut_mad}")

    # ---- adjacent pixel differences
    say("\n== 1. Computing adjacent-frame pixel differences ==")
    mad = {}
    prev = None
    for n, s in enumerate(stems):
        if n % 5000 == 0:
            print(f"  thumbnails {n}/{len(stems)}", flush=True)
        im = cv2.imread(paths[s], cv2.IMREAD_REDUCED_GRAYSCALE_4)
        th = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None and nums[n] == nums[n - 1] + 1:
            mad[s] = float(np.abs(th - prev).mean())   # difference to previous image
        prev = th
    pd.DataFrame({"stem": list(mad), "mad_to_prev": list(mad.values())}).to_csv(out / "adjacent_mad.csv", index=False)
    v = np.array(list(mad.values()))
    say(f"  adjacent MAD: median={np.median(v):.1f} p95={np.percentile(v, 95):.1f} "
        f"p99={np.percentile(v, 99):.1f} max={v.max():.1f}")
    say(f"  adjacent pairs with MAD > {cut_mad}: {(v > cut_mad).sum()}")

    # ---- segmentation (two definitions)
    def segment(use_cut):
        seq_of, cur = {}, 0
        for i, s in enumerate(stems):
            if i > 0:
                p = stems[i - 1]
                brk = (nums[i] != nums[i - 1] + 1) or not (cls_sets[s] & cls_sets[p])
                if use_cut and mad.get(s, 0) > cut_mad:
                    brk = True
                if brk:
                    cur += 1
            seq_of[s] = cur
        return seq_of

    seq_a, seq_b = segment(False), segment(True)
    say(f"\n== 2. Number of sequences ==")
    say(f"  class-run only            : {len(set(seq_a.values()))}")
    say(f"  class-run + scene-cut     : {len(set(seq_b.values()))}   <- used below")

    # ---- sequence table
    df = pd.DataFrame({"stem": stems, "id": nums, "seq": [seq_b[s] for s in stems]})
    df["mad_to_prev"] = df.stem.map(mad)
    df["area_ratio"] = df.stem.map(area_ratio)
    df["classes"] = df.stem.map(lambda s: sorted(cls_sets[s]))
    rows = []
    for sid, g in df.groupby("seq"):
        c = Counter(x for lst in g.classes for x in lst)
        rows.append(dict(
            seq=sid, start_id=g.id.min(), end_id=g.id.max(), n_frames=len(g),
            main_class=c.most_common(1)[0][0], all_classes="|".join(sorted(c)),
            mean_area_ratio=round(g.area_ratio.mean(), 4),
            mean_adj_mad=round(g.mad_to_prev.mean(), 2), max_adj_mad=round(g.mad_to_prev.max(), 1)))
    seqs = pd.DataFrame(rows)
    seqs.to_csv(out / "sequences.csv", index=False)
    say(f"\n== 3. Sequence length ==")
    say(f"  frames per sequence: min={seqs.n_frames.min()} median={seqs.n_frames.median():.0f} "
        f"mean={seqs.n_frames.mean():.1f} max={seqs.n_frames.max()}")
    say(f"  sequences shorter than 50 frames: {(seqs.n_frames < 50).sum()}")

    # ---- per-class table (by main class)
    pc = (seqs.groupby("main_class")
          .agg(n_sequences=("seq", "count"), n_frames=("n_frames", "sum"),
               min_len=("n_frames", "min"), max_len=("n_frames", "max"),
               mean_area_ratio=("mean_area_ratio", "mean"))
          .sort_values(["n_sequences", "n_frames"]))
    pc["mean_area_ratio"] = pc.mean_area_ratio.round(4)
    pc.to_csv(out / "class_sequence_table.csv")
    say("\n== Per-class sequences (sorted: fewest sequences first) ==")
    say(pc.to_string())
    say(f"\nClasses with 1 sequence : {(pc.n_sequences == 1).sum()}")
    say(f"Classes with 2 sequences: {(pc.n_sequences == 2).sum()}")
    say(f"Classes with 3+ sequences: {(pc.n_sequences >= 3).sum()}")
    say(f"Number of classes: {len(pc)}")

    folder_scan(root, say)
    (out / "step3_report.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\nDone. Report saved in: {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--eda", default="outputs/eda")
    ap.add_argument("--out", default="outputs/step3")
    ap.add_argument("--cut-mad", type=float, default=15.0)
    a = ap.parse_args()
    main(a.root, a.eda, a.out, a.cut_mad)