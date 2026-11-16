"""
Step 2: sequence / near-duplicate leakage check. READ-ONLY.

Run from D:\\FOD_project:
    python src\\step2_adjacency.py --root "data\\raw\\FODPascalVOCFormat-V.2.1"

Needs outputs\\eda\\annotations.csv (made by eda.py) for the class-run analysis.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
rng = np.random.default_rng(0)


def read_ids(path):
    return [l.split()[0] for l in path.read_text(errors="replace").splitlines() if l.strip()]


def main(root, eda_dir, out):
    root, eda_dir, out = Path(root), Path(eda_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg=""):
        print(msg)
        log.append(str(msg))

    # ---- images + split files
    images = {p.stem: p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT}
    txts = {p.stem: p for p in root.rglob("*.txt") if "ImageSets" in p.parts}
    test_ids, tv_ids = read_ids(txts["test"]), read_ids(txts["trainval"])
    test_set, tv_set = set(test_ids), set(tv_ids)
    both = test_set & tv_set
    test_only, tv_only = test_set - both, tv_set - both
    neither = set(images) - test_set - tv_set
    say(f"Images: {len(images)} | test-only: {len(test_only)} | trainval-only: {len(tv_only)} "
        f"| in both: {len(both)} | in neither: {len(neither)}")
    if not all(s.isdigit() for s in images):
        say("Image names are not all numeric; adjacency analysis needs numeric ids. Stopping.")
        return

    num = {s: int(s) for s in images}
    t_num = np.array(sorted(num[s] for s in test_only))
    v_num = np.array(sorted(num[s] for s in tv_only))
    all_nums = np.array(sorted(num.values()))

    # ---- 1. id adjacency
    say("\n== 1. Id adjacency: test ids that have a trainval id nearby ==")
    idx = np.searchsorted(v_num, t_num)
    left = np.where(idx > 0, v_num[np.clip(idx - 1, 0, len(v_num) - 1)], -10**9)
    right = np.where(idx < len(v_num), v_num[np.clip(idx, 0, len(v_num) - 1)], 10**9)
    dist = np.minimum(np.abs(t_num - left), np.abs(right - t_num))
    for k in (1, 2, 5, 10):
        say(f"  distance <= {k}: {100 * (dist <= k).mean():.1f}% of test images")
    say(f"  median distance to nearest trainval id: {np.median(dist):.0f}")
    p_tv = len(v_num) / (len(v_num) + len(t_num))
    say(f"  (If the split were a pure random per-image split with {100*p_tv:.0f}% trainval, "
        f"expected 'distance<=1' is about {100 * (1 - (1 - p_tv) ** 2):.0f}%)")

    # ---- 2. thumbnails -> pixel similarity
    say("\n== 2. Pixel similarity (32x32 grayscale thumbnails, mean abs difference 0-255) ==")
    ids_sorted = sorted(images, key=lambda s: num[s])
    thumbs = {}
    for n, s in enumerate(ids_sorted):
        if n % 5000 == 0:
            print(f"  loading thumbnails {n}/{len(ids_sorted)}", flush=True)
        im = cv2.imread(str(images[s]), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if im is None:
            continue
        thumbs[num[s]] = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)

    def mad(a, b):
        return float(np.abs(thumbs[a] - thumbs[b]).mean())

    keys = sorted(thumbs)
    adj = [mad(a, a + 1) for a in keys if (a + 1) in thumbs]
    ks = np.array(keys)
    ra, rb = rng.choice(ks, 20000), rng.choice(ks, 20000)
    rnd = [mad(a, b) for a, b in zip(ra, rb) if a != b]
    say(f"  adjacent ids (i, i+1)  : median={np.median(adj):.1f}  mean={np.mean(adj):.1f}")
    say(f"  random image pairs     : median={np.median(rnd):.1f}  mean={np.mean(rnd):.1f}")
    say(f"  adjacent pairs more similar than the typical random pair: "
        f"{100 * (np.array(adj) < np.median(rnd)).mean():.1f}%")

    # leakage: test image vs its nearest trainval neighbour (<=2) vs random trainval image
    near, base = [], []
    vset = set(v_num.tolist())
    v_list = v_num.tolist()
    for t in t_num.tolist():
        if t not in thumbs:
            continue
        for d in (1, 2):
            c = [x for x in (t - d, t + d) if x in vset and x in thumbs]
            if c:
                near.append(min(mad(t, x) for x in c))
                break
        base.append(mad(t, v_list[rng.integers(len(v_list))]))
    say(f"\n  test image vs nearest trainval neighbour (id within 2): "
        f"n={len(near)} median={np.median(near):.1f}")
    say(f"  test image vs RANDOM trainval image                  : "
        f"n={len(base)} median={np.median(base):.1f}")
    say("  -> if the first number is much smaller than the second, test images have near-copies in trainval.")

    # ---- 3. class runs
    ann_path = eda_dir / "annotations.csv"
    if ann_path.exists():
        say("\n== 3. Runs of consecutive ids with the same class ==")
        ann = pd.read_csv(ann_path, dtype={"stem": str})
        cls_sets = ann.groupby("stem").cls.agg(lambda x: frozenset(x)).to_dict()
        order = [s for s in ids_sorted if s in cls_sets]
        same = [bool(cls_sets[a] & cls_sets[b]) for a, b in zip(order[:-1], order[1:])
                if num[b] == num[a] + 1]
        say(f"  P(next id shares a class with this id) = {100 * np.mean(same):.1f}%")
        rand_same = np.mean([bool(cls_sets[order[i]] & cls_sets[order[j]])
                             for i, j in zip(rng.integers(len(order), size=20000),
                                             rng.integers(len(order), size=20000))])
        say(f"  same measure for random pairs          = {100 * rand_same:.1f}%")
        runs, cur, start = [], 1, order[0]
        for a, b in zip(order[:-1], order[1:]):
            if num[b] == num[a] + 1 and (cls_sets[a] & cls_sets[b]):
                cur += 1
            else:
                runs.append((start, cur))
                cur, start = 1, b
        runs.append((start, cur))
        lens = np.array([r[1] for r in runs])
        say(f"  number of runs: {len(runs)} | length: min={lens.min()} median={np.median(lens):.0f} "
            f"mean={lens.mean():.1f} max={lens.max()}")
        pd.DataFrame(runs, columns=["start_id", "length"]).to_csv(out / "class_runs.csv", index=False)
    else:
        say(f"\n(annotations.csv not found at {ann_path}; skipping class-run analysis)")

    # ---- 4. montages for your own eyes
    def tile(s, label):
        im = cv2.imread(str(images[s]))
        im = cv2.resize(im, (300, 300))
        cv2.rectangle(im, (0, 0), (300, 22), (0, 0, 0), -1)
        cv2.putText(im, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return im

    fmt = lambda n: f"{n:06d}"
    cands = [t for t in t_num.tolist() if (t - 1) in vset and (t + 1) in vset and fmt(t) in images]
    rows = []
    for t in rng.choice(cands, size=min(4, len(cands)), replace=False) if cands else []:
        t = int(t)
        rows.append(np.hstack([tile(fmt(t - 1), f"{fmt(t-1)} TRAINVAL"),
                               tile(fmt(t), f"{fmt(t)} TEST"),
                               tile(fmt(t + 1), f"{fmt(t+1)} TRAINVAL")]))
    if rows:
        cv2.imwrite(str(out / "montage_test_neighbours.jpg"), np.vstack(rows))
    tail = sorted(neither)
    if tail:
        pick = [tail[0], tail[len(tail) // 3], tail[2 * len(tail) // 3], tail[-1]]
        cv2.imwrite(str(out / "montage_unassigned.jpg"),
                    np.hstack([tile(s, f"{s} NEITHER") for s in pick]))

    (out / "step2_report.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\nDone. Report saved in: {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--eda", default="outputs/eda")
    ap.add_argument("--out", default="outputs/step2")
    a = ap.parse_args()
    main(a.root, a.eda, a.out)