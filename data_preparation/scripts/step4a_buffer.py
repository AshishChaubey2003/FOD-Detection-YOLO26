"""
Step 4a: buffer-gap analysis. READ-ONLY on the dataset.

Run from D:\\FOD_project:
    python src\\step4a_buffer.py --root "data\\raw\\FODPascalVOCFormat-V.2.1"

Needs outputs\\eda\\annotations.csv + images.csv (eda.py) and, optionally,
outputs\\step3\\sequences.csv (step3_sequences.py).
Writes only to outputs\\step4a\\.
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)


def load_thumbs(paths, cache):
    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == len(paths):
            print(f"  loaded cached thumbnails: {cache}")
            return arr
    arr = np.zeros((len(paths), 32, 32), np.uint8)
    for n, p in enumerate(paths):
        if n % 5000 == 0:
            print(f"  thumbnails {n}/{len(paths)}", flush=True)
        im = cv2.imread(p, cv2.IMREAD_REDUCED_GRAYSCALE_4)
        arr[n] = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)
    np.save(cache, arr)
    return arr


def main(eda_dir, step3_dir, out, ks):
    eda_dir, step3_dir, out = Path(eda_dir), Path(step3_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg=""):
        print(msg)
        log.append(str(msg))

    ann = pd.read_csv(eda_dir / "annotations.csv", dtype={"stem": str})
    imgs = pd.read_csv(eda_dir / "images.csv", dtype={"stem": str})
    imgs["id"] = imgs.stem.astype(int)
    imgs = imgs.sort_values("id").reset_index(drop=True)
    N = len(imgs)
    if not (imgs.id.diff().dropna() == 1).all():
        say("Image ids are not contiguous; this analysis needs contiguous ids. Stopping.")
        return
    stems = imgs.stem.tolist()

    # main (largest) box per image -> class, area ratio
    main_box = ann.sort_values("area").groupby("stem").tail(1).set_index("stem")
    cls1 = np.array([main_box.cls[s] for s in stems])
    area1 = np.array([main_box.area_ratio[s] for s in stems])
    cls_sets = ann.groupby("stem").cls.agg(lambda x: frozenset(x)).to_dict()

    # ---- sequence definition A: class runs (conservative: 54)
    runA = np.zeros(N, int)
    for i in range(1, N):
        runA[i] = runA[i - 1] + (0 if (cls_sets[stems[i]] & cls_sets[stems[i - 1]]) else 1)
    defs = {"A: class-run": runA}
    s3 = step3_dir / "sequences.csv"
    if s3.exists():
        sq = pd.read_csv(s3)
        segB = np.zeros(N, int)
        ids = imgs.id.values
        for r in sq.itertuples():
            segB[(ids >= r.start_id) & (ids <= r.end_id)] = r.seq
        defs["B: class-run+scene-cut (Step 3)"] = segB
    for name, d in defs.items():
        say(f"Sequence definition {name}: {len(set(d))} sequences")

    say("\n== Loading thumbnails ==")
    T = load_thumbs(imgs.path.tolist(), out / "thumbs_32.npy").astype(np.int16)

    def mad_k(k):
        return np.abs(T[:-k] - T[k:]).mean(axis=(1, 2))

    # ---- baselines
    a, b = rng.integers(0, N, 60000), rng.integers(0, N, 60000)
    keep = a != b
    rand_any = np.abs(T[a[keep]] - T[b[keep]]).mean(axis=(1, 2))
    sc = []
    skipped = []
    for c in sorted(set(cls1)):
        idx = np.where(cls1 == c)[0]
        if len(set(runA[idx])) < 2:
            skipped.append(str(c))
            continue
        i, j = rng.choice(idx, 2500), rng.choice(idx, 2500)
        m = runA[i] != runA[j]
        sc.append(np.abs(T[i[m]] - T[j[m]]).mean(axis=(1, 2)))
    rand_same = np.concatenate(sc)
    tau5, tau1 = np.percentile(rand_same, 5), np.percentile(rand_same, 1)
    say("\n== Baselines (MAD, 0-255) ==")
    say(f"  random pairs (any)                         : median={np.median(rand_any):.1f} "
        f"p5={np.percentile(rand_any, 5):.1f}")
    say(f"  random pairs, SAME class, DIFFERENT video : median={np.median(rand_same):.1f} "
        f"p5={tau5:.1f} p1={tau1:.1f}  (n={len(rand_same)})")
    if skipped:
        say(f"  classes with only 1 class-run (no same-class/different-run baseline): {skipped}")
    say(f"  near-duplicate threshold tau5={tau5:.1f}, strict tau1={tau1:.1f} (from the baseline, not a universal constant)")

    # ---- decay table per definition
    kstar = {}
    tables = {}
    D = {k: mad_k(k) for k in ks if k < N}
    for name, seg in defs.items():
        rows = []
        for k, d in D.items():
            ok = seg[:-k] == seg[k:]
            v = d[ok]
            if len(v) == 0:
                rows.append(dict(k=k, n_pairs=0))
                continue
            rows.append(dict(k=k, n_pairs=len(v), median=np.median(v), p10=np.percentile(v, 10),
                             p90=np.percentile(v, 90), pct_below_tau5=100 * (v < tau5).mean(),
                             pct_below_tau1=100 * (v < tau1).mean()))
        t = pd.DataFrame(rows).round(2)
        tables[name] = t
        t.to_csv(out / f"decay_{name[0]}.csv", index=False)
        say(f"\n== Within-sequence MAD vs frame distance k | definition {name} ==")
        say(t.to_string(index=False))
        tv = t[t.n_pairs > 0].reset_index(drop=True)
        good = (tv.pct_below_tau5 <= 5.0).tolist()
        first = next((int(tv.k[i]) for i, g in enumerate(good) if g), None)
        stable = next((int(tv.k[i]) for i in range(len(good)) if all(good[i:])), None)
        kstar[name] = stable
        say(f"  first k with <=5% of pairs below tau5       : {first if first else 'NONE in tested range'}")
        say(f"  STABLE k* (this k and every larger tested k): {stable if stable else 'NONE in tested range'}")

    # ---- per-class (definition A)
    say("\n== Per class (definition A): % of within-run pairs below tau5, by k ==")
    rows = []
    for c in sorted(set(cls1)):
        r = {"class": c}
        for k, d in D.items():
            ok = (runA[:-k] == runA[k:]) & (cls1[:-k] == c)
            r[f"k={k}"] = round(100 * (d[ok] < tau5).mean(), 1) if ok.sum() >= 100 else np.nan
        rows.append(r)
    pc = pd.DataFrame(rows)
    pc.to_csv(out / "per_class_defA.csv", index=False)
    say(pc.to_string(index=False))
    say("  (NaN = fewer than 100 pairs at that k for that class)")
    kcols = [c for c in pc.columns if c.startswith("k=")]
    per_cls_k = []
    for r in pc.itertuples(index=False):
        vals = dict(zip(pc.columns, r))
        seq_ = [(int(c[2:]), vals[c]) for c in kcols if not np.isnan(vals[c])]
        flags = [v <= 5.0 for _, v in seq_]
        st = next((seq_[i][0] for i in range(len(flags)) if all(flags[i:])), np.nan)
        per_cls_k.append(st)
    pcs = pd.Series(per_cls_k)
    say(f"  per-class STABLE k*: median={pcs.median()} max={pcs.max()} "
        f"| classes never reaching it: {int(pcs.isna().sum())}")

    # ---- per-sequence (definition A)
    rows = []
    for sid in sorted(set(runA)):
        pos = np.where(runA == sid)[0]
        r = dict(seq=sid, cls=pd.Series(cls1[pos]).mode()[0], length=len(pos), start_id=imgs.id[pos[0]])
        for k, d in D.items():
            r[f"k={k}"] = round(float(np.median(d[pos[0]:pos[-1] - k + 1])), 1) if len(pos) > k else np.nan
        rows.append(r)
    ps = pd.DataFrame(rows)
    ps.to_csv(out / "per_sequence_defA.csv", index=False)
    say("\n== Per-sequence median MAD (definition A): spread across sequences ==")
    say(ps[[c for c in ps.columns if c.startswith("k=")]].describe().loc[["count", "min", "50%", "max"]].round(1).to_string())

    # ---- object-size confound
    say("\n== Limitation check: object size vs MAD (definition A, within-run pairs) ==")
    bins = [0, 0.01, 0.05, 0.15, 1.01]
    lab = ["<1%", "1-5%", "5-15%", ">15%"]
    sb = pd.cut(pd.Series(area1), bins=bins, labels=lab, right=False)
    rows = []
    for k in [k for k in (1, 25) if k in D]:
        d = D[k]
        ok = runA[:-k] == runA[k:]
        for L in lab:
            m = ok & (sb.values[:-k] == L)
            rows.append(dict(k=k, object_area=L, n_pairs=int(m.sum()),
                             median_mad=round(float(np.median(d[m])), 1) if m.sum() else np.nan))
    sbt = pd.DataFrame(rows)
    sbt.to_csv(out / "size_bins.csv", index=False)
    say(sbt.to_string(index=False))

    # ---- plot
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, t in tables.items():
        t = t[t.n_pairs > 0]
        ax.plot(t.k, t["median"], marker="o", label=f"median MAD | {name}")
        ax.plot(t.k, t["p10"], ls=":", marker=".", label=f"p10 | {name}")
    ax.axhline(np.median(rand_same), color="k", ls="--", label="random same-class, other video: median")
    ax.axhline(tau5, color="r", ls="--", label="tau5 (near-duplicate threshold)")
    ax.set_xscale("log"); ax.set_xlabel("frame distance k"); ax.set_ylabel("MAD (0-255)")
    ax.set_title("Visual difference vs frame distance (within sequence)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "buffer_decay.png", dpi=150)

    (out / "step4a_report.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\nDone. Report saved in: {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="unused (kept for a consistent command line)")
    ap.add_argument("--eda", default="outputs/eda")
    ap.add_argument("--step3", default="outputs/step3")
    ap.add_argument("--out", default="outputs/step4a")
    ap.add_argument("--ks", default="1,5,10,25,50,100,200")
    a = ap.parse_args()
    main(a.eda, a.step3, a.out, [int(x) for x in a.ks.split(",")])