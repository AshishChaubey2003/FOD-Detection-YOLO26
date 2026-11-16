"""
Step 5: Recording-session group construction. READ-ONLY on the dataset.
Koi split NAHI banta, koi training NAHI hoti.

Run from D:\\FOD_project:
    python src\\step5_group_construction.py                      (pehla run: calibration)
    python src\\step5_group_construction.py --pj-threshold 12.5   (threshold approve hone ke baad)

Needs: outputs\\eda\\{annotations,images}.csv, outputs\\step3\\sequences.csv,
       outputs\\step4a\\thumbs_32.npy, outputs\\step4b\\weather_light_by_image.csv
Writes ONLY to outputs\\step5\\.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

W = 5            # persistent-jump window (frames each side)
MIN_SIDE = 3     # window mein kam se kam itne frames chahiye, warna jump 'not evaluable'
FRAG = 30        # isse chhote group = fragment
EXPECTED = dict(N=33793, Dry=26577, Wet=7216, Bright=16942, Dim=12464, Dark=4387,
                cells={("Dry", "Bright"): 16942, ("Dry", "Dim"): 5248,
                       ("Dry", "Dark"): 4387, ("Wet", "Dim"): 7216})


def main(a):
    eda, s3, s4a, s4b = (Path(a.eda), Path(a.step3), Path(a.step4a), Path(a.step4b))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(m=""):
        print(m)
        log.append(str(m))

    # ---------------- load ----------------
    imgs = pd.read_csv(eda / "images.csv", dtype={"stem": str})
    ann = pd.read_csv(eda / "annotations.csv", dtype={"stem": str})
    wl = pd.read_csv(s4b / "weather_light_by_image.csv", dtype={"stem": str})
    sq = pd.read_csv(s3 / "sequences.csv")
    T = np.load(s4a / "thumbs_32.npy").astype(np.float32)

    imgs["id"] = imgs.stem.astype(int)
    imgs = imgs.sort_values("id").reset_index(drop=True)
    N = len(imgs)
    say(f"Images: {N} | thumbs: {T.shape} | weather/light rows: {len(wl)}")
    if T.shape[0] != N:
        raise SystemExit("ERROR: thumbs_32.npy ki length images.csv se match nahi karti. Step 4a dobara chalao.")
    if imgs.stem.duplicated().any():
        raise SystemExit("ERROR: images.csv mein duplicate ids.")
    if wl.stem.duplicated().any():
        raise SystemExit("ERROR: weather_light_by_image.csv mein duplicate ids.")
    df = imgs.merge(wl[["stem", "weather", "light"]], on="stem", how="left")
    if df.weather.isna().any() or df.light.isna().any():
        raise SystemExit(f"ERROR: {int(df.weather.isna().sum())} images ka weather/light label join nahi hua.")

    main_box = ann.sort_values("area").groupby("stem").tail(1).set_index("stem")
    df["class_name"] = df.stem.map(main_box.cls)
    cls_sets = ann.groupby("stem").cls.agg(lambda x: frozenset(x)).to_dict()
    if df.class_name.isna().any():
        raise SystemExit("ERROR: kuch images ka annotation nahi mila.")

    df["sequence_id"] = -1
    for r in sq.itertuples():
        df.loc[(df.id >= r.start_id) & (df.id <= r.end_id), "sequence_id"] = r.seq
    ids = df.id.values
    wea, lig = df.weather.values, df.light.values
    stems = df.stem.tolist()

    # ---------------- label-based boundaries (pair i-1 | i) ----------------
    b_cls = np.zeros(N, bool)
    b_wea = np.zeros(N, bool)
    b_lig = np.zeros(N, bool)
    b_gap = np.zeros(N, bool)
    for i in range(1, N):
        b_cls[i] = not (cls_sets[stems[i]] & cls_sets[stems[i - 1]])
        b_wea[i] = wea[i] != wea[i - 1]
        b_lig[i] = lig[i] != lig[i - 1]
        b_gap[i] = ids[i] - ids[i - 1] != 1
    b_label = b_cls | b_wea | b_lig | b_gap
    say(f"Label-based boundaries: class {b_cls.sum()} | weather {b_wea.sum()} | light {b_lig.sum()} "
        f"| id-gap {b_gap.sum()} | union {b_label.sum()}  -> label-only groups = {b_label.sum() + 1}")

    # label-segment edges (heuristic window inhi ke andar clip hoga)
    seg_id = np.cumsum(b_label)
    seg_start = {s: np.where(seg_id == s)[0][0] for s in np.unique(seg_id)}
    seg_end = {s: np.where(seg_id == s)[0][-1] for s in np.unique(seg_id)}

    # ---------------- persistent pixel jump ----------------
    pj = np.full(N, np.nan)
    for i in range(1, N):
        l0 = max(i - W, seg_start[seg_id[i - 1]])
        r1 = min(i + W - 1, seg_end[seg_id[i]])
        nl, nr = i - l0, r1 - i + 1
        if nl < MIN_SIDE or nr < MIN_SIDE:
            continue
        pj[i] = np.abs(T[l0:i].mean(0) - T[i:r1 + 1].mean(0)).mean()
    ev = ~np.isnan(pj)

    # ---------------- calibration ----------------
    within = ev & ~b_label
    lab_ev = ev & b_label
    pc = [50, 90, 95, 99, 99.9]
    say("\n== Calibration: persistent jump (|mean(prev 5) - mean(next 5)|, 32x32) ==")
    say(f"Evaluable positions: within-segment {within.sum()} | label boundaries {lab_ev.sum()} "
        f"(not evaluable label boundaries: {(b_label & ~ev).sum()})")
    if within.sum():
        v = pj[within]
        say("Within-segment (no label change) percentiles: " +
            ", ".join(f"p{p}={np.percentile(v, p):.2f}" for p in pc) + f", max={v.max():.2f}")
    if lab_ev.sum():
        v = pj[lab_ev]
        say("At label boundaries percentiles: " +
            ", ".join(f"p{p}={np.percentile(v, p):.2f}" for p in (5, 10, 25, 50, 75)) + f", min={v.min():.2f}")
        for nm, b in (("class", b_cls), ("weather", b_wea), ("light", b_lig)):
            m = b & ev
            if m.sum():
                say(f"   {nm:8s} boundaries: n={m.sum():3d} median={np.median(pj[m]):.2f} min={pj[m].min():.2f}")
    proposed = float(np.percentile(pj[within], 99.9)) if within.sum() else float("nan")
    say(f"Proposed threshold (within-segment p99.9): {proposed:.2f}  <- sirf SUGGESTION, aap decide karenge")
    if within.sum():
        top = np.argsort(-np.where(within, pj, -1))[:15]
        say("Top within-segment jump candidates (id: jump):")
        say("   " + ", ".join(f"{ids[i]}:{pj[i]:.1f}" for i in top))
    cal = pd.DataFrame({"id": ids, "persistent_jump": pj, "label_boundary": b_label,
                        "b_class": b_cls, "b_weather": b_wea, "b_light": b_lig, "b_gap": b_gap})
    cal.to_csv(out / "calibration_jumps.csv", index=False)

    # ---------------- heuristic boundaries (NMS) ----------------
    thr = a.pj_threshold
    b_heur = np.zeros(N, bool)
    if thr is None:
        say("\n[MODE: CALIBRATION] --pj-threshold nahi diya -> heuristic boundaries OFF. "
            "Neeche ke outputs 'label-only baseline' hain.")
    else:
        cand = np.where(within & (pj > thr))[0]
        cand = cand[np.argsort(-pj[cand])]
        taken = np.zeros(N, bool)
        for i in cand:
            if not taken[max(0, i - W):i + W + 1].any():
                b_heur[i] = True
                taken[i] = True
        say(f"\n[MODE: FINAL] threshold={thr}: raw candidates {len(cand)} -> after non-max suppression "
            f"(+-{W} frames) heuristic boundaries {b_heur.sum()}")

    # ---------------- groups ----------------
    bound = b_label | b_heur
    gidx = np.cumsum(bound)
    df["gnum"] = gidx
    ng = int(gidx.max()) + 1
    width = max(3, len(str(ng)))
    df["group_id"] = ["GROUP_" + str(g + 1).zfill(width) for g in gidx]

    def reason(i):
        if i == 0:
            return "dataset_start", "start"
        parts = [n for n, b in (("class", b_cls), ("weather", b_wea), ("light", b_lig), ("id_gap", b_gap)) if b[i]]
        if parts:
            return "+".join(parts), "label-based"
        return "pixel_jump", "heuristic"

    start_reason, start_type = {}, {}
    for i in np.where(bound)[0].tolist() + [0]:
        r, t = reason(i)
        start_reason[gidx[i]], start_type[gidx[i]] = r, t
    df["grouping_reason"] = df.gnum.map(start_reason)
    df["start_boundary_type"] = df.gnum.map(start_type)

    cols = ["stem", "path", "class_name", "weather", "light", "sequence_id", "group_id",
            "grouping_reason", "start_boundary_type"]
    asg = df[cols].rename(columns={"stem": "image_id", "path": "image_path"})
    asg.to_csv(out / "group_assignment.csv", index=False)

    # ---------------- group stats ----------------
    rows = []
    for g, d in df.groupby("gnum"):
        mode_cls = d.class_name.mode().iloc[0]
        rows.append(dict(group_id=d.group_id.iloc[0], start_id=int(d.id.min()), end_id=int(d.id.max()),
                         n_frames=len(d), class_name=mode_cls, n_main_classes=d.class_name.nunique(),
                         weather="/".join(sorted(d.weather.unique())), light="/".join(sorted(d.light.unique())),
                         n_step3_seqs=d.sequence_id.nunique(),
                         start_reason=start_reason[g], start_type=start_type[g]))
    gs = pd.DataFrame(rows).sort_values("start_id").reset_index(drop=True)
    gs["end_type"] = gs.start_type.shift(-1).fillna("end")
    gs["end_reason"] = gs.start_reason.shift(-1).fillna("dataset_end")
    gs.to_csv(out / "group_stats.csv", index=False)

    say(f"\n== Group statistics ==")
    say(f"Total groups: {len(gs)} | images assigned: {len(df)} | unassigned: {N - len(df)}")
    say(f"Group length: min {gs.n_frames.min()} | median {gs.n_frames.median():.0f} | max {gs.n_frames.max()}")
    say(f"Group start type counts: {gs.start_type.value_counts().to_dict()}")

    # class table
    ct = gs.groupby("class_name").agg(n_groups=("group_id", "count"), frames=("n_frames", "sum"),
                                      min_len=("n_frames", "min"), median_len=("n_frames", "median"),
                                      max_len=("n_frames", "max")).reset_index()
    ct["flag"] = np.where(ct.n_groups <= 2, "ONLY 1-2 GROUPS (split mein dikkat)", "")
    ct.to_csv(out / "class_table.csv", index=False)
    say("\n== Class-wise ==")
    say(ct.to_string(index=False))

    # condition table
    ctab = []
    for (w, l), d in df.groupby(["weather", "light"]):
        gg = gs[(gs.weather == w) & (gs.light == l)]
        ctab.append(dict(weather=w, light=l, n_groups=len(gg), frames=len(d), pct=round(100 * len(d) / N, 2),
                         min_len=gg.n_frames.min(), median_len=gg.n_frames.median(), max_len=gg.n_frames.max()))
    ctab = pd.DataFrame(ctab)
    mixed = gs[gs.weather.str.contains("/") | gs.light.str.contains("/")]
    ctab["mixed_condition_groups_total"] = len(mixed)
    ctab.to_csv(out / "condition_table.csv", index=False)
    say("\n== Condition-wise ==")
    say(ctab.to_string(index=False))

    # fragments
    fr = []
    for k in gs.index[gs.n_frames < FRAG]:
        g = gs.loc[k]
        prv = gs.loc[k - 1] if k > 0 else None
        nxt = gs.loc[k + 1] if k < len(gs) - 1 else None

        def same(o):
            return o is not None and (o.class_name, o.weather, o.light) == (g.class_name, g.weather, g.light)
        sb, eb = g.start_type, g.end_type
        both_heur = sb == "heuristic" and eb == "heuristic"
        both_label = sb in ("label-based", "start") and eb in ("label-based", "end")
        if both_heur and (same(prv) or same(nxt)):
            rec = "MERGE (dono boundary heuristic, labels padosi se same)"
        elif both_label:
            rec = "KEEP SEPARATE (dono boundary label-based, condition sach mein badli)"
        elif (sb == "heuristic" and same(prv)) or (eb == "heuristic" and same(nxt)):
            rec = "REVIEW (ek taraf heuristic: us padosi se merge possible)"
        else:
            rec = "REVIEW"
        fr.append(dict(group_id=g.group_id, start_id=g.start_id, end_id=g.end_id, n_frames=g.n_frames,
                       class_name=g.class_name, weather=g.weather, light=g.light,
                       start_type=sb, start_reason=g.start_reason, end_type=eb, end_reason=g.end_reason,
                       prev_group=None if prv is None else prv.group_id,
                       prev_same_labels=same(prv), next_group=None if nxt is None else nxt.group_id,
                       next_same_labels=same(nxt), recommendation=rec))
    fr = pd.DataFrame(fr)
    fr.to_csv(out / "fragment_review.csv", index=False)
    say(f"\n== Fragments (<{FRAG} frames): {len(fr)} == (koi merge apply NAHI hua)")
    if len(fr):
        say(fr.recommendation.value_counts().to_string())

    # suspicious boundaries
    low = float(np.percentile(pj[within], 95)) if within.sum() else 0.0
    sus = []
    for i in np.where(b_label)[0]:
        if i == 0:
            continue
        kind = "+".join(n for n, b in (("class", b_cls), ("weather", b_wea), ("light", b_lig), ("id_gap", b_gap)) if b[i])
        if not ev[i]:
            sus.append(dict(id_before=ids[i - 1], id_after=ids[i], type="label:" + kind, persistent_jump=np.nan,
                            flag="label boundary, jump not evaluable (chhota segment)", step6_advice="tie adjacent groups (caution)"))
        elif pj[i] < low:
            sus.append(dict(id_before=ids[i - 1], id_after=ids[i], type="label:" + kind, persistent_jump=round(pj[i], 2),
                            flag=f"label badla par images continuous dikhti hain (jump < within-seg p95={low:.2f})",
                            step6_advice="tie: dono groups ek hi split mein"))
    for i in np.where(b_heur)[0]:
        sus.append(dict(id_before=ids[i - 1], id_after=ids[i], type="heuristic:pixel_jump",
                        persistent_jump=round(pj[i], 2), flag="pixel jump bada, labels same (object move ya nayi session?)",
                        step6_advice="tie unless visually confirmed"))
    sus = pd.DataFrame(sus)
    sus.to_csv(out / "suspicious_boundaries.csv", index=False)
    say(f"\n== Suspicious boundaries: {len(sus)} ==")
    if len(sus):
        say(sus.type.value_counts().to_string())

    # ---------------- validation ----------------
    say("\n== VALIDATION ==")
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        say(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    chk("har image exactly ek group", asg.image_id.is_unique and asg.group_id.notna().all(), f"({len(asg)} rows)")
    chk("total images == images.csv", len(asg) == N)
    chk("group sizes ka sum == N", gs.n_frames.sum() == N)
    chk("mixed-condition groups == 0", len(mixed) == 0, f"(found {len(mixed)})")
    inside = gidx[1:] == gidx[:-1]
    chk("label boundary hamesha group todta hai", not (b_label[1:] & inside).any())
    say(f"Weather counts: {df.weather.value_counts().to_dict()} | Light counts: {df.light.value_counts().to_dict()}")
    say(f"Cells: {df.groupby(['weather', 'light']).size().to_dict()}")
    if N == EXPECTED["N"]:
        vc = df.weather.value_counts()
        lc = df.light.value_counts()
        cells = df.groupby(["weather", "light"]).size().to_dict()
        chk("weather counts expected se match", vc.get("Dry") == EXPECTED["Dry"] and vc.get("Wet") == EXPECTED["Wet"])
        chk("light counts expected se match", all(lc.get(k) == EXPECTED[k] for k in ("Bright", "Dim", "Dark")))
        chk("4 condition cells expected se match", cells == EXPECTED["cells"])
    else:
        say(f"(N={N} != 33793: expected-count checks skip kiye -> ye test/synthetic data lagta hai)")
    say("OVERALL: " + ("ALL PASS" if ok else "KUCH FAIL - upar dekho"))

    # ---------------- montages ----------------
    try:
        make_montages(df, gs, sus, T, out, say)
    except Exception as e:
        say(f"[montage skipped: {e}]")

    say("\nLimitations: groups 'best estimate' hain, confirmed sessions nahi (timestamp/GPS metadata nahi). "
        "32x32 jump object size/motion se prabhavit. Wet hamesha Dim hai (confounded).")
    (out / "step5_report.txt").write_text("\n".join(log), encoding="utf-8")
    say(f"\nSaved to {out}")


def tile(path, label, size=220):
    im = cv2.imread(str(path))
    if im is None:
        im = np.zeros((size, size, 3), np.uint8)
    im = cv2.resize(im, (size, size))
    cv2.rectangle(im, (0, 0), (size, 34), (0, 0, 0), -1)
    for k, ln in enumerate(label.split("\n")):
        cv2.putText(im, ln, (3, 13 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def make_montages(df, gs, sus, T, out, say, per_row=6):
    rows = []
    for (w, l) in [("Dry", "Bright"), ("Dry", "Dim"), ("Dry", "Dark"), ("Wet", "Dim")]:
        gg = gs[(gs.weather == w) & (gs.light == l)].sort_values("group_id")
        chosen, used = [], set()
        for g in gg.itertuples():
            if g.class_name not in used and len(chosen) < per_row:
                chosen.append(g)
                used.add(g.class_name)
        for g in gg.itertuples():
            if len(chosen) >= per_row:
                break
            if g not in chosen:
                chosen.append(g)
        tiles = []
        for g in chosen:
            d = df[df.group_id == g.group_id]
            r = d.iloc[len(d) // 2]
            tiles.append(tile(r.path, f"{w}/{l} {r.group_id}\n{r.class_name} id={r.stem}"))
        while len(tiles) < per_row:
            tiles.append(np.zeros((220, 220, 3), np.uint8))
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(out / "montage_conditions.jpg"), np.vstack(rows))
    say("Saved montage_conditions.jpg (4 rows: Dry/Bright, Dry/Dim, Dry/Dark, Wet/Dim)")

    if len(sus):
        pairs = []
        for r in sus.head(8).itertuples():
            b = df[df.id == r.id_before].iloc[0]
            a_ = df[df.id == r.id_after].iloc[0]
            pairs.append(np.hstack([tile(b.path, f"BEFORE {b.stem} {b.class_name}\n{b.weather}/{b.light}"),
                                    tile(a_.path, f"AFTER {a_.stem} {a_.class_name}\n{a_.weather}/{a_.light}")]))
        if pairs:
            if len(pairs) % 2:
                pairs.append(np.zeros_like(pairs[0]))
            grid = np.vstack([np.hstack(pairs[i:i + 2]) for i in range(0, len(pairs), 2)])
            cv2.imwrite(str(out / "montage_suspicious_boundaries.jpg"), grid)
            say("Saved montage_suspicious_boundaries.jpg (pehli 8 suspicious boundaries)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda", default="outputs/eda")
    ap.add_argument("--step3", default="outputs/step3")
    ap.add_argument("--step4a", default="outputs/step4a")
    ap.add_argument("--step4b", default="outputs/step4b")
    ap.add_argument("--out", default="outputs/step5")
    ap.add_argument("--pj-threshold", type=float, default=None)
    main(ap.parse_args())