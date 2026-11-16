"""
Step 6 PLANNING analysis. READ-ONLY. Koi split NAHI banta.
Koi train/val/test folder ya split CSV nahi banti. Sirf analysis tables + report.

Run from D:\\FOD_project:
    python src\\step6_plan_analysis.py

Reads : outputs\\step5\\*  (final label-only), outputs\\step3\\class_sequence_table.csv,
        outputs\\eda\\annotations.csv
Writes: ONLY outputs\\step6_plan\\
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

BUFFERS = (0, 100, 200)
VAL_FRAC, TEST_FRAC = 0.10, 0.20
CELLS = [("Dry", "Bright"), ("Dry", "Dim"), ("Dry", "Dark"), ("Wet", "Dim")]


def main(a):
    s5, s3, eda, heur = Path(a.step5), Path(a.step3), Path(a.eda), Path(a.heur)
    out = Path(a.out)
    if out.resolve() == s5.resolve() or out.name != "step6_plan":
        raise SystemExit("ERROR: output folder 'step6_plan' hona chahiye (original outputs safe rakhne ke liye).")
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(m=""):
        print(m)
        log.append(str(m))

    asg = pd.read_csv(s5 / "group_assignment.csv", dtype={"image_id": str})
    gs = pd.read_csv(s5 / "group_stats.csv")
    sus = pd.read_csv(s5 / "suspicious_boundaries.csv")
    asg["id"] = asg.image_id.astype(int)
    asg = asg.sort_values("id").reset_index(drop=True)
    N = len(asg)

    # ======== 1. FINALIZATION CHECKS ========
    say("=" * 70)
    say("1. STEP 5 FINALIZATION CHECKS (verified facts)")
    say("=" * 70)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        say(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    chk("image_id unique, total rows", asg.image_id.is_unique, f"(rows={N})")
    chk("33,793 images (full dataset)", N == 33793, f"(found {N})")
    chk("unique groups == 116", asg.group_id.nunique() == 116, f"(found {asg.group_id.nunique()})")
    chk("group_stats frames sum == rows", gs.n_frames.sum() == N)
    chk("group_stats rows == unique groups", len(gs) == asg.group_id.nunique())
    sb = asg.drop_duplicates("group_id").start_boundary_type.value_counts().to_dict()
    chk("heuristic boundaries == 0 (label-only final)", sb.get("heuristic", 0) == 0, f"(types: {sb})")
    chk("mixed-condition groups == 0",
        not (gs.weather.str.contains("/") | gs.light.str.contains("/")).any())
    cells = asg.groupby(["weather", "light"]).size().to_dict()
    say(f"Cells: {cells}")
    if N == 33793:
        chk("4 cells expected", cells == {("Dry", "Bright"): 16942, ("Dry", "Dim"): 5248,
                                          ("Dry", "Dark"): 4387, ("Wet", "Dim"): 7216})
    if heur.exists():
        say(f"\nReference folder {heur} (read-only stat, files ke last-modified):")
        for p in sorted(heur.glob("*.csv"))[:6]:
            say(f"   {p.name}: {time.strftime('%d-%m-%Y %H:%M', time.localtime(os.path.getmtime(p)))}")
    else:
        say(f"(note: {heur} nahi mila)")
    say("OVERALL finalization: " + ("ALL PASS" if ok else "KUCH FAIL - upar dekho"))

    # ======== 2. RARE-CLASS RECONCILE ========
    say("\n" + "=" * 70)
    say("2. RARE-CLASS DISCREPANCY (Step 3 vs class-run A vs Step 5)")
    say("=" * 70)
    ann = pd.read_csv(eda / "annotations.csv", dtype={"stem": str})
    cls_sets = ann.groupby("stem").cls.agg(lambda x: frozenset(x)).to_dict()
    stems = asg.image_id.tolist()
    runA = np.zeros(N, int)
    for i in range(1, N):
        runA[i] = runA[i - 1] + (0 if (cls_sets[stems[i]] & cls_sets[stems[i - 1]]) else 1)
    asg["runA"] = runA
    A = asg.groupby("class_name").runA.nunique().rename("A_class_runs")
    g5 = gs.groupby("class_name").agg(step5_groups=("group_id", "count"), frames=("n_frames", "sum"))
    mc = gs[gs.n_main_classes > 1]
    rec = g5.join(A, how="outer")
    p3 = s3 / "class_sequence_table.csv"
    if p3.exists():
        t3 = pd.read_csv(p3).set_index("main_class")
        rec = rec.join(t3.n_sequences.rename("step3_B_sequences"), how="outer")
    rec["rare_A(<=2)"] = rec.A_class_runs <= 2
    rec["rare_step5(<=2)"] = rec.step5_groups <= 2
    if "step3_B_sequences" in rec:
        rec["rare_step3B(<=2)"] = rec.step3_B_sequences <= 2
    rec = rec.reset_index().rename(columns={"index": "class_name"}).sort_values("step5_groups")
    rec.to_csv(out / "rare_class_reconcile.csv", index=False)
    say(rec.to_string(index=False))
    say("\nCounts of classes with <=2:")
    for c in [c for c in rec.columns if c.startswith("rare_")]:
        say(f"   {c}: {int(rec[c].sum())} -> {rec.loc[rec[c], 'class_name'].tolist()}")
    say(f"Groups jinme >1 main class hai: {len(mc)} (0 ho to class_name mode ka masla nahi)")
    diff = rec[(rec["rare_step5(<=2)"]) != (rec["rare_A(<=2)"])]
    say(f"Classes jahan Step5 aur class-run A ka 'rare' verdict alag hai: {diff.class_name.tolist()}")

    # ======== 3. TIED GROUPS ========
    say("\n" + "=" * 70)
    say("3. TIED BOUNDARIES AUR ASSIGNMENT UNITS")
    say("=" * 70)
    id2g = dict(zip(asg.id, asg.group_id))
    ties, keep = [], []
    for r in sus.itertuples():
        row = dict(id_before=r.id_before, id_after=r.id_after, type=r.type,
                   persistent_jump=r.persistent_jump, g_before=id2g.get(r.id_before), g_after=id2g.get(r.id_after))
        (ties if r.type == "label:class" else keep).append(row)
    ties, keep = pd.DataFrame(ties), pd.DataFrame(keep)
    say(f"suspicious_boundaries.csv rows: {len(sus)} | tie candidates (label:class): {len(ties)} | other: {len(keep)}")
    if len(ties) != 7:
        say("WARNING: 7 tied class boundaries expected, found %d. Check karein." % len(ties))
    for t in (ties, keep):
        if len(t):
            say(t.to_string(index=False))
    parent = {g: g for g in gs.group_id}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in ties.itertuples():
        if r.g_before and r.g_after and r.g_before != r.g_after:
            parent[find(r.g_before)] = find(r.g_after)
    gs["root"] = gs.group_id.map(find)
    order = gs.groupby("root").start_id.min().sort_values().index.tolist()
    uid = {r: f"UNIT_{i + 1:03d}" for i, r in enumerate(order)}
    gs["unit_id"] = gs.root.map(uid)
    ug = gs.groupby("unit_id").size()
    say(f"\nGroups: {len(gs)} -> assignment units: {gs.unit_id.nunique()} "
        f"(units with >1 group: {(ug > 1).sum()}; sabse badi chain: {ug.max()} groups)")
    units = gs.groupby("unit_id").agg(n_groups=("group_id", "count"), frames=("n_frames", "sum"),
                                      groups=("group_id", lambda x: "+".join(x)),
                                      classes=("class_name", lambda x: "+".join(sorted(set(x)))),
                                      weather=("weather", "first"), light=("light", "first"),
                                      n_cells=("light", lambda x: x.nunique())).reset_index()
    units.to_csv(out / "assignment_units.csv", index=False)
    say(units[units.n_groups > 1].to_string(index=False))
    if (units.n_cells > 1).any():
        say("WARNING: koi unit multiple lighting values ka hai.")

    # untied class boundaries with lowest jump (review ke liye)
    pj = s5 / "calibration_jumps.csv"
    if pj.exists():
        cj = pd.read_csv(pj)
        c = cj[cj.b_class & cj.persistent_jump.notna()].copy()
        tied_after = set(ties.id_after) if len(ties) else set()
        c = c[~c.id.isin(tied_after)].sort_values("persistent_jump").head(15)
        c["g_before"] = (c.id - 1).map(id2g)
        c["g_after"] = c.id.map(id2g)
        c[["id", "persistent_jump", "g_before", "g_after"]].to_csv(out / "untied_class_boundaries_lowest_jump.csv", index=False)
        say("\nUntied class boundaries jin par jump sabse kam hai (inhe dekhna chahiye; automatically tie NAHI kiye):")
        say(c[["id", "persistent_jump", "g_before", "g_after"]].to_string(index=False))

    # ======== 4/5. distributions ========
    say("\n" + "=" * 70)
    say("4. UNIT-LEVEL CLASS AUR CONDITION DISTRIBUTION")
    say("=" * 70)
    gsu = gs.copy()
    cu = gsu.groupby("class_name").agg(n_groups=("group_id", "count"), n_units=("unit_id", "nunique"),
                                       frames=("n_frames", "sum")).reset_index()
    cu["support"] = np.where(cu.n_units >= 3, "full (>=3 units)",
                             np.where(cu.n_units == 2, "2 units: train+test only", "1 unit: no independent test"))
    sizes = {c: gsu[gsu.class_name == c].groupby("unit_id").n_frames.sum().sort_values().tolist()
             for c in cu.class_name}
    cu["unit_sizes"] = cu.class_name.map(lambda c: ",".join(map(str, sizes[c])))
    # subset-sum: kya koi unit-subset test (10-30% frames) bana sakta hai
    feas = []
    for c in cu.class_name:
        s = sizes[c]
        F = sum(s)
        if len(s) < 3:
            feas.append("n/a")
            continue
        if len(s) > 16:
            feas.append("skipped(n>16)")
            continue
        okk = False
        for m in range(1, 2 ** len(s) - 1):
            t = sum(s[j] for j in range(len(s)) if m >> j & 1)
            if 0.10 * F <= t <= 0.30 * F:
                okk = True
                break
        feas.append("yes" if okk else "NO")
    cu["test_10to30pct_via_whole_units"] = feas
    cu = cu.sort_values(["n_units", "frames"]).reset_index(drop=True)
    cu.to_csv(out / "class_unit_support.csv", index=False)
    say(cu.to_string(index=False))
    say(f"\nClasses by support: {cu.support.value_counts().to_dict()}")

    cd = []
    for (w, l) in CELLS:
        u = units[(units.weather == w) & (units.light == l)]
        F = int(u.frames.sum())
        if len(u) == 0:
            say(f"WARNING: cell {w}/{l} dataset mein nahi hai")
            continue
        cd.append(dict(weather=w, light=l, n_units=len(u), frames=F, pct=round(100 * F / N, 2),
                       target_test_20pct=round(0.2 * F), largest_unit=int(u.frames.max()),
                       largest_unit_pct_of_cell=round(100 * u.frames.max() / F, 1),
                       n_units_over_20pct_of_cell=int((u.frames > 0.2 * F).sum())))
    cd = pd.DataFrame(cd)
    cd.to_csv(out / "condition_unit_support.csv", index=False)
    say("\nCondition cells (units):")
    say(cd.to_string(index=False))
    top = units.sort_values("frames", ascending=False).head(8)
    say("\nSabse bade units (kisi ek split ka bada hissa ban sakte hain):")
    say(top[["unit_id", "groups", "classes", "weather", "light", "frames"]].assign(
        pct_of_dataset=(100 * top.frames / N).round(2)).to_string(index=False))

    # ======== 6. RARE CLASS FEASIBILITY ========
    say("\n" + "=" * 70)
    say("5. RARE-CLASS FEASIBILITY (classes with <=2 units)")
    say("=" * 70)
    say(f"Assumption: val={VAL_FRAC:.0%}, test={TEST_FRAC:.0%} of class frames; layout train|buffer|val|buffer|test; "
        "buffer sirf heuristic hai, leakage-free guarantee nahi.")
    rows = []
    rare = cu[cu.n_units <= 2]
    for r in rare.itertuples():
        F = int(r.frames)
        row = dict(class_name=r.class_name, n_units=r.n_units, frames=F, unit_sizes=r.unit_sizes)
        for b in BUFFERS:
            v, t = round(VAL_FRAC * F), round(TEST_FRAC * F)
            tr = F - v - t - 2 * b
            row[f"b{b}_train/val/test"] = f"{tr}/{v}/{t}" if tr > 0 else "INFEASIBLE"
        if r.n_units == 2:
            s = sizes[r.class_name]
            row["group_level_option"] = f"train={s[1]} / test={s[0]} (val nahi)"
        else:
            row["group_level_option"] = "n/a (1 unit)"
        rows.append(row)
    rf = pd.DataFrame(rows)
    rf.to_csv(out / "rare_class_feasibility.csv", index=False)
    say(rf.to_string(index=False))
    say("\nOption 1 = in classes ko sirf train mein rakho (independent test nahi). "
        "Option 2 = same-session temporal block val/test mein (optimistic flag). "
        "Upar b0/b100/b200 columns Option 2 ke frames dikhate hain.")

    # ======== reporting ========
    say("\nLimitations: groups 'candidate recording-session groups' hain, confirmed sessions nahi "
        "(timestamp/GPS nahi). Wet hamesha Dim hai. 200-frame buffer sirf heuristic hai.")
    say(f"\nKoi split file nahi banai. Saved analysis only: {out}")
    (out / "step6_plan_report.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step5", default="outputs/step5")
    ap.add_argument("--step3", default="outputs/step3")
    ap.add_argument("--eda", default="outputs/eda")
    ap.add_argument("--heur", default="outputs/step5_heur")
    ap.add_argument("--out", default="outputs/step6_plan")
    main(ap.parse_args())