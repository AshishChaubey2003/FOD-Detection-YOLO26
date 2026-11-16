"""
Step 7: FINAL SPLIT IMPLEMENTATION.

Uses the EXACT Step 6B "improved" assignment that you approved (unit -> split),
transcribed verbatim from the verified Step 6B terminal output. This script does
NOT run any optimizer and does NOT change that assignment. It only:
  1) re-derives the 116 candidate groups -> 109 tied assignment units (same
     deterministic read-only logic used and verified in Step 5 / Step 6 / Step 6B),
  2) applies the approved UNIT -> split mapping below,
  3) writes CSV manifests (no image files are copied/moved/deleted),
  4) validates the result against the mandatory approved numbers.

Run from D:\\FOD_project:
    python src\\step7_build_split.py

Reads : outputs\\step5\\group_assignment.csv, group_stats.csv, suspicious_boundaries.csv
Writes: outputs\\step6_final\\split_manifest.csv, train.csv, val.csv, test.csv, split_summary.txt

Does NOT touch outputs\\step5\\*, the raw dataset, YOLO labels, or training.
"""
import argparse
from pathlib import Path

import pandas as pd

SINGLE = ["Tape", "Screw", "Wood", "Hose", "MetalSheet", "AdjustableWrench", "BoltNutSet", "AdjustableClamp"]
EXPECTED_CELLS = {("Dry", "Bright"): 16942, ("Dry", "Dim"): 5248, ("Dry", "Dark"): 4387, ("Wet", "Dim"): 7216}
EXPECTED_TOTAL = dict(train=23688, val=3384, test=6721)
EXPECTED_UNITS = dict(train=63, val=20, test=26)
EXPECTED_CELL_SPLIT = {
    ("Dry", "Bright"): dict(train=11994, val=1694, test=3254),
    ("Dry", "Dim"): dict(train=3595, val=525, test=1128),
    ("Dry", "Dark"): dict(train=3062, val=445, test=880),
    ("Wet", "Dim"): dict(train=5037, val=720, test=1459),
}

# ---------------------------------------------------------------------------
# APPROVED ASSIGNMENT: UNIT number (1..109) -> split.
# Transcribed verbatim from your verified Step 6B terminal output, section 9
# ("IMPROVED ASSIGNMENT"). Do not edit this without re-verifying against that
# terminal output again.
# ---------------------------------------------------------------------------
_TEST = [3, 7, 15, 18, 21, 23, 31, 34, 35, 38, 42, 47, 48, 51, 54, 58, 64, 67, 72, 82, 86, 91, 96, 102, 103, 105]
_VAL = [2, 4, 10, 11, 16, 25, 36, 40, 45, 46, 57, 60, 62, 70, 77, 81, 89, 93, 97, 108]
_TRAIN = [1, 5, 6, 8, 9, 12, 13, 14, 17, 19, 20, 22, 24, 26, 27, 28, 29, 30, 32, 33, 37, 39, 41, 43, 44, 49, 50,
          52, 53, 55, 56, 59, 61, 63, 65, 66, 68, 69, 71, 73, 74, 75, 76, 78, 79, 80, 83, 84, 85, 87, 88, 90, 92,
          94, 95, 98, 99, 100, 101, 104, 106, 107, 109]
UNIT_SPLIT_NUM = {}
for n in _TEST:
    UNIT_SPLIT_NUM[n] = "test"
for n in _VAL:
    UNIT_SPLIT_NUM[n] = "val"
for n in _TRAIN:
    UNIT_SPLIT_NUM[n] = "train"
assert len(UNIT_SPLIT_NUM) == 109, f"UNIT_SPLIT_NUM has {len(UNIT_SPLIT_NUM)} entries, expected 109"
assert set(UNIT_SPLIT_NUM) == set(range(1, 110)), "UNIT_SPLIT_NUM does not cover UNIT_001..UNIT_109 exactly"
assert len(_TEST) == 26 and len(_VAL) == 20 and len(_TRAIN) == 63, "approved unit counts per split are wrong"


def say(log, m=""):
    print(m)
    log.append(str(m))


def main(a):
    s5, out = Path(a.step5), Path(a.out)
    if out.name != "step6_final":
        raise SystemExit("ERROR: output folder 'step6_final' hona chahiye.")
    out.mkdir(parents=True, exist_ok=True)
    log = []

    say(log, "=" * 74)
    say(log, "STEP 7: FINAL SPLIT -- rebuilding groups/units (read-only) + applying approved assignment")
    say(log, "=" * 74)

    asg = pd.read_csv(s5 / "group_assignment.csv", dtype={"image_id": str})
    gs = pd.read_csv(s5 / "group_stats.csv")
    sus = pd.read_csv(s5 / "suspicious_boundaries.csv")
    N = len(asg)
    if N != 33793 or not asg.image_id.is_unique or asg.group_id.nunique() != 116:
        say(log, f"[FAIL] Step 5 outputs expected shape nahi hai (rows={N}, unique groups={asg.group_id.nunique()}).")
        say(log, "STOPPING. Koi file nahi likhi. Pehle Step 5 outputs verify karein.")
        (out / "step7_ABORTED.txt").write_text("\n".join(log), encoding="utf-8")
        return

    # ---- rebuild 109 tied units (identical deterministic logic to Step 6 / Step 6B) ----
    asg["id"] = asg.image_id.astype(int)
    id2g = dict(zip(asg.id, asg.group_id))
    ties = sus[sus.type == "label:class"]
    if len(ties) != 7:
        say(log, f"[FAIL] 7 tied class boundaries expected, found {len(ties)}. STOPPING, koi file nahi likhi.")
        (out / "step7_ABORTED.txt").write_text("\n".join(log), encoding="utf-8")
        return
    parent = {g: g for g in gs.group_id}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in ties.itertuples():
        parent[find(id2g[r.id_before])] = find(id2g[r.id_after])
    gs = gs.copy()
    gs["root"] = gs.group_id.map(find)
    order = gs.groupby("root").start_id.min().sort_values().index.tolist()
    if len(order) != 109:
        say(log, f"[FAIL] 109 units expected, {len(order)} bane. STOPPING, koi file nahi likhi.")
        (out / "step7_ABORTED.txt").write_text("\n".join(log), encoding="utf-8")
        return
    uid = {r: i + 1 for i, r in enumerate(order)}   # unit number 1..109, same ordering as Step 6/6B
    gs["unit_num"] = gs.root.map(uid)
    gs["unit_id"] = gs.unit_num.map(lambda n: f"UNIT_{n:03d}")

    def unit_of(gids):
        return {gs.loc[gs.group_id == g, "unit_id"].iloc[0] for g in gids}
    u19 = unit_of([f"GROUP_{i:03d}" for i in range(19, 25)])
    u106 = unit_of(["GROUP_112", "GROUP_113"])
    if len(u19) != 1 or len(u106) != 1:
        say(log, "[FAIL] UNIT_019 (groups 019-024) ya UNIT_106 (112+113) ki composition match nahi hui. "
                 "STOPPING, koi file nahi likhi.")
        (out / "step7_ABORTED.txt").write_text("\n".join(log), encoding="utf-8")
        return
    say(log, "[PASS] 116 groups -> 109 units rebuilt, 7 ties applied, UNIT_019 aur UNIT_106 composition verified.")

    # ---- apply approved unit -> split mapping ----
    gs["split"] = gs.unit_num.map(UNIT_SPLIT_NUM)
    if gs.split.isna().any():
        say(log, "[FAIL] Kuch units ko approved mapping mein split nahi mila. STOPPING.")
        (out / "step7_ABORTED.txt").write_text("\n".join(log), encoding="utf-8")
        return
    g2split = dict(zip(gs.group_id, gs.split))
    g2unit = dict(zip(gs.group_id, gs.unit_id))
    asg["assignment_unit"] = asg.group_id.map(g2unit)
    asg["split"] = asg.group_id.map(g2split)
    asg = asg.rename(columns={"group_id": "assignment_group"})

    manifest_cols = ["image_id", "image_path", "class_name", "assignment_group", "assignment_unit",
                      "weather", "light", "split"]
    manifest = asg[manifest_cols].sort_values("image_id").reset_index(drop=True)

    # ---- write files ----
    say(log, "\n" + "=" * 74)
    say(log, "FILES WRITTEN")
    say(log, "=" * 74)
    manifest.to_csv(out / "split_manifest.csv", index=False)
    say(log, f"{out / 'split_manifest.csv'}  ({len(manifest)} rows)")
    for s in ("train", "val", "test"):
        d = manifest[manifest.split == s]
        d.to_csv(out / f"{s}.csv", index=False)
        say(log, f"{out / (s + '.csv')}  ({len(d)} rows)")

    # ---- mandatory validation ----
    say(log, "\n" + "=" * 74)
    say(log, "MANDATORY VALIDATION (against generated CSVs)")
    say(log, "=" * 74)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        say(log, f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    tot = {s: len(manifest[manifest.split == s]) for s in ("train", "val", "test")}
    chk("Total images = 33,793", len(manifest) == 33793, f"(found {len(manifest)})")
    chk("Train = 23,688", tot["train"] == EXPECTED_TOTAL["train"], f"(found {tot['train']})")
    chk("Validation = 3,384", tot["val"] == EXPECTED_TOTAL["val"], f"(found {tot['val']})")
    chk("Test = 6,721", tot["test"] == EXPECTED_TOTAL["test"], f"(found {tot['test']})")
    uc = {s: gs[gs.split == s].unit_id.nunique() for s in ("train", "val", "test")}
    chk("Unit counts = 63/20/26", uc == EXPECTED_UNITS, f"(found {uc})")
    chk("No duplicate image IDs", manifest.image_id.is_unique)
    chk("No missing image IDs (manifest covers all Step 5 ids)",
        set(manifest.image_id) == set(asg.image_id))
    chk("No image in multiple splits",
        manifest.groupby("image_id").split.nunique().max() == 1)
    chk("Every image assigned exactly once", len(manifest) == len(asg) and manifest.split.notna().all())
    unit_splits = manifest.groupby("assignment_unit").split.nunique()
    chk("No assignment unit divided across splits", (unit_splits == 1).all(),
        f"(units with >1 split: {int((unit_splits > 1).sum())})")
    chk("UNIT_019 in Train", gs.loc[gs.unit_id == next(iter(u19)), "split"].iloc[0] == "train")
    chk("UNIT_106 in Train", gs.loc[gs.unit_id == next(iter(u106)), "split"].iloc[0] == "train")

    single_ok = True
    for c in SINGLE:
        sp = manifest.loc[manifest.class_name == c, "split"].unique()
        single_ok &= list(sp) == ["train"]
    chk("8 single-unit classes are Train-only", single_ok)

    cells = manifest.groupby(["weather", "light"]).size().to_dict()
    chk("4 cells match Step 5 (16942/5248/4387/7216)", cells == EXPECTED_CELLS, f"{cells}")
    chk("Test contains all four condition cells",
        all(((manifest.weather == w) & (manifest.light == l) & (manifest.split == "test")).sum() > 0
            for (w, l) in EXPECTED_CELLS))

    cell_rows = []
    cell_ok = True
    for (w, l), exp in EXPECTED_CELL_SPLIT.items():
        d = manifest[(manifest.weather == w) & (manifest.light == l)]
        got = {s: int((d.split == s).sum()) for s in ("train", "val", "test")}
        cell_ok &= got == exp
        cell_rows.append((f"{w}/{l}", got, exp))
    chk("All condition-wise counts match approved table", cell_ok)
    for name, got, exp in cell_rows:
        say(log, f"    {name}: got {got}  expected {exp}")

    # class-wise policy checks (derived from actual unit composition, same as Step 6B)
    classes = {}
    for r in gs.itertuples():
        classes.setdefault(r.class_name, {})
        classes[r.class_name][r.unit_id] = classes[r.class_name].get(r.unit_id, 0) + r.n_frames
    pol_ok = True
    pol_detail = []
    for c, d in classes.items():
        k = len(d)
        sp = [gs.loc[gs.unit_id == u, "split"].iloc[0] for u in d]
        cnt = {s: sp.count(s) for s in ("train", "val", "test")}
        if c in SINGLE:
            good = cnt == {"train": 1, "val": 0, "test": 0}
        elif k == 3:
            good = cnt == {"train": 2, "val": 0, "test": 1}
        elif c == "Nail":
            good = cnt == {"train": 1, "val": 0, "test": 1}
        elif k >= 4:
            good = cnt["train"] >= 1 and cnt["val"] >= 1 and cnt["test"] >= 1
        else:
            good = True
        pol_ok &= good
        if not good:
            pol_detail.append(f"{c} (units={k}): {cnt}")
    chk("Class-wise split policies preserved (single-unit, 3-unit, Nail, >=4-unit)", pol_ok)
    if pol_detail:
        for d in pol_detail:
            say(log, f"    MISMATCH: {d}")

    say(log, "\nOVERALL: " + ("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED -- see [FAIL] lines above"))
    if not ok:
        say(log, "Files outputs\\step6_final\\*.csv upar ke mismatch ke saath bane hain. "
                 "Inhe YOLO ya training ke liye USE NA KAREIN jab tak yeh resolve na ho jaaye.")

    # ---- summary report ----
    say(log, "\n" + "=" * 74)
    say(log, "CLASS-WISE COVERAGE SUMMARY")
    say(log, "=" * 74)
    crows = []
    for c, d in sorted(classes.items(), key=lambda kv: (len(kv[1]), sum(kv[1].values()))):
        tot_c = sum(d.values())
        im = {s: 0 for s in ("train", "val", "test")}
        for u, f in d.items():
            im[gs.loc[gs.unit_id == u, "split"].iloc[0]] += f
        crows.append(dict(Class=c, Units=len(d), Images=tot_c, TrainImg=im["train"], ValImg=im["val"],
                          TestImg=im["test"]))
    cdf = pd.DataFrame(crows)
    say(log, cdf.to_string(index=False))

    (out / "split_summary.txt").write_text("\n".join(log), encoding="utf-8")
    say(log, f"\nsplit_summary.txt bhi saved.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step5", default="outputs/step5")
    ap.add_argument("--out", default="outputs/step6_final")
    main(ap.parse_args())