"""
Step 6 GLOBAL FEASIBILITY (analysis only). READ-ONLY + PRINT-ONLY.
Koi file save NAHI hoti. Koi split CSV, folder, copy/move, YOLO labels NAHI banta.
Sab kuch sirf terminal mein print hota hai.

Run from D:\\FOD_project  (scipy chahiye:  pip install scipy):
    python src\\step6_feasibility.py

Reads: outputs\\step5\\group_assignment.csv, group_stats.csv, suspicious_boundaries.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
except Exception:
    raise SystemExit("ERROR: scipy nahi mila (ya purana hai). Chalayein:  pip install scipy")

S = ["Train", "Val", "Test"]
CELLS = [("Dry", "Bright"), ("Dry", "Dim"), ("Dry", "Dark"), ("Wet", "Dim")]
SINGLE = ["Tape", "Screw", "Wood", "Hose", "MetalSheet", "AdjustableWrench", "BoltNutSet", "AdjustableClamp"]
TARGET = dict(Train=0.70, Val=0.10, Test=0.20)
EXPECTED_CELLS = {("Dry", "Bright"): 16942, ("Dry", "Dim"): 5248, ("Dry", "Dark"): 4387, ("Wet", "Dim"): 7216}


def hdr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


class Model:
    """Binary x[u,s] + deviation variables; constraints as rows."""

    def __init__(self, nU):
        self.nU = nU
        self.nv = nU * 3
        self.rows, self.lo, self.hi = [], [], []
        self.obj = {}
        self.fixed = {}

    def xi(self, u, s):
        return u * 3 + s

    def new_var(self):
        self.nv += 1
        return self.nv - 1

    def add(self, coefs, lo, hi):
        self.rows.append(coefs)
        self.lo.append(lo)
        self.hi.append(hi)

    def dev(self, coefs, target, weight):
        dp, dm = self.new_var(), self.new_var()
        c = dict(coefs)
        c[dp] = -1.0
        c[dm] = 1.0
        self.add(c, target, target)
        self.obj[dp] = self.obj.get(dp, 0) + weight
        self.obj[dm] = self.obj.get(dm, 0) + weight

    def solve(self, extra_obj=None, time_limit=60):
        n = self.nv
        A = np.zeros((len(self.rows), n))
        for i, r in enumerate(self.rows):
            for j, v in r.items():
                A[i, j] = v
        lb = np.zeros(n)
        ub = np.ones(n)
        ub[self.nU * 3:] = np.inf
        for (u, s), val in self.fixed.items():
            lb[self.xi(u, s)] = ub[self.xi(u, s)] = val
        integ = np.zeros(n)
        integ[:self.nU * 3] = 1
        c = np.zeros(n)
        for j, v in (extra_obj if extra_obj is not None else self.obj).items():
            c[j] += v
        res = milp(c, constraints=LinearConstraint(A, self.lo, self.hi) if len(self.rows) else None,
                   integrality=integ, bounds=Bounds(lb, ub), options={"time_limit": time_limit})
        return res


def build(units, classes, families, fixed_train, cell_idx, N, weights):
    """units: DataFrame; classes: dict class -> {unit_index: class_frames}"""
    nU = len(units)
    m = Model(nU)
    fr = units.frames.values.astype(float)
    for u in range(nU):
        m.add({m.xi(u, s): 1.0 for s in range(3)}, 1, 1)
    if "fixed" in families:
        for u in fixed_train:
            m.fixed[(u, 0)] = 1
    for c, d in classes.items():
        us = list(d)
        k = len(us)
        if k == 1 and c in SINGLE and "single" in families:
            m.add({m.xi(us[0], 0): 1.0}, 1, 1)
        elif k == 3 and "three" in families:
            m.add({m.xi(u, 2): 1.0 for u in us}, 1, 1)
            m.add({m.xi(u, 1): 1.0 for u in us}, 0, 0)
        elif k == 2 and c == "Nail" and "nail" in families:
            m.add({m.xi(u, 2): 1.0 for u in us}, 1, 1)
            m.add({m.xi(u, 1): 1.0 for u in us}, 0, 0)
        elif k >= 4 and "val_cov" in families:
            for s in range(3):
                m.add({m.xi(u, s): 1.0 for u in us}, 1, k)
    if "cell_cov" in families:
        for ci in range(4):
            m.add({m.xi(u, 2): 1.0 for u in cell_idx[ci]}, 1, len(cell_idx[ci]))
    # soft objective (priority 5: overall proportions), plus secondary balance terms
    for s, name in enumerate(S):
        m.dev({m.xi(u, s): fr[u] for u in range(nU)}, TARGET[name] * N, weights["split"] / N)
    for ci in range(4):
        tot = fr[cell_idx[ci]].sum()
        m.dev({m.xi(u, 2): fr[u] for u in cell_idx[ci]}, 0.2 * tot, weights["cell"] / N)
    for c, d in classes.items():
        if len(d) >= 2:
            F = sum(d.values())
            m.dev({m.xi(u, 2): float(f) for u, f in d.items()}, 0.2 * F, weights["cls"] / F)
    return m


def status(res):
    return {0: "OPTIMAL", 1: "TIME/ITER LIMIT", 2: "INFEASIBLE", 3: "UNBOUNDED"}.get(res.status, str(res.status))


def main(a):
    s5 = Path(a.step5)
    hdr("1. DATA VERIFICATION (verified facts)")
    asg = pd.read_csv(s5 / "group_assignment.csv", dtype={"image_id": str})
    gs = pd.read_csv(s5 / "group_stats.csv")
    sus = pd.read_csv(s5 / "suspicious_boundaries.csv")
    N = len(asg)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    chk("33,793 images, ids unique", N == 33793 and asg.image_id.is_unique, f"(rows={N})")
    chk("116 groups", asg.group_id.nunique() == 116 and len(gs) == 116)
    chk("group frames sum == images", gs.n_frames.sum() == N)
    chk("all groups single-class", (gs.n_main_classes == 1).all())
    chk("no mixed-condition group", not (gs.weather.str.contains("/") | gs.light.str.contains("/")).any())
    cells = asg.groupby(["weather", "light"]).size().to_dict()
    chk("4 cells match Step 5", cells == EXPECTED_CELLS, f"{cells}")
    if not ok:
        print("\nDATA INCONSISTENT -> analysis ruk raha hai (kuch guess nahi karunga).")
        return

    # tied units (same logic as step6_plan_analysis)
    asg["id"] = asg.image_id.astype(int)
    id2g = dict(zip(asg.id, asg.group_id))
    ties = sus[sus.type == "label:class"]
    chk("7 tied class boundaries", len(ties) == 7, f"(found {len(ties)})")
    parent = {g: g for g in gs.group_id}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in ties.itertuples():
        ga, gb = id2g[r.id_before], id2g[r.id_after]
        parent[find(ga)] = find(gb)
    gs["root"] = gs.group_id.map(find)
    order = gs.groupby("root").start_id.min().sort_values().index.tolist()
    uid = {r: f"UNIT_{i + 1:03d}" for i, r in enumerate(order)}
    gs["unit_id"] = gs.root.map(uid)
    chk("109 assignment units", gs.unit_id.nunique() == 109, f"(found {gs.unit_id.nunique()})")
    def unit_of(gids):
        us = {gs.loc[gs.group_id == g, "unit_id"].iloc[0] for g in gids}
        return us
    u19 = unit_of([f"GROUP_{i:03d}" for i in range(19, 25)])
    u106 = unit_of(["GROUP_112", "GROUP_113"])
    chk("GROUP_019..024 ek hi unit mein", len(u19) == 1, f"({u19})")
    chk("GROUP_112+113 ek hi unit mein", len(u106) == 1, f"({u106})")
    if not ok:
        print("\nDecision A ke units match nahi hue -> ruk raha hoon.")
        return
    UN19, UN106 = next(iter(u19)), next(iter(u106))
    print(f"Decision A units: {UN19} (Train), {UN106} (Train)")

    units = gs.groupby("unit_id").agg(frames=("n_frames", "sum"), groups=("group_id", lambda x: "+".join(x)),
                                      classes=("class_name", lambda x: "+".join(sorted(set(x)))),
                                      weather=("weather", "first"), light=("light", "first")).reset_index()
    units = units.sort_values("unit_id").reset_index(drop=True)
    uix = {u: i for i, u in enumerate(units.unit_id)}
    classes = {}
    for r in gs.itertuples():
        classes.setdefault(r.class_name, {})
        classes[r.class_name][uix[r.unit_id]] = classes[r.class_name].get(uix[r.unit_id], 0) + r.n_frames
    cell_idx = []
    for (w, l) in CELLS:
        cell_idx.append(np.where((units.weather == w) & (units.light == l))[0])
    fixed_train = [uix[UN19], uix[UN106]]
    nU = len(units)
    fr = units.frames.values

    # support tiers
    tier = {c: len(d) for c, d in classes.items()}
    print(f"Single-unit classes (data se): {sorted([c for c, k in tier.items() if k == 1])}")
    chk("8 single-unit classes == approved list",
        sorted([c for c, k in tier.items() if k == 1]) == sorted(SINGLE))
    if not ok:
        return

    weights = dict(split=10.0, cell=1.0, cls=0.2)
    all_fam = {"fixed", "single", "three", "nail", "val_cov", "cell_cov"}

    # ============ forced deductions ============
    hdr("2. FORCED CONSEQUENCES (hard constraints se seedha nikalta hai)")
    nail_units = list(classes["Nail"])
    print("Nail units:", [(units.unit_id[u], classes['Nail'][u], units.classes[u]) for u in nail_units])
    forced_test = [u for u in nail_units if u not in fixed_train]
    for u in forced_test:
        print(f"-> Nail ko 1 Train + 1 Test chahiye; {units.unit_id[u]} Train mein nahi ja sakta -> Test mein jayega "
              f"({units.classes[u]}, {int(fr[u])} frames)")

    # ============ feasibility ladder ============
    hdr("3. GLOBAL FEASIBILITY")
    zero = {}
    m = build(units, classes, all_fam, fixed_train, cell_idx, N, weights)
    r = m.solve(extra_obj=zero)
    feasible = r.status in (0, 1) and r.x is not None
    print(f"All constraints together (feasibility only): {status(r)}")
    if not feasible:
        print("\nConflict diagnosis: ek-ek constraint family hata kar dekhte hain:")
        for fam in sorted(all_fam):
            mm = build(units, classes, all_fam - {fam}, fixed_train, cell_idx, N, weights)
            rr = mm.solve(extra_obj=zero)
            print(f"   {fam:9s} hata do -> {status(rr)}")
        print("\nKoi full assignment nahi bani; proposal skip.")
        return

    # achievable ranges
    print("\nAchievable ranges (hard constraints ke andar, exact optimisation):")
    for s, name in ((1, "Val"), (2, "Test")):
        for sign, lab in ((1, "min"), (-1, "max")):
            obj = {m.xi(u, s): sign * float(fr[u]) for u in range(nU)}
            rr = m.solve(extra_obj=obj)
            val = sum(fr[u] * rr.x[m.xi(u, s)] for u in range(nU))
            print(f"   {name} {lab}: {val:.0f} images ({100 * val / N:.1f}%) [{status(rr)}]")

    # best proposal
    res = m.solve()
    print(f"\nBest proposal solve (objective = overall 70/10/20 + cell/class balance): {status(res)}")
    X = np.array([[round(res.x[m.xi(u, s)]) for s in range(3)] for u in range(nU)])
    units["split"] = [S[int(np.argmax(X[u]))] for u in range(nU)]

    # ============ validation of proposal ============
    hdr("4. PROPOSAL KI VALIDATION (constraints sach mein pure huye?)")
    chk("har unit exactly ek split", (X.sum(1) == 1).all())
    chk("UNIT fixed Train", units.split[uix[UN19]] == "Train" and units.split[uix[UN106]] == "Train")
    chk("single-unit classes sab Train",
        all(units.split[list(classes[c])[0]] == "Train" for c in SINGLE))
    pol_ok = True
    rows = []
    for c, d in sorted(classes.items(), key=lambda kv: (len(kv[1]), sum(kv[1].values()))):
        cnt = {s: 0 for s in S}
        im = {s: 0 for s in S}
        for u, f in d.items():
            cnt[units.split[u]] += 1
            im[units.split[u]] += f
        k = len(d)
        tot = sum(d.values())
        if k == 1:
            lim = "Train only: independent val/test nahi"
            pol_ok &= cnt["Train"] == 1
        elif k == 2:
            lim = "1 Train + 1 Test; val nahi"
            pol_ok &= (cnt["Train"], cnt["Test"], cnt["Val"]) == (1, 1, 0)
        elif k == 3:
            lim = "2 Train + 1 Test; val nahi"
            pol_ok &= (cnt["Train"], cnt["Test"], cnt["Val"]) == (2, 1, 0)
        else:
            full = cnt["Train"] >= 1 and cnt["Val"] >= 1 and cnt["Test"] >= 1
            lim = "full T/V/E coverage" if full else "COVERAGE ADHURA"
            pol_ok &= full
        rows.append(dict(Class=c, Units=k, Images=tot, TrainU=cnt["Train"], ValU=cnt["Val"], TestU=cnt["Test"],
                         TrainImg=im["Train"], ValImg=im["Val"], TestImg=im["Test"],
                         Test_pct=round(100 * im["Test"] / tot, 1), Note=lim))
    chk("class-wise policies (Decision B & C) pure", pol_ok)
    ct = pd.DataFrame(rows)
    cellrows = []
    for (w, l), idx in zip(CELLS, cell_idx):
        d = units.iloc[idx]
        tot = int(d.frames.sum())
        tr, va, te = (int(d[d.split == s].frames.sum()) for s in S)
        cellrows.append(dict(Cell=f"{w}/{l}", Total=tot, Train=tr, Val=va, Test=te, Test_pct=round(100 * te / tot, 1),
                             Val_pct=round(100 * va / tot, 1), TestUnits=int((d.split == "Test").sum())))
    cd = pd.DataFrame(cellrows)
    chk("4 cells Test mein represented", (cd.Test > 0).all())

    hdr("5. CLASS-WISE FEASIBILITY TABLE")
    print(ct.to_string(index=False))
    no_val = ct[ct.ValU == 0].Class.tolist()
    no_test = ct[ct.TestU == 0].Class.tolist()
    print(f"\nIndependent TEST nahi: {no_test}")
    print(f"Independent VAL nahi : {no_val}")

    hdr("6. CONDITION-WISE FEASIBILITY TABLE")
    print(cd.to_string(index=False))
    t19 = fr[uix[UN19]]
    dbt = cd.loc[cd.Cell == "Dry/Bright"].iloc[0]
    print(f"\nUNIT_019 = {int(t19)} images = {100 * t19 / dbt.Total:.1f}% of Dry/Bright, "
          f"{100 * t19 / dbt.Train:.1f}% of Dry/Bright Train")
    print("Dry/Bright ka Train share:", f"{100 * dbt.Train / dbt.Total:.1f}%",
          "| baaki cells Train share:",
          {r.Cell: f"{100 * r.Train / r.Total:.1f}%" for r in cd.itertuples() if r.Cell != "Dry/Bright"})

    hdr("7. OVERALL IMAGE DISTRIBUTION (actual unit sizes se)")
    tot = {s: int(units[units.split == s].frames.sum()) for s in S}
    ov = pd.DataFrame([dict(Split=s, Images=tot[s], Pct=round(100 * tot[s] / N, 2), Target=f"{100 * TARGET[s]:.0f}%",
                            Units=int((units.split == s).sum())) for s in S])
    print(ov.to_string(index=False))
    chk("total == 33,793", sum(tot.values()) == N)

    hdr("8. PROPOSED ASSIGNMENT (sirf review ke liye; koi file save nahi)")
    pv = units[["unit_id", "groups", "classes", "weather", "light", "frames", "split"]].copy()
    pv["cell"] = pv.weather + "/" + pv.light
    for s in ("Test", "Val", "Train"):
        d = pv[pv.split == s]
        print(f"\n--- {s}: {len(d)} units, {int(d.frames.sum())} images ---")
        print(d[["unit_id", "groups", "classes", "cell", "frames"]].to_string(index=False))
    print("\nKoi file save nahi hui. Implementation sirf aapki explicit approval ke baad.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step5", default="outputs/step5")
    main(ap.parse_args())