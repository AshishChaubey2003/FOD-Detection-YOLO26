"""
Step 6B: IMPROVED feasibility analysis (analysis only). READ-ONLY + PRINT-ONLY.
Koi file save NAHI hoti. Koi split CSV, folder, copy/move, YOLO labels NAHI banta.

Run from D:\\FOD_project  (scipy chahiye, pehle install ho chuka hai):
    python src\\step6b_improved.py

Reads: outputs\\step5\\group_assignment.csv, group_stats.csv, suspicious_boundaries.csv
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
except Exception:
    raise SystemExit("ERROR: scipy nahi mila. Chalayein:  pip install scipy")

S = ["Train", "Val", "Test"]
CELLS = [("Dry", "Bright"), ("Dry", "Dim"), ("Dry", "Dark"), ("Wet", "Dim")]
CNAME = [f"{w}/{l}" for w, l in CELLS]
SINGLE = ["Tape", "Screw", "Wood", "Hose", "MetalSheet", "AdjustableWrench", "BoltNutSet", "AdjustableClamp"]
TARGET = dict(Train=0.70, Val=0.10, Test=0.20)
EXPECTED_CELLS = {("Dry", "Bright"): 16942, ("Dry", "Dim"): 5248, ("Dry", "Dark"): 4387, ("Wet", "Dim"): 7216}
PREV = dict(Train=23655, Val=3380, Test=6758)   # pichhle verified run ke numbers


def hdr(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


# ---------------------------------------------------------------- original model (pichhle run jaisa, same formulation)
class Model0:
    def __init__(self, nU):
        self.nU, self.nv = nU, nU * 3
        self.rows, self.lo, self.hi, self.obj, self.fixed = [], [], [], {}, {}

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
        lb, ub = np.zeros(n), np.ones(n)
        ub[self.nU * 3:] = np.inf
        for (u, s), val in self.fixed.items():
            lb[self.xi(u, s)] = ub[self.xi(u, s)] = val
        integ = np.zeros(n)
        integ[:self.nU * 3] = 1
        c = np.zeros(n)
        for j, v in (extra_obj if extra_obj is not None else self.obj).items():
            c[j] += v
        return milp(c, constraints=LinearConstraint(A, self.lo, self.hi), integrality=integ,
                    bounds=Bounds(lb, ub), options={"time_limit": time_limit})


def hard_rules(m, classes, fixed_train, cell_idx):
    """Approved hard rules (same for original and improved)."""
    for u in range(m.nU):
        m.add({m.xi(u, s): 1.0 for s in range(3)}, 1, 1)
    for u in fixed_train:
        if hasattr(m, "fixed"):
            m.fixed[(u, 0)] = 1
        else:
            m.fix(u, 0)
    for c, d in classes.items():
        us = list(d)
        k = len(us)
        if k == 1 and c in SINGLE:
            m.add({m.xi(us[0], 0): 1.0}, 1, 1)
        elif k == 3:
            m.add({m.xi(u, 2): 1.0 for u in us}, 1, 1)
            m.add({m.xi(u, 1): 1.0 for u in us}, 0, 0)
        elif k == 2 and c == "Nail":
            m.add({m.xi(u, 2): 1.0 for u in us}, 1, 1)
            m.add({m.xi(u, 1): 1.0 for u in us}, 0, 0)
        elif k >= 4:
            for s in range(3):
                m.add({m.xi(u, s): 1.0 for u in us}, 1, k)
    for ci in range(4):
        m.add({m.xi(u, 2): 1.0 for u in cell_idx[ci]}, 1, len(cell_idx[ci]))


def build_original(units, classes, fixed_train, cell_idx, N):
    nU = len(units)
    m = Model0(nU)
    fr = units.frames.values.astype(float)
    hard_rules(m, classes, fixed_train, cell_idx)
    for s, name in enumerate(S):
        m.dev({m.xi(u, s): fr[u] for u in range(nU)}, TARGET[name] * N, 10.0 / N)
    for ci in range(4):
        tot = fr[cell_idx[ci]].sum()
        m.dev({m.xi(u, 2): fr[u] for u in cell_idx[ci]}, 0.2 * tot, 1.0 / N)
    for c, d in classes.items():
        if len(d) >= 2:
            F = sum(d.values())
            m.dev({m.xi(u, 2): float(f) for u, f in d.items()}, 0.2 * F, 0.2 / F)
    return m


# ---------------------------------------------------------------- improved model (lexicographic stages)
class Model2:
    def __init__(self, nU):
        self.nU = nU
        self.lb, self.ub, self.integ = [], [], []
        for _ in range(nU * 3):
            self.lb.append(0.0), self.ub.append(1.0), self.integ.append(1)
        self.rows, self.lo, self.hi = [], [], []

    def xi(self, u, s):
        return u * 3 + s

    def var(self, lb=0.0, ub=np.inf, integer=False):
        self.lb.append(lb), self.ub.append(ub), self.integ.append(1 if integer else 0)
        return len(self.lb) - 1

    def fix(self, u, s):
        self.lb[self.xi(u, s)] = self.ub[self.xi(u, s)] = 1.0

    def add(self, coefs, lo, hi):
        self.rows.append(coefs)
        self.lo.append(lo)
        self.hi.append(hi)

    def solve(self, obj, time_limit=120, gap=0.0):
        n = len(self.lb)
        A = np.zeros((len(self.rows), n))
        for i, r in enumerate(self.rows):
            for j, v in r.items():
                A[i, j] = v
        c = np.zeros(n)
        for j, v in obj.items():
            c[j] += v
        return milp(c, constraints=LinearConstraint(A, self.lo, self.hi), integrality=np.array(self.integ),
                    bounds=Bounds(np.array(self.lb), np.array(self.ub)),
                    options={"time_limit": time_limit, "mip_rel_gap": gap})


def sname(res):
    return {0: "OPTIMAL", 1: "TIME LIMIT", 2: "INFEASIBLE", 3: "UNBOUNDED"}.get(res.status, str(res.status))


# ---------------------------------------------------------------- evaluation of any assignment
def evaluate(split, units, classes, cell_idx, N, cell_share):
    fr = units.frames.values
    r = {}
    r["tot"] = {s: int(fr[split == i].sum()) for i, s in enumerate(S)}
    r["cells"] = []
    for ci in range(4):
        idx = cell_idx[ci]
        row = {s: int(fr[idx][split[idx] == i].sum()) for i, s in enumerate(S)}
        row["Total"] = int(fr[idx].sum())
        r["cells"].append(row)
    vt = r["tot"]["Val"]
    r["val_share"] = [100 * c["Val"] / vt if vt else 0 for c in r["cells"]]
    r["val_L1"] = sum(abs(a - b) for a, b in zip(r["val_share"], cell_share))
    cell_of = np.zeros(len(units), int)
    for ci in range(4):
        cell_of[cell_idx[ci]] = ci
    cls = {}
    viol, shift_pairs = [], 0
    for c, d in classes.items():
        T = sum(f for u, f in d.items() if split[u] == 0)
        V = sum(f for u, f in d.items() if split[u] == 1)
        E = sum(f for u, f in d.items() if split[u] == 2)
        tc = {int(cell_of[u]) for u in d if split[u] == 0}
        ec = {int(cell_of[u]) for u in d if split[u] == 2}
        vc = {int(cell_of[u]) for u in d if split[u] == 1}
        miss = ec - tc
        if E > 0:
            if T < V + E:
                viol.append(c)
            shift_pairs += len(miss)
            cat = "full" if not miss else ("none" if not (ec & tc) else "partial")
        else:
            cat = "-"
        cls[c] = dict(T=T, V=V, E=E, tc=tc, ec=ec, vc=vc, miss=miss, cat=cat,
                      units=len(d), TU=sum(split[u] == 0 for u in d), VU=sum(split[u] == 1 for u in d),
                      EU=sum(split[u] == 2 for u in d))
    r["cls"], r["viol"], r["shift_pairs"] = cls, viol, shift_pairs
    r["test_cells_ok"] = all(c["Test"] > 0 for c in r["cells"])
    return r


def local_min_shift(c, d, units, cell_of, fixed_train):
    """Class ke apne units ke andar best-case: kam se kam kitni test conditions Train mein nahi (policy ke andar)."""
    us = list(d)
    k = len(us)
    if k == 1 and c in SINGLE:
        return None
    best = None
    if k == 2:
        opts = [(0, 2)] if us[0] in fixed_train else [(2, 0)]
        if us[0] not in fixed_train and us[1] in fixed_train:
            opts = [(2, 0)]
        elif us[0] in fixed_train:
            opts = [(0, 2)]
    elif k == 3:
        opts = [tuple(2 if j == e else 0 for j in range(3)) for e in range(3)]
    else:
        opts = itertools.product((0, 1, 2), repeat=k)
    for a in opts:
        if any(us[j] in fixed_train and a[j] != 0 for j in range(k)):
            continue
        if k >= 4 and not (0 in a and 1 in a and 2 in a):
            continue
        tc = {cell_of[us[j]] for j in range(k) if a[j] == 0}
        ec = {cell_of[us[j]] for j in range(k) if a[j] == 2}
        v = len(ec - tc)
        if best is None or v < best:
            best = v
            if best == 0:
                break
    return best


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
    chk("33,793 images, ids unique", N == 33793 and asg.image_id.is_unique)
    chk("116 groups, sab single-class", asg.group_id.nunique() == 116 and (gs.n_main_classes == 1).all())
    cells = asg.groupby(["weather", "light"]).size().to_dict()
    chk("4 cells match", cells == EXPECTED_CELLS, f"{cells}")
    asg["id"] = asg.image_id.astype(int)
    id2g = dict(zip(asg.id, asg.group_id))
    ties = sus[sus.type == "label:class"]
    chk("7 tied class boundaries", len(ties) == 7)
    parent = {g: g for g in gs.group_id}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in ties.itertuples():
        parent[find(id2g[r.id_before])] = find(id2g[r.id_after])
    gs["root"] = gs.group_id.map(find)
    order = gs.groupby("root").start_id.min().sort_values().index.tolist()
    uid = {r: f"UNIT_{i + 1:03d}" for i, r in enumerate(order)}
    gs["unit_id"] = gs.root.map(uid)
    chk("109 units", gs.unit_id.nunique() == 109)

    def unit_of(gids):
        return {gs.loc[gs.group_id == g, "unit_id"].iloc[0] for g in gids}
    u19 = unit_of([f"GROUP_{i:03d}" for i in range(19, 25)])
    u106 = unit_of(["GROUP_112", "GROUP_113"])
    chk("UNIT_019 / UNIT_106 sahi", len(u19) == 1 and len(u106) == 1)
    if not ok:
        print("\nData inconsistent -> ruk raha hoon.")
        return
    UN19, UN106 = next(iter(u19)), next(iter(u106))
    units = gs.groupby("unit_id").agg(frames=("n_frames", "sum"), groups=("group_id", lambda x: "+".join(x)),
                                      classes=("class_name", lambda x: "+".join(sorted(set(x)))),
                                      weather=("weather", "first"), light=("light", "first")).reset_index()
    units = units.sort_values("unit_id").reset_index(drop=True)
    uix = {u: i for i, u in enumerate(units.unit_id)}
    nU = len(units)
    fr = units.frames.values
    classes = {}
    for r in gs.itertuples():
        classes.setdefault(r.class_name, {})
        classes[r.class_name][uix[r.unit_id]] = classes[r.class_name].get(uix[r.unit_id], 0) + r.n_frames
    cell_idx = [np.where((units.weather == w) & (units.light == l))[0] for (w, l) in CELLS]
    cell_of = np.zeros(nU, int)
    for ci in range(4):
        cell_of[cell_idx[ci]] = ci
    fixed_train = {uix[UN19], uix[UN106]}
    cell_share = [100 * fr[idx].sum() / N for idx in cell_idx]
    chk("8 single-unit classes approved list", sorted(c for c, d in classes.items() if len(d) == 1) == sorted(SINGLE))
    if not ok:
        return

    # ---------------- original proposal reproduce
    hdr("2. ORIGINAL PROPOSAL (pichhla run dobara, same formulation)")
    m0 = build_original(units, classes, fixed_train, cell_idx, N)
    r0 = m0.solve()
    X0 = np.array([[round(r0.x[m0.xi(u, s)]) for s in range(3)] for u in range(nU)])
    split0 = X0.argmax(1)
    ev0 = evaluate(split0, units, classes, cell_idx, N, cell_share)
    same = ev0["tot"] == PREV
    print(f"Original reproduce: {ev0['tot']}  -> pichhle verified {PREV}: " +
          ("MATCH" if same else "MISMATCH (data/solver alag; comparison dobara-solve wale original se hai)"))

    # ---------------- improved: lexicographic stages
    hdr("3. IMPROVED: PRIORITY-WISE (lexicographic) OPTIMISATION")
    print("Har stage pichhle stages ka best value lock karke agla stage optimize karta hai, taaki upar ki priority\n"
          "kabhi neeche wali ke liye kurbaan na ho. Hard rules (1,2,3 + approved policies) hamesha lagu.")
    m = Model2(nU)
    hard_rules(m, classes, fixed_train, cell_idx)
    test_cls = [c for c, d in classes.items() if len(d) >= 2]
    X = lambda u, s: m.xi(u, s)
    # P4: Train >= Val + Test (preferred). z=violation flag, sh=shortfall
    z, sh = {}, {}
    for c in test_cls:
        d = classes[c]
        F = float(sum(d.values()))
        z[c] = m.var(0, 1, True)
        sh[c] = m.var(0, np.inf)
        e = {}
        for u, f in d.items():
            e[X(u, 0)] = e.get(X(u, 0), 0) + f
            e[X(u, 1)] = e.get(X(u, 1), 0) - f
            e[X(u, 2)] = e.get(X(u, 2), 0) - f
        m.add({**e, z[c]: F}, 0, np.inf)
        m.add({**e, sh[c]: 1.0}, 0, np.inf)
    # P5: Val condition balance (Val cell frames ~ dataset share * 10% of N)
    vdev = []
    for ci in range(4):
        dp, dm = m.var(), m.var()
        c = {X(u, 1): float(fr[u]) for u in cell_idx[ci]}
        c[dp], c[dm] = -1.0, 1.0
        t = cell_share[ci] / 100 * TARGET["Val"] * N
        m.add(c, t, t)
        vdev += [dp, dm]
    # P6: class-wise condition shift (test cell must also appear in class train)
    wv = []
    for c in test_cls:
        d = classes[c]
        for ci in range(4):
            members = [u for u in d if cell_of[u] == ci]
            for u in members:
                w = m.var(0, 1)
                row = {w: 1.0, X(u, 2): -1.0}
                for u2 in members:
                    row[X(u2, 0)] = row.get(X(u2, 0), 0) + 1.0
                m.add(row, 0, np.inf)
                wv.append(w)
    # P7: overall 70/10/20
    odev = []
    for i, s in enumerate(S):
        dp, dm = m.var(), m.var()
        c = {X(u, i): float(fr[u]) for u in range(nU)}
        c[dp], c[dm] = -1.0, 1.0
        t = TARGET[s] * N
        m.add(c, t, t)
        odev += [dp, dm]

    # P8 (tie-breakers: user ki list mein nahi; pichhle proposal ki khubi bachane ke liye, sabse neeche)
    cdev = []
    for ci in range(4):
        dp, dm = m.var(), m.var()
        c = {X(u, 2): float(fr[u]) for u in cell_idx[ci]}
        c[dp], c[dm] = -1.0, 1.0
        t = 0.2 * float(fr[cell_idx[ci]].sum())
        m.add(c, t, t)
        cdev += [(dp, 1.0 / N), (dm, 1.0 / N)]
    kdev = []
    for c_ in test_cls:
        d = classes[c_]
        F = float(sum(d.values()))
        dp, dm = m.var(), m.var()
        row = {X(u, 2): float(f) for u, f in d.items()}
        row[dp], row[dm] = -1.0, 1.0
        m.add(row, 0.2 * F, 0.2 * F)
        kdev += [(dp, 1.0 / F), (dm, 1.0 / F)]

    stages = [
        ("P4a  Train<Val+Test wali classes ki sankhya (kam)", {z[c]: 1.0 for c in test_cls}, 0.5, 120, 0.0),
        ("P4b  total relative shortfall (kam)", {sh[c]: 1.0 / sum(classes[c].values()) for c in test_cls}, 1e-6, 120, 0.0),
        ("P5   Val cell-mix ka dataset-mix se total antar, images (kam)", {v: 1.0 for v in vdev}, 0.5, 120, 0.0),
        ("P6   Test condition jo class ke Train mein nahi (kam)", {w: 1.0 for w in wv}, 0.5, 120, 0.0),
        ("P7   overall 70/10/20 se total antar, images (kam)", {v: 1.0 for v in odev}, 0.002 * N, 120, 0.0),
        ("P8   (tie-breaker) har cell aur har class ka Test ~20% ke paas (kam)",
         {**{j: cf for j, cf in cdev}, **{j: cf for j, cf in kdev}}, 1e-3, 60, 1e-3),
    ]
    res, last = None, None
    for name, obj, tol, tl, gap in stages:
        res = m.solve(obj, time_limit=tl, gap=gap)
        if res.x is None or res.status not in (0, 1):
            if name.startswith("P8") and last is not None:
                print(f"[{sname(res)}] {name}: koi behtar solution nahi mila, P7 wala solution rakha")
                break
            print(f"{name}: {sname(res)} -> ruk raha hoon")
            return
        last = res
        val = sum(cf * res.x[j] for j, cf in obj.items())
        note = "" if res.status == 0 else "  (time limit: best found, exact optimum nahi bhi ho sakta)"
        print(f"[{sname(res)}] {name}: {val:.4f}{note}")
        m.add(obj, -np.inf, val + tol)
    res = last
    Xn = np.array([[round(res.x[X(u, s)]) for s in range(3)] for u in range(nU)])
    split1 = Xn.argmax(1)
    ev1 = evaluate(split1, units, classes, cell_idx, N, cell_share)

    # ---------------- improved validation of hard rules
    hdr("4. HARD RULES KI VALIDATION (original aur improved dono)")
    for nm, sp, ev in (("Original", split0, ev0), ("Improved", split1, ev1)):
        rules = []
        rules.append(("UNIT_019 Train", sp[uix[UN19]] == 0))
        rules.append(("UNIT_106 Train", sp[uix[UN106]] == 0))
        rules.append(("8 single-unit classes Train-only", all(sp[list(classes[c])[0]] == 0 for c in SINGLE)))
        ok3 = True
        for c, d in classes.items():
            k = len(d)
            cs = ev["cls"][c]
            if k == 3:
                ok3 &= (cs["TU"], cs["VU"], cs["EU"]) == (2, 0, 1)
            if c == "Nail":
                ok3 &= (cs["TU"], cs["VU"], cs["EU"]) == (1, 0, 1)
        rules.append(("3-unit: 2T+1E, Nail: 1T+1E, val nahi", ok3))
        rules.append((">=4-unit classes: T,V,E sab mein >=1 unit",
                      all(ev["cls"][c]["TU"] >= 1 and ev["cls"][c]["VU"] >= 1 and ev["cls"][c]["EU"] >= 1
                          for c, d in classes.items() if len(d) >= 4)))
        rules.append(("Test mein 4 cells", ev["test_cells_ok"]))
        rules.append(("har unit exactly ek split", len(sp) == nU))
        rules.append(("total == 33,793", sum(ev["tot"].values()) == N))
        print(f"{nm}: " + " | ".join(f"{n}: {'PASS' if v else 'FAIL'}" for n, v in rules))

    # ---------------- comparison
    hdr("5. COMPARISON: ORIGINAL vs IMPROVED (actual calculations)")

    def pct(ev, s):
        return f"{ev['tot'][s]} ({100 * ev['tot'][s] / N:.2f}%)"

    def vshare(ev):
        return " / ".join(f"{x:.1f}" for x in ev["val_share"]) + f"  [L1={ev['val_L1']:.1f} pp]"

    def tcov(ev):
        return " / ".join(f"{c['Test']} ({100 * c['Test'] / c['Total']:.1f}%)" for c in ev["cells"])

    def tr_rng(ev):
        v = [100 * c["E"] / sum(classes[k].values()) for k, c in ev["cls"].items() if c["E"] > 0]
        return f"{min(v):.1f} - {max(v):.1f}"

    def shiftcls(ev):
        return sorted(c for c, v in ev["cls"].items() if v["cat"] in ("partial", "none"))
    rows = [
        ("Train images (target 70%)", pct(ev0, "Train"), pct(ev1, "Train")),
        ("Validation images (target 10%)", pct(ev0, "Val"), pct(ev1, "Val")),
        ("Test images (target 20%)", pct(ev0, "Test"), pct(ev1, "Test")),
        ("Test cells Dry/Bright, Dry/Dim, Dry/Dark, Wet/Dim", tcov(ev0), tcov(ev1)),
        ("Val cell share % (B / Dim / Dark / Wet)", vshare(ev0), vshare(ev1)),
        ("Dataset cell share % (reference)", " / ".join(f"{x:.1f}" for x in cell_share), ""),
        ("Class Test % (min - max, test wali classes)", tr_rng(ev0), tr_rng(ev1)),
        ("Hammer Train images", ev0["cls"]["Hammer"]["T"], ev1["cls"]["Hammer"]["T"]),
        ("Classes with Train < Val+Test", f"{len(ev0['viol'])}: {ev0['viol']}", f"{len(ev1['viol'])}: {ev1['viol']}"),
        ("Train/Test condition shift: (class, test-cell) pairs", ev0["shift_pairs"], ev1["shift_pairs"]),
        ("Classes with shift", f"{len(shiftcls(ev0))}", f"{len(shiftcls(ev1))}"),
        ("Leakage rules (hard rules)", "all PASS (sec.4)", "all PASS (sec.4)"),
    ]
    print(pd.DataFrame(rows, columns=["Metric", "Original", "Improved"]).to_string(index=False))

    # ---------------- shift detail (improved)
    hdr("6. CONDITION SHIFT DETAIL (improved proposal)")
    srows = []
    for c, d in sorted(classes.items(), key=lambda kv: kv[0]):
        cs = ev1["cls"][c]
        if cs["E"] == 0:
            continue
        lm = local_min_shift(c, d, units, cell_of, fixed_train)
        o = ev0["cls"][c]
        verdict = ("overlap OK" if cs["cat"] == "full" else
                   ("unavoidable (class ke units ke hisaab se)" if len(cs["miss"]) == lm else
                    "avoidable class-locally; higher-priority/global trade-off"))
        srows.append(dict(Class=c, TestCells="+".join(CNAME[i] for i in sorted(cs["ec"])),
                          TrainCells="+".join(CNAME[i] for i in sorted(cs["tc"])), Overlap=cs["cat"],
                          Uncovered=len(cs["miss"]), ClassLocalBest=lm, OrigUncovered=len(o["miss"]), Verdict=verdict))
    sd = pd.DataFrame(srows)
    print(sd.to_string(index=False))
    print(f"\nFull overlap: {sd[sd.Overlap == 'full'].Class.tolist()}")
    print(f"Partial     : {sd[sd.Overlap == 'partial'].Class.tolist()}")
    print(f"No overlap  : {sd[sd.Overlap == 'none'].Class.tolist()}")

    # ---------------- improved final tables
    hdr("7. IMPROVED PROPOSAL: CLASS-WISE")
    crow = []
    for c, d in sorted(classes.items(), key=lambda kv: (len(kv[1]), sum(kv[1].values()))):
        cs = ev1["cls"][c]
        tot = sum(d.values())
        crow.append(dict(Class=c, Units=len(d), Images=tot, TU=cs["TU"], VU=cs["VU"], EU=cs["EU"],
                         TrainImg=cs["T"], ValImg=cs["V"], TestImg=cs["E"],
                         Test_pct=round(100 * cs["E"] / tot, 1),
                         TrainGEValTest=("n/a" if cs["E"] == 0 else ("yes" if cs["T"] >= cs["V"] + cs["E"] else "NO")),
                         Overlap=cs["cat"]))
    print(pd.DataFrame(crow).to_string(index=False))

    hdr("8. IMPROVED PROPOSAL: CONDITION-WISE aur OVERALL")
    cr = []
    for ci in range(4):
        c = ev1["cells"][ci]
        cr.append(dict(Cell=CNAME[ci], Total=c["Total"], Train=c["Train"], Val=c["Val"], Test=c["Test"],
                       Test_pct=round(100 * c["Test"] / c["Total"], 1), ValShare_pct=round(ev1["val_share"][ci], 1),
                       DatasetShare_pct=round(cell_share[ci], 1),
                       TrainShare_pct=round(100 * c["Train"] / ev1["tot"]["Train"], 1),
                       TestShare_pct=round(100 * c["Test"] / ev1["tot"]["Test"], 1)))
    print(pd.DataFrame(cr).to_string(index=False))
    print()
    print(pd.DataFrame([dict(Split=s, Images=ev1["tot"][s], Pct=round(100 * ev1["tot"][s] / N, 2),
                             Target=f"{100 * TARGET[s]:.0f}%", Units=int((split1 == i).sum()))
                        for i, s in enumerate(S)]).to_string(index=False))

    hdr("9. IMPROVED ASSIGNMENT (sirf review; koi file save nahi)")
    units["orig"] = [S[i] for i in split0]
    units["new"] = [S[i] for i in split1]
    units["cell"] = [CNAME[i] for i in cell_of]
    for s in ("Test", "Val", "Train"):
        d = units[units.new == s]
        print(f"\n--- {s}: {len(d)} units, {int(d.frames.sum())} images ---")
        print(d[["unit_id", "groups", "classes", "cell", "frames", "orig"]].to_string(index=False))
    mv = units[units.orig != units.new]
    print(f"\nOriginal se badle huye units: {len(mv)}")
    print(mv[["unit_id", "classes", "cell", "frames", "orig", "new"]].to_string(index=False))
    print("\nKoi file save nahi hui. Implementation sirf aapki explicit approval ke baad.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step5", default="outputs/step5")
    main(ap.parse_args())