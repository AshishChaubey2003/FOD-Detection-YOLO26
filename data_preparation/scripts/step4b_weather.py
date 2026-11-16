"""
Step 4b: inspect FOD_categorization_annotations.csv (weather / light labels). READ-ONLY.

Run from D:\\FOD_project:
    python src\\step4b_weather.py --root "data\\raw\\FODPascalVOCFormat-V.2.1"

Optional overrides if auto-detection fails:
    --id-col NAME --weather-col NAME --light-col NAME
Writes only to outputs\\step4b\\.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def norm_id(x):
    s = str(x).strip().replace("\\", "/").split("/")[-1]
    s = re.sub(r"\.(jpg|jpeg|png|bmp|xml)$", "", s, flags=re.I)
    return s.zfill(6) if s.isdigit() else s


def parse_categories(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\w+)\s*:\s*\[(.*?)\]", line)
        if m:
            out[m.group(1).lower()] = [x.strip() for x in m.group(2).split(",") if x.strip()]
    return out


def to_label(series, names):
    """numeric code -> name (using category_information list); strings kept."""
    def f(v):
        if pd.isna(v):
            return np.nan
        t = str(v).strip()
        try:
            n = int(float(t))
            return names[n] if names and 0 <= n < len(names) else f"UNKNOWN({t})"
        except ValueError:
            return t
    return series.map(f)


def class_runs(ann_csv, stems):
    ann = pd.read_csv(ann_csv, dtype={"stem": str})
    cls_sets = ann.groupby("stem").cls.agg(lambda x: frozenset(x)).to_dict()
    run, cur = {}, 0
    for i, s in enumerate(stems):
        if i > 0 and not (cls_sets[s] & cls_sets[stems[i - 1]]):
            cur += 1
        run[s] = cur
    main_box = ann.sort_values("area").groupby("stem").tail(1).set_index("stem")
    return run, main_box.cls.to_dict()


def main(root, eda_dir, step3_dir, out, id_col, w_col, l_col):
    root, eda_dir, step3_dir, out = Path(root), Path(eda_dir), Path(step3_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg=""):
        print(msg)
        log.append(str(msg))

    csvs = [p for p in root.rglob("*.csv") if "categorization" in p.name.lower()]
    infos = [p for p in root.rglob("category_information*.txt")]
    if not csvs:
        say("No *categorization*.csv found under --root. Stopping.")
        return
    csv_path = csvs[0]
    say(f"CSV: {csv_path.relative_to(root)} ({csv_path.stat().st_size} bytes)")
    cats = {}
    if infos:
        txt = infos[0].read_text(errors="replace")
        cats = parse_categories(txt)
        say(f"category_information parsed: {cats}")

    # ---- 1. read + columns
    df = pd.read_csv(csv_path, dtype=str)
    if any(re.fullmatch(r"\d+(\.\w+)?", str(c).strip()) for c in df.columns):
        say("Header looks like data (no header row). Re-reading with header=None.")
        df = pd.read_csv(csv_path, dtype=str, header=None)
        df.columns = [f"col{i}" for i in range(df.shape[1])]
    say(f"\n== 1. Structure ==\nrows={len(df)} columns={list(df.columns)}")
    say("first 5 rows:")
    say(df.head().to_string())
    num_try = {c: pd.to_numeric(df[c], errors="coerce").notna().mean() for c in df.columns}
    say("dtype guess (share of values that are numeric): " + str({c: round(v, 3) for c, v in num_try.items()}))
    say("missing values per column: " + str(df.isna().sum().to_dict()))
    say("unique values per column (first 12): " +
        str({c: df[c].dropna().unique()[:12].tolist() for c in df.columns}))

    # ---- 2. dataset stems
    stems_all = sorted({p.stem for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT})
    stemset = set(stems_all)
    say(f"\nImages on disk: {len(stems_all)}")

    # ---- 3. id column
    if id_col is None:
        best, bestf = None, 0
        for c in df.columns:
            f = df[c].dropna().map(norm_id).isin(stemset).mean()
            if f > bestf:
                best, bestf = c, f
        id_col = best if bestf > 0.5 else None
        say(f"auto-detected id column: {id_col} (match fraction {bestf:.3f})")
    if id_col is None:
        say("Could not detect the id column. Re-run with --id-col NAME. Stopping.")
        (out / "step4b_report.txt").write_text("\n".join(log), encoding="utf-8")
        return
    df["stem"] = df[id_col].map(norm_id)

    # ---- 4. weather / light columns
    cols = [c for c in df.columns if c not in (id_col, "stem")]
    w_col = w_col or next((c for c in cols if "weather" in str(c).lower()), None)
    l_col = l_col or next((c for c in cols if "light" in str(c).lower()), None)
    if not (w_col and l_col):
        say(f"Could not detect weather/light columns among {cols}. Use --weather-col / --light-col. Stopping.")
        (out / "step4b_report.txt").write_text("\n".join(log), encoding="utf-8")
        return
    say(f"weather column: {w_col} | light column: {l_col}")
    df["weather"] = to_label(df[w_col], cats.get("weather"))
    df["light"] = to_label(df[l_col], cats.get("light"))

    # ---- 5. id mapping quality
    say("\n== 2. ID mapping ==")
    dup = df[df.stem.duplicated(keep=False)]
    conflict = dup.groupby("stem").apply(lambda g: g[["weather", "light"]].drop_duplicates().shape[0] > 1,
                                         include_groups=False).sum() if len(dup) else 0
    say(f"csv rows: {len(df)} | unique ids: {df.stem.nunique()} | duplicate-id rows: {len(dup)} "
        f"| duplicate ids with conflicting labels: {int(conflict)}")
    unmatched = sorted(set(df.stem) - stemset)
    no_row = sorted(stemset - set(df.stem))
    say(f"csv ids with NO image on disk: {len(unmatched)} {unmatched[:10]}")
    say(f"images on disk with NO csv row: {len(no_row)} {no_row[:10]}")
    lab = df.drop_duplicates("stem").set_index("stem")[["weather", "light"]]
    lab.to_csv(out / "weather_light_by_image.csv")

    # ---- 6. distributions
    say("\n== 3. Distributions (unique ids) ==")
    for c in ("weather", "light"):
        vc = lab[c].value_counts(dropna=False)
        say(f"{c}:\n" + pd.DataFrame({"count": vc, "pct": (100 * vc / len(lab)).round(2)}).to_string())
    say("weather x light:\n" + pd.crosstab(lab.weather, lab.light, margins=True).to_string())

    # ---- 7. constancy within sequences
    eda_ann = eda_dir / "annotations.csv"
    if eda_ann.exists():
        stems_lab = [s for s in stems_all if s in lab.index]
        run, main_cls = class_runs(eda_ann, stems_all)
        seqdefs = {"A: class-run": run}
        s3 = step3_dir / "sequences.csv"
        if s3.exists():
            sq = pd.read_csv(s3)
            B = {}
            for r in sq.itertuples():
                for i in range(r.start_id, r.end_id + 1):
                    B[str(i).zfill(6)] = r.seq
            seqdefs["B: class-run+scene-cut (Step 3)"] = B
        ordered = [s for s in stems_all if s in lab.index]
        # per-class distribution
        cl = pd.DataFrame({"stem": ordered, "cls": [main_cls.get(s) for s in ordered],
                           "weather": lab.loc[ordered, "weather"].values, "light": lab.loc[ordered, "light"].values})
        cl.groupby("cls").weather.value_counts().unstack(fill_value=0).to_csv(out / "class_x_weather.csv")
        cl.groupby("cls").light.value_counts().unstack(fill_value=0).to_csv(out / "class_x_light.csv")
        for name, seg in seqdefs.items():
            d = pd.DataFrame({"stem": ordered, "seq": [seg.get(s, -1) for s in ordered],
                              "weather": lab.loc[ordered, "weather"].values,
                              "light": lab.loc[ordered, "light"].values,
                              "cls": [main_cls.get(s) for s in ordered]})
            g = d.groupby("seq")
            nw, nl = g.weather.nunique(dropna=False), g.light.nunique(dropna=False)
            say(f"\n== 4. Constancy inside sequences | definition {name} ==")
            say(f"sequences: {len(nw)} | constant weather: {(nw == 1).sum()} ({100 * (nw == 1).mean():.1f}%) "
                f"| constant light: {(nl == 1).sum()} ({100 * (nl == 1).mean():.1f}%)")
            flips_w = (d.weather.values[1:] != d.weather.values[:-1]).sum()
            flips_l = (d.light.values[1:] != d.light.values[:-1]).sum()
            say(f"label changes between consecutive ids: weather={flips_w}, light={flips_l} (of {len(d) - 1} pairs)")
            mixed = d[d.seq.isin(nw[(nw > 1)].index.union(nl[(nl > 1)].index))]
            if len(mixed):
                comp = mixed.groupby("seq").agg(first=("stem", "min"), last=("stem", "max"), n=("stem", "count"),
                                                cls=("cls", lambda x: x.mode()[0]),
                                                weather=("weather", lambda x: dict(x.value_counts())),
                                                light=("light", lambda x: dict(x.value_counts())))
                comp.to_csv(out / f"mixed_sequences_{name[0]}.csv")
                say(f"sequences with mixed labels: {len(comp)} (saved)")
                say(comp.head(15).to_string())
            # independent videos per label value
            rows = []
            for col in ("weather", "light"):
                for v, gg in d.groupby(col):
                    maj = g[col].agg(lambda x: x.mode()[0])
                    rows.append(dict(label=f"{col}={v}", frames=len(gg), sequences_containing=gg.seq.nunique(),
                                     sequences_majority=int((maj == v).sum()), classes=gg.cls.nunique()))
            say(f"\nHow many independent sequences support each label value ({name}):")
            say(pd.DataFrame(rows).to_string(index=False))
            pd.DataFrame(rows).to_csv(out / f"label_support_{name[0]}.csv", index=False)
    else:
        say("\n(outputs\\eda\\annotations.csv missing: skipping sequence constancy analysis)")

    (out / "step4b_report.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\nDone. Report saved in: {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--eda", default="outputs/eda")
    ap.add_argument("--step3", default="outputs/step3")
    ap.add_argument("--out", default="outputs/step4b")
    ap.add_argument("--id-col")
    ap.add_argument("--weather-col")
    ap.add_argument("--light-col")
    a = ap.parse_args()
    main(a.root, a.eda, a.step3, a.out, a.id_col, a.weather_col, a.light_col)