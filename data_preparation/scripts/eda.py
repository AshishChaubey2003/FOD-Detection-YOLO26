"""
FOD EDA script (Pascal VOC format).

Usage (from D:\\FOD_project):
    python src/eda.py --root "data/raw/<VOC folder>" --out outputs/eda

It auto-discovers images, XML annotations and split files (ImageSets/Main/*.txt)
anywhere under --root, so no folder names are assumed.
"""
import argparse
import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def discover(root: Path):
    images, xmls, splits = defaultdict(list), defaultdict(list), {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMG_EXT:
            images[p.stem].append(p)
        elif ext == ".xml":
            xmls[p.stem].append(p)
        elif ext == ".txt" and "ImageSets" in p.parts:
            splits[p.stem] = [l.split()[0] for l in p.read_text().splitlines() if l.strip()]
    return images, xmls, splits


def parse_xml(path: Path):
    r = ET.parse(path).getroot()
    size = r.find("size")
    w = int(float(size.findtext("width", 0))) if size is not None else 0
    h = int(float(size.findtext("height", 0))) if size is not None else 0
    objs = []
    for o in r.findall("object"):
        bb = o.find("bndbox")
        objs.append(dict(
            cls=o.findtext("name", "").strip(),
            xmin=float(bb.findtext("xmin")), ymin=float(bb.findtext("ymin")),
            xmax=float(bb.findtext("xmax")), ymax=float(bb.findtext("ymax")),
        ))
    return w, h, objs


def main(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg=""):
        print(msg)
        log.append(msg)

    images, xmls, splits = discover(root)
    say(f"Image files found : {sum(len(v) for v in images.values())}")
    say(f"XML files found   : {sum(len(v) for v in xmls.values())}")
    say(f"Split files found : {list(splits.keys())}")
    dup_stems = [s for s, v in images.items() if len(v) > 1]
    say(f"Image stems appearing more than once (different paths): {len(dup_stems)}")

    # ---- pair images and xmls
    img_only = sorted(set(images) - set(xmls))
    xml_only = sorted(set(xmls) - set(images))
    say(f"Images without XML: {len(img_only)} | XML without image: {len(xml_only)}")

    # ---- per-image table + per-object table
    img_rows, obj_rows, bad_imgs, bad_xml = [], [], [], []
    pairs = sorted(set(images) & set(xmls))
    for n, stem in enumerate(pairs):
        if n % 2000 == 0:
            print(f"  processed {n}/{len(pairs)}", flush=True)
        ip, xp = images[stem][0], xmls[stem][0]
        try:
            with Image.open(ip) as im:
                im.verify()
            with Image.open(ip) as im:
                iw, ih = im.size
        except Exception as e:
            bad_imgs.append((str(ip), repr(e)))
            continue
        try:
            xw, xh, objs = parse_xml(xp)
        except Exception as e:
            bad_xml.append((str(xp), repr(e)))
            continue
        img_rows.append(dict(stem=stem, path=str(ip), width=iw, height=ih,
                             xml_w=xw, xml_h=xh, n_obj=len(objs),
                             size_bytes=ip.stat().st_size))
        for o in objs:
            o.update(stem=stem, img_w=iw, img_h=ih)
            obj_rows.append(o)
    imgs, objs = pd.DataFrame(img_rows), pd.DataFrame(obj_rows)
    say(f"Corrupted images: {len(bad_imgs)} | Unparseable XML: {len(bad_xml)}")
    if imgs.empty:
        say("No valid image+XML pairs found. Check --root.")
        (out / "eda_summary.txt").write_text("\n".join(log))
        return

    # ---- image size info
    say("\n== Image resolution ==")
    say(imgs.groupby(["width", "height"]).size().sort_values(ascending=False).head(10).to_string())
    mism = imgs[(imgs.width != imgs.xml_w) | (imgs.height != imgs.xml_h)]
    say(f"Images where XML size != real image size: {len(mism)}")

    # ---- box metrics + invalid boxes
    objs["bw"] = objs.xmax - objs.xmin
    objs["bh"] = objs.ymax - objs.ymin
    objs["area"] = objs.bw * objs.bh
    objs["area_ratio"] = objs.area / (objs.img_w * objs.img_h)
    invalid = objs[(objs.bw <= 0) | (objs.bh <= 0) | (objs.xmin < 0) | (objs.ymin < 0)
                   | (objs.xmax > objs.img_w) | (objs.ymax > objs.img_h)]
    say(f"\n== Boxes ==\nTotal boxes: {len(objs)} | Invalid boxes: {len(invalid)}")
    say(f"Images with 0 objects: {(imgs.n_obj == 0).sum()}")
    say(f"Objects per image: mean={imgs.n_obj.mean():.2f}, max={imgs.n_obj.max()}")

    # ---- class distribution
    cc = objs.cls.value_counts()
    say("\n== Class distribution (boxes) ==")
    say(cc.to_string())
    say(f"Imbalance ratio (max/min): {cc.max() / cc.min():.1f}")
    small_abs = (objs.area < 32 * 32).mean() * 100
    small_rel = (objs.area_ratio < 0.01).mean() * 100
    say(f"Small boxes: {small_abs:.1f}% have area < 32x32 px | {small_rel:.1f}% cover < 1% of image")
    say("\nBox area ratio by class:")
    say(objs.groupby("cls").area_ratio.describe()[["count", "mean", "min", "50%", "max"]].round(4).to_string())

    # ---- duplicates (exact, by file hash)
    hashes = defaultdict(list)
    print("  hashing images for duplicate check...", flush=True)
    for r in imgs.itertuples():
        hashes[hashlib.md5(Path(r.path).read_bytes()).hexdigest()].append(r.stem)
    dup_groups = [v for v in hashes.values() if len(v) > 1]
    say(f"\nExact duplicate image groups (same bytes): {len(dup_groups)} "
        f"({sum(len(g) for g in dup_groups)} files)")

    # ---- provided splits + leakage
    say("\n== Provided splits ==")
    for name, ids in splits.items():
        say(f"{name}: {len(ids)} ids")
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = set(splits[names[i]]) & set(splits[names[j]])
            say(f"overlap {names[i]} vs {names[j]}: {len(ov)}")
    stem2split = {s: n for n, ids in splits.items() if n in ("train", "val", "test") for s in ids}
    if stem2split:
        objs["split"] = objs.stem.map(stem2split)
        say("\nClass counts per split:")
        say(pd.crosstab(objs.cls, objs.split.fillna("unassigned")).to_string())

    # ---- save tables
    imgs.to_csv(out / "images.csv", index=False)
    objs.to_csv(out / "annotations.csv", index=False)
    invalid.to_csv(out / "invalid_boxes.csv", index=False)
    pd.DataFrame(bad_imgs, columns=["file", "error"]).to_csv(out / "corrupted_images.csv", index=False)
    pd.DataFrame(dup_groups).to_csv(out / "duplicate_groups.csv", index=False)

    # ---- plots
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=cc.index, y=cc.values, ax=ax)
    ax.set_title("Boxes per class"); ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(cc.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(objs.area_ratio, bins=50, ax=axs[0]); axs[0].set_title("Box area / image area"); axs[0].set_xscale("log")
    sns.histplot(objs.bw, bins=40, ax=axs[1]); axs[1].set_title("Box width (px)")
    sns.histplot(objs.bh, bins=40, ax=axs[2]); axs[2].set_title("Box height (px)")
    fig.tight_layout(); fig.savefig(out / "box_size_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x=imgs.n_obj, ax=ax); ax.set_title("Objects per image")
    fig.tight_layout(); fig.savefig(out / "objects_per_image.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=objs, x="cls", y="area_ratio", ax=ax)
    ax.set_yscale("log"); ax.set_title("Box area ratio by class"); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(out / "area_ratio_by_class.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    cx = (objs.xmin + objs.xmax) / 2 / objs.img_w
    cy = (objs.ymin + objs.ymax) / 2 / objs.img_h
    ax.hist2d(cx, cy, bins=30, range=[[0, 1], [0, 1]]); ax.invert_yaxis()
    ax.set_title("Box centre heatmap (normalised)")
    fig.tight_layout(); fig.savefig(out / "box_center_heatmap.png", dpi=150); plt.close(fig)

    # ---- sample images with boxes drawn
    import cv2
    samp_dir = out / "samples"
    samp_dir.mkdir(exist_ok=True)
    for stem in imgs.sample(min(12, len(imgs)), random_state=42).stem:
        p = imgs.loc[imgs.stem == stem, "path"].iloc[0]
        im = cv2.imread(p)
        if im is None:
            continue
        for o in objs[objs.stem == stem].itertuples():
            cv2.rectangle(im, (int(o.xmin), int(o.ymin)), (int(o.xmax), int(o.ymax)), (0, 255, 0), 2)
            cv2.putText(im, o.cls, (int(o.xmin), max(12, int(o.ymin) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(samp_dir / f"{stem}.jpg"), im)

    (out / "eda_summary.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\nDone. Outputs saved in: {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="VOC dataset root (folder with JPEGImages/Annotations)")
    ap.add_argument("--out", default="outputs/eda")
    a = ap.parse_args()
    main(a.root, a.out)