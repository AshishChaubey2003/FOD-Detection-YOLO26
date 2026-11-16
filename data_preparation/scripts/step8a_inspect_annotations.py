"""
Step 8A: INSPECT Pascal VOC XML annotations. READ-ONLY.
Dataset ya XML files ko kuch nahi badalta. Koi YOLO label, koi mapping, koi split
file nahi banti -- sirf ek inspection report.

Run from D:\\FOD_project:
    python src\\step8a_inspect_annotations.py --root "data\\raw\\FODPascalVOCFormat-V.2.1"

Writes: outputs\\step8_yolo\\step8a_inspection_report.txt (aur terminal par print)
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png"}


def find_files(root, exts):
    return {p.stem: p for p in root.rglob("*") if p.suffix.lower() in exts}


def text(node, tag, default=None):
    e = node.find(tag)
    return e.text.strip() if e is not None and e.text is not None else default


def parse_xml(path):
    """Returns dict ya None (parse error par)."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        return {"error": f"XML parse error: {e}"}
    size = root.find("size")
    w = int(text(size, "width", "0")) if size is not None else None
    h = int(text(size, "height", "0")) if size is not None else None
    filename = text(root, "filename")
    objs = []
    for o in root.findall("object"):
        name = text(o, "name", "<missing>")
        difficult = text(o, "difficult")
        truncated = text(o, "truncated")
        pose = text(o, "pose")
        bb = o.find("bndbox")
        box = None
        if bb is not None:
            try:
                box = tuple(float(text(bb, t)) for t in ("xmin", "ymin", "xmax", "ymax"))
            except (TypeError, ValueError):
                box = None
        objs.append(dict(name=name, difficult=difficult, truncated=truncated, pose=pose, box=box))
    return dict(filename=filename, width=w, height=h, objects=objs, has_size_tag=size is not None)


def main(a):
    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(m=""):
        print(m)
        log.append(str(m))

    say("=" * 74)
    say("STEP 8A: XML ANNOTATION INSPECTION (read-only)")
    say("=" * 74)
    say(f"Root: {root.resolve()}")

    xmls = find_files(root, {".xml"})
    imgs = find_files(root, IMG_EXT)
    say(f"XML files found (recursive): {len(xmls)}")
    say(f"Image files found (recursive): {len(imgs)}")
    stems = sorted(set(xmls) & set(imgs))
    only_xml = sorted(set(xmls) - set(imgs))
    only_img = sorted(set(imgs) - set(xmls))
    say(f"Stems with BOTH xml+image: {len(stems)}")
    say(f"XML with no matching image: {len(only_xml)}  (sample: {only_xml[:5]})")
    say(f"Image with no matching XML: {len(only_img)}  (sample: {only_img[:5]})")
    if not stems:
        say("\n[STOP] Koi bhi xml+image pair nahi mili. Root path check karein.")
        (out / "step8a_inspection_report.txt").write_text("\n".join(log), encoding="utf-8")
        return

    # ---- parse all XML ----
    say("\n" + "=" * 74)
    say("PARSING")
    say("=" * 74)
    parsed = {}
    parse_errors = []
    for n, s in enumerate(stems):
        if n % 5000 == 0:
            print(f"  parsing {n}/{len(stems)}", flush=True)
        p = parse_xml(xmls[s])
        if "error" in p:
            parse_errors.append((s, p["error"]))
        else:
            parsed[s] = p
    say(f"Parsed OK: {len(parsed)}  |  Parse errors: {len(parse_errors)}")
    for s, e in parse_errors[:10]:
        say(f"   {s}: {e}")

    missing_size = [s for s, p in parsed.items() if not p["has_size_tag"] or not p["width"] or not p["height"]]
    say(f"XML missing <size>/<width>/<height>: {len(missing_size)}  (sample: {missing_size[:5]})")

    # ---- filename field ----
    say("\n" + "=" * 74)
    say("FILENAME FIELD")
    say("=" * 74)
    has_fn = sum(1 for p in parsed.values() if p["filename"])
    say(f"XML with a <filename> tag: {has_fn} / {len(parsed)}")
    mismatch_fn = []
    for s, p in parsed.items():
        if p["filename"]:
            fn_stem = Path(p["filename"]).stem
            if fn_stem != s:
                mismatch_fn.append((s, p["filename"]))
    say(f"<filename> stem != xml/image stem: {len(mismatch_fn)}  (sample: {mismatch_fn[:5]})")

    # ---- objects per image, empty annotations, multi-object ----
    say("\n" + "=" * 74)
    say("OBJECTS PER IMAGE")
    say("=" * 74)
    counts = {s: len(p["objects"]) for s, p in parsed.items()}
    n_empty = sum(1 for c in counts.values() if c == 0)
    n_multi = sum(1 for c in counts.values() if c > 1)
    mx = max(counts.values()) if counts else 0
    say(f"Images with 0 objects (empty annotation): {n_empty}")
    say(f"Images with exactly 1 object: {sum(1 for c in counts.values() if c == 1)}")
    say(f"Images with >1 object: {n_multi}  (max objects in one image: {mx})")
    from collections import Counter
    hist = Counter(counts.values())
    say("Distribution (n_objects: n_images), top 10 by count:")
    for k in sorted(hist)[:10]:
        say(f"   {k}: {hist[k]}")
    empty_ids = [s for s, c in counts.items() if c == 0][:10]
    say(f"Sample empty-annotation ids: {empty_ids}")

    # ---- class names ----
    say("\n" + "=" * 74)
    say("CLASS NAMES (as found verbatim in XML)")
    say("=" * 74)
    cls_count = Counter()
    missing_name = 0
    for p in parsed.values():
        for o in p["objects"]:
            if o["name"] == "<missing>":
                missing_name += 1
            else:
                cls_count[o["name"]] += 1
    say(f"Unique class names found: {len(cls_count)}")
    say(f"Objects with missing <name>: {missing_name}")
    for c, n in sorted(cls_count.items()):
        say(f"   {c!r}: {n}")
    lowered = Counter(c.lower() for c in cls_count)
    dup_casing = [c for c in lowered if lowered[c] > 1]
    if dup_casing:
        say(f"\nWARNING: casing/whitespace ke hisaab se possible duplicate class names: {dup_casing}")
    stripped_diff = [c for c in cls_count if c != c.strip()]
    if stripped_diff:
        say(f"WARNING: class names jinme leading/trailing space hai: {stripped_diff}")

    # ---- difficult / truncated / pose ----
    say("\n" + "=" * 74)
    say("DIFFICULT / TRUNCATED / POSE FLAGS")
    say("=" * 74)
    diff_vals = Counter()
    trunc_vals = Counter()
    pose_vals = Counter()
    for p in parsed.values():
        for o in p["objects"]:
            diff_vals[o["difficult"]] += 1
            trunc_vals[o["truncated"]] += 1
            pose_vals[o["pose"]] += 1
    say(f"difficult tag value counts (None = tag absent): {dict(diff_vals)}")
    say(f"truncated tag value counts (None = tag absent): {dict(trunc_vals)}")
    say(f"pose tag value counts (None = tag absent): {dict(pose_vals)}")

    # ---- box validity + coordinate convention ----
    say("\n" + "=" * 74)
    say("BOUNDING BOX VALIDITY AND COORDINATE CONVENTION")
    say("=" * 74)
    all_xmin, all_ymin, all_xmax_eq_w, all_ymax_eq_h = [], [], 0, 0
    bad_order, bad_neg, bad_oob, no_box = [], [], [], []
    n_boxes = 0
    for s, p in parsed.items():
        W, H = p["width"] or 0, p["height"] or 0
        for o in p["objects"]:
            if o["box"] is None:
                no_box.append(s)
                continue
            n_boxes += 1
            xmin, ymin, xmax, ymax = o["box"]
            all_xmin.append(xmin)
            all_ymin.append(ymin)
            if W and xmax == W:
                all_xmax_eq_w += 1
            if H and ymax == H:
                all_ymax_eq_h += 1
            if xmin >= xmax or ymin >= ymax:
                bad_order.append((s, o["box"]))
            if xmin < 0 or ymin < 0:
                bad_neg.append((s, o["box"]))
            if W and H and (xmax > W or ymax > H):
                bad_oob.append((s, o["box"], W, H))
    say(f"Total boxes parsed: {n_boxes}  |  <bndbox> missing/unparseable: {len(no_box)}")
    if all_xmin:
        say(f"xmin: min={min(all_xmin):.1f} max={max(all_xmin):.1f} | count(xmin==0): "
            f"{sum(1 for v in all_xmin if v == 0)} | count(xmin==1): {sum(1 for v in all_xmin if v == 1)}")
        say(f"ymin: min={min(all_ymin):.1f} max={max(all_ymin):.1f} | count(ymin==0): "
            f"{sum(1 for v in all_ymin if v == 0)} | count(ymin==1): {sum(1 for v in all_ymin if v == 1)}")
        say(f"Boxes touching right edge (xmax == image width): {all_xmax_eq_w}")
        say(f"Boxes touching bottom edge (ymax == image height): {all_ymax_eq_h}")
        if min(all_xmin) == 0 and min(all_ymin) == 0:
            say("-> xmin/ymin ka minimum 0 hai: ZERO-BASED, half-open-pixel convention jaisa dikhta hai "
                "(standard midpoint formula bina adjustment ke chalega).")
        elif min(all_xmin) >= 1 and min(all_ymin) >= 1:
            say("-> xmin/ymin kabhi 0 nahi: ONE-BASED (classic Pascal VOC) convention jaisa dikhta hai. "
                "YOLO normalize karne se pehle -1 adjustment par vichar chahiye (effect chhota hota hai, "
                "lekin boxes ko ek pixel se consistently right/down shift kar sakta hai).")
        else:
            say("-> MIXED: kuch boxes 0 se shuru hote hain, kuch 1 se. Convention clear nahi hai -- "
                "ise resolve karne ke baad hi conversion aage badhegi.")
    say(f"\nxmin>=xmax ya ymin>=ymax (invalid order): {len(bad_order)}  (sample: {bad_order[:5]})")
    say(f"Negative coordinates: {len(bad_neg)}  (sample: {bad_neg[:5]})")
    say(f"Box bada image se (xmax>width ya ymax>height, XML ke apne width/height ke hisaab se): "
        f"{len(bad_oob)}  (sample: {bad_oob[:5]})")

    # ---- XML size vs actual image size ----
    say("\n" + "=" * 74)
    say("XML <size> vs ACTUAL IMAGE DIMENSIONS")
    say("=" * 74)
    mismatches = []
    unreadable = []
    checked = 0
    for n, s in enumerate(stems):
        if n % 5000 == 0:
            print(f"  checking image dims {n}/{len(stems)}", flush=True)
        p = parsed.get(s)
        if p is None or not p["width"] or not p["height"]:
            continue
        try:
            with Image.open(imgs[s]) as im:
                aw, ah = im.size
        except Exception as e:
            unreadable.append((s, str(e)))
            continue
        checked += 1
        if (aw, ah) != (p["width"], p["height"]):
            mismatches.append((s, (p["width"], p["height"]), (aw, ah)))
    say(f"Images checked: {checked}  |  Unreadable: {len(unreadable)}  (sample: {unreadable[:5]})")
    say(f"XML size != actual image size: {len(mismatches)}  (sample: {mismatches[:5]})")

    # ---- representative examples ----
    say("\n" + "=" * 74)
    say("SAMPLE PARSED ENTRIES (representative)")
    say("=" * 74)
    sample_stems = stems[:3] + [s for s in stems if counts.get(s, 0) > 1][:2] + empty_ids[:2]
    for s in dict.fromkeys(sample_stems):
        p = parsed.get(s)
        if p is None:
            continue
        say(f"\n{s}: filename={p['filename']!r} size=({p['width']},{p['height']})")
        for o in p["objects"]:
            say(f"   name={o['name']!r} difficult={o['difficult']} truncated={o['truncated']} box={o['box']}")

    say("\n" + "=" * 74)
    say(f"TOTAL UNIQUE CLASSES FOUND: {len(cls_count)}")
    say("=" * 74)
    say("Koi YOLO label, class mapping, ya split file is step mein NAHI banayi gayi.")
    say("Step 8B (class mapping) aur Step 8C (conversion) iske actual numbers dekhne ke baad hi banenge.")

    (out / "step8a_inspection_report.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/FODPascalVOCFormat-V.2.1")
    ap.add_argument("--out", default="outputs/step8_yolo")
    main(ap.parse_args())