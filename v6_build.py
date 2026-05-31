"""
BrainScope AI — v6 CLAHE Build
Applies CLAHE preprocessing to all local datasets, saves processed images,
then uploads to Roboflow, generates a dataset version, and starts training.

Resume capability: tracks uploaded files in uploaded_clahe.json
Logs everything to v6_log.txt
"""

import os
import glob
import time
import json
import shutil
import traceback
from datetime import datetime

import cv2
import numpy as np
from roboflow import Roboflow

# ── Config ───────────────────────────────────────────────────────────────────
API_KEY   = "CXCNe93GGPiEeF2XrcYE"
DST_WS    = "fire-cjxu1"
DST_PROJ  = "find-glioma-and-brainscope-ai"

BASE_DIR  = r"C:\Users\Acer NItro\Desktop\claude work\brainscope-ai"
DATASETS_DIR  = os.path.join(BASE_DIR, "datasets")
CLAHE_OUT_DIR = os.path.join(BASE_DIR, "clahe_datasets")
LOG_FILE  = os.path.join(BASE_DIR, "v6_log.txt")
DONE_FILE = os.path.join(BASE_DIR, "uploaded_clahe.json")

# CLAHE settings
CLAHE_CLIP_LIMIT  = 4.0
CLAHE_TILE_GRID   = (8, 8)
JPEG_QUALITY      = 95

# Datasets with YOLO split structure (train/valid/test each with images/ labels/)
YOLO_DATASETS = [
    "ali-rostami__labeled-mri-brain-tumor-dataset",
    "astrid__alzheimer-s-disease",
    "fcis-oudzj__alzheimer-s-detection",
]

# Classification dataset (subfolders = class names, no labels)
CLASSIFICATION_DATASET = {
    "source_path": os.path.join(BASE_DIR, "test_images", "test"),
    "source_name": "original",
    "classes": ["bleeding", "glioma", "ischemia", "meningioma", "normal", "pituitary"],
}

# Class name normaliser — maps raw names → project class names
NAME_MAP = {
    "glioma":              "glioma",
    "meningioma":          "meningioma",
    "pituitary":           "pituitary",
    "no tumor":            "normal",
    "no_tumor":            "normal",
    "notumor":             "normal",
    "normal":              "normal",
    "healthy":             "normal",
    "alzheimer":           "alzheimer",
    "alzheimers":          "alzheimer",
    "ad":                  "alzheimer",
    "mci":                 "alzheimer",
    "emci":                "alzheimer",
    "lmci":                "alzheimer",
    "cn":                  "normal",
    "non demented":        "normal",
    "nondemented":         "normal",
    "very mild dementia":  "alzheimer",
    "mild dementia":       "alzheimer",
    "moderate dementia":   "alzheimer",
    "milddemented":        "alzheimer",
    "moderatedemented":    "alzheimer",
    "severedemented":      "alzheimer",
    "verymilddemented":    "alzheimer",
    "mild_demented":       "alzheimer",
    "moderate_demented":   "alzheimer",
    "severe_demented":     "alzheimer",
    "very_mild_demented":  "alzheimer",
    "non_demented":        "normal",
    "label0":              "glioma",
    "label1":              "glioma",
    "label2":              "glioma",
    "tumor":               "glioma",
    "brain tumor":         "glioma",
    "positive":            "glioma",
    "negative":            "normal",
    "bleeding":            "bleeding",
    "hemorrhage":          "bleeding",
    "ischemia":            "ischemia",
    "stroke":              "ischemia",
}

# ── Logging ──────────────────────────────────────────────────────────────────
def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ── Resume helpers ────────────────────────────────────────────────────────────
def load_done():
    if os.path.exists(DONE_FILE):
        try:
            with open(DONE_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_done(done_set):
    with open(DONE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(done_set), f, indent=2)

# ── CLAHE ────────────────────────────────────────────────────────────────────
def apply_clahe(img_bgr):
    """Apply CLAHE to a BGR image via grayscale, return BGR result."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    eq    = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def process_and_save(src_path, dst_path):
    """Read image, apply CLAHE, save as JPEG quality 95. Returns True on success."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    img = cv2.imread(src_path)
    if img is None:
        log(f"  WARNING: Could not read image: {src_path}")
        return False
    result = apply_clahe(img)
    # Always save as .jpg
    dst_jpg = os.path.splitext(dst_path)[0] + ".jpg"
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    ok = cv2.imwrite(dst_jpg, result, encode_params)
    if not ok:
        log(f"  WARNING: Could not write image: {dst_jpg}")
        return False
    return True

# ── Label copy helper ────────────────────────────────────────────────────────
def copy_label(src_lbl_path, dst_lbl_dir, stem):
    """Copy .txt label file to destination labels directory if it exists."""
    os.makedirs(dst_lbl_dir, exist_ok=True)
    if os.path.exists(src_lbl_path):
        dst_lbl_path = os.path.join(dst_lbl_dir, stem + ".txt")
        if not os.path.exists(dst_lbl_path):
            shutil.copy2(src_lbl_path, dst_lbl_path)

# ── YOLO label class-id remapper ────────────────────────────────────────────
def remap_label_file(src_lbl_path, dst_lbl_path, labelmap):
    """
    Rewrite a YOLO .txt label file remapping class IDs to a target label map.
    Lines that map to an unknown class are dropped.
    labelmap: {old_class_id_int: new_class_name_str}
    Since Roboflow accepts annotation_labelmap per-upload, we just copy
    the label as-is and pass the labelmap at upload time.
    """
    os.makedirs(os.path.dirname(dst_lbl_path), exist_ok=True)
    if os.path.exists(src_lbl_path):
        shutil.copy2(src_lbl_path, dst_lbl_path)

# ── data.yaml reader ────────────────────────────────────────────────────────
def build_labelmap(yaml_path):
    """Read data.yaml and return {class_id: project_class_name}."""
    try:
        import yaml as _yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = _yaml.safe_load(f)
    except ImportError:
        # Minimal YAML parser for simple key: value / list format
        data = _simple_yaml(yaml_path)

    names = data.get("names", [])
    labelmap = {}
    for i, raw in enumerate(names):
        key    = str(raw).lower().strip()
        mapped = NAME_MAP.get(key, key)
        labelmap[i] = mapped
    log(f"  Class map: {dict(zip(names, labelmap.values()))}")
    return labelmap

def _simple_yaml(yaml_path):
    """Very minimal YAML loader for data.yaml files (names list only)."""
    result = {"names": []}
    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    in_names = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if stripped.startswith("- "):
                result["names"].append(stripped[2:].strip())
            elif stripped and not stripped.startswith("#"):
                in_names = False
    return result

# ── Full-image bounding box label writer (for classification images) ─────────
def write_fullimage_label(dst_lbl_dir, stem, class_name, all_class_names):
    """
    Write a YOLO-format .txt label file with a full-image bounding box.
    class_id is the index of class_name in all_class_names list.
    """
    os.makedirs(dst_lbl_dir, exist_ok=True)
    try:
        class_id = all_class_names.index(class_name)
    except ValueError:
        class_id = 0
    dst_lbl_path = os.path.join(dst_lbl_dir, stem + ".txt")
    with open(dst_lbl_path, "w", encoding="utf-8") as f:
        # Full image bounding box: cx=0.5 cy=0.5 w=1.0 h=1.0
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 — CLAHE preprocessing
# ═════════════════════════════════════════════════════════════════════════════

def phase1_clahe_yolo():
    """Process YOLO-format datasets with train/valid/test splits."""
    log("\n" + "=" * 60)
    log("PHASE 1a — CLAHE preprocessing (YOLO datasets)")
    log("=" * 60)

    for ds_name in YOLO_DATASETS:
        src_ds_dir = os.path.join(DATASETS_DIR, ds_name)
        if not os.path.isdir(src_ds_dir):
            log(f"  SKIP (not found): {src_ds_dir}")
            continue

        yaml_path = os.path.join(src_ds_dir, "data.yaml")
        if not os.path.exists(yaml_path):
            log(f"  SKIP (no data.yaml): {src_ds_dir}")
            continue

        log(f"\n  Dataset: {ds_name}")
        labelmap = build_labelmap(yaml_path)

        for split in ["train", "valid", "test"]:
            src_img_dir = os.path.join(src_ds_dir, split, "images")
            src_lbl_dir = os.path.join(src_ds_dir, split, "labels")
            if not os.path.isdir(src_img_dir):
                continue

            dst_img_dir = os.path.join(CLAHE_OUT_DIR, ds_name, split, "images")
            dst_lbl_dir = os.path.join(CLAHE_OUT_DIR, ds_name, split, "labels")
            os.makedirs(dst_img_dir, exist_ok=True)
            os.makedirs(dst_lbl_dir, exist_ok=True)

            images = (glob.glob(os.path.join(src_img_dir, "*.jpg"))  +
                      glob.glob(os.path.join(src_img_dir, "*.jpeg")) +
                      glob.glob(os.path.join(src_img_dir, "*.png"))  +
                      glob.glob(os.path.join(src_img_dir, "*.bmp")))

            log(f"    [{split}] {len(images)} images")
            ok_count = skip_count = fail_count = 0

            for img_path in images:
                fname      = os.path.basename(img_path)
                stem       = os.path.splitext(fname)[0]
                dst_path   = os.path.join(dst_img_dir, stem + ".jpg")

                if os.path.exists(dst_path):
                    skip_count += 1
                    # Still ensure label is copied
                    src_lbl = os.path.join(src_lbl_dir, stem + ".txt")
                    dst_lbl = os.path.join(dst_lbl_dir, stem + ".txt")
                    remap_label_file(src_lbl, dst_lbl, labelmap)
                    continue

                success = process_and_save(img_path, dst_path)
                if success:
                    src_lbl = os.path.join(src_lbl_dir, stem + ".txt")
                    dst_lbl = os.path.join(dst_lbl_dir, stem + ".txt")
                    remap_label_file(src_lbl, dst_lbl, labelmap)
                    ok_count += 1
                else:
                    fail_count += 1

            log(f"    [{split}] done — processed: {ok_count}, skipped: {skip_count}, failed: {fail_count}")

    log("\n  YOLO datasets CLAHE preprocessing complete.")


def phase1_clahe_classification():
    """Process classification-format test_images/test dataset."""
    log("\n" + "=" * 60)
    log("PHASE 1b — CLAHE preprocessing (classification dataset)")
    log("=" * 60)

    src_root   = CLASSIFICATION_DATASET["source_path"]
    src_name   = CLASSIFICATION_DATASET["source_name"]
    classes    = CLASSIFICATION_DATASET["classes"]

    if not os.path.isdir(src_root):
        log(f"  SKIP (not found): {src_root}")
        return

    for class_name in classes:
        src_class_dir = os.path.join(src_root, class_name)
        if not os.path.isdir(src_class_dir):
            log(f"  SKIP class dir (not found): {src_class_dir}")
            continue

        dst_img_dir = os.path.join(CLAHE_OUT_DIR, src_name, class_name)
        os.makedirs(dst_img_dir, exist_ok=True)

        images = (glob.glob(os.path.join(src_class_dir, "*.jpg"))  +
                  glob.glob(os.path.join(src_class_dir, "*.jpeg")) +
                  glob.glob(os.path.join(src_class_dir, "*.png"))  +
                  glob.glob(os.path.join(src_class_dir, "*.bmp")))

        log(f"  Class '{class_name}': {len(images)} images")
        ok_count = skip_count = fail_count = 0

        for img_path in images:
            fname    = os.path.basename(img_path)
            stem     = os.path.splitext(fname)[0]
            dst_path = os.path.join(dst_img_dir, stem + ".jpg")

            if os.path.exists(dst_path):
                skip_count += 1
                continue

            success = process_and_save(img_path, dst_path)
            if success:
                ok_count += 1
            else:
                fail_count += 1

        log(f"  Class '{class_name}' done — processed: {ok_count}, skipped: {skip_count}, failed: {fail_count}")

    log("\n  Classification dataset CLAHE preprocessing complete.")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Roboflow upload
# ═════════════════════════════════════════════════════════════════════════════

def phase2_upload_yolo(dst_project, done):
    """Upload YOLO-format CLAHE datasets to Roboflow."""
    log("\n" + "=" * 60)
    log("PHASE 2a — Upload YOLO CLAHE datasets to Roboflow")
    log("=" * 60)

    total_uploaded = total_skipped = total_failed = 0

    for ds_name in YOLO_DATASETS:
        clahe_ds_dir = os.path.join(CLAHE_OUT_DIR, ds_name)
        if not os.path.isdir(clahe_ds_dir):
            log(f"  SKIP (clahe dir not found): {clahe_ds_dir}")
            continue

        yaml_path = os.path.join(DATASETS_DIR, ds_name, "data.yaml")
        if not os.path.exists(yaml_path):
            log(f"  SKIP (no data.yaml): {ds_name}")
            continue

        labelmap = build_labelmap(yaml_path)
        log(f"\n  Uploading: {ds_name}")

        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(clahe_ds_dir, split, "images")
            lbl_dir = os.path.join(clahe_ds_dir, split, "labels")
            if not os.path.isdir(img_dir):
                continue

            images = (glob.glob(os.path.join(img_dir, "*.jpg"))  +
                      glob.glob(os.path.join(img_dir, "*.jpeg")) +
                      glob.glob(os.path.join(img_dir, "*.png")))
            if not images:
                continue

            log(f"    [{split}] {len(images)} images to upload")

            for img_path in images:
                fname  = os.path.basename(img_path)
                stem   = os.path.splitext(fname)[0]
                img_id = f"clahe|{ds_name}|{split}|{fname}"

                if img_id in done:
                    total_skipped += 1
                    continue

                lbl_path = os.path.join(lbl_dir, stem + ".txt")

                for attempt in range(3):
                    try:
                        kwargs = dict(
                            split=split if split != "valid" else "valid",
                            num_retry_uploads=2,
                            batch_name=f"v6_clahe_{ds_name}",
                        )
                        if os.path.exists(lbl_path):
                            kwargs["annotation_path"]     = lbl_path
                            kwargs["annotation_labelmap"] = labelmap

                        dst_project.upload(image_path=img_path, **kwargs)
                        done.add(img_id)
                        total_uploaded += 1

                        if total_uploaded % 100 == 0:
                            save_done(done)
                            log(f"    checkpoint — uploaded: {total_uploaded} | failed: {total_failed} | skipped: {total_skipped}")

                        time.sleep(0.1)
                        break

                    except Exception as e:
                        if attempt == 2:
                            total_failed += 1
                            log(f"    FAIL [{attempt+1}/3]: {fname} — {e}")
                        else:
                            time.sleep(2)

    return total_uploaded, total_skipped, total_failed


def phase2_upload_classification(dst_project, done):
    """Upload classification CLAHE images to Roboflow with full-image bbox annotations."""
    log("\n" + "=" * 60)
    log("PHASE 2b — Upload classification CLAHE dataset to Roboflow")
    log("=" * 60)

    src_name = CLASSIFICATION_DATASET["source_name"]
    classes  = CLASSIFICATION_DATASET["classes"]

    # Build labelmap: class_name -> class_id (numeric)
    labelmap = {i: classes[i] for i in range(len(classes))}

    total_uploaded = total_skipped = total_failed = 0

    for class_name in classes:
        class_img_dir = os.path.join(CLAHE_OUT_DIR, src_name, class_name)
        if not os.path.isdir(class_img_dir):
            log(f"  SKIP class dir (not found): {class_img_dir}")
            continue

        images = (glob.glob(os.path.join(class_img_dir, "*.jpg"))  +
                  glob.glob(os.path.join(class_img_dir, "*.jpeg")) +
                  glob.glob(os.path.join(class_img_dir, "*.png")))
        if not images:
            continue

        log(f"  Class '{class_name}': {len(images)} images")

        for img_path in images:
            fname  = os.path.basename(img_path)
            stem   = os.path.splitext(fname)[0]
            img_id = f"clahe|{src_name}|{class_name}|{fname}"

            if img_id in done:
                total_skipped += 1
                continue

            # Write a temp full-image bounding box label file
            tmp_lbl_dir  = os.path.join(CLAHE_OUT_DIR, src_name, "_tmp_labels", class_name)
            os.makedirs(tmp_lbl_dir, exist_ok=True)
            tmp_lbl_path = os.path.join(tmp_lbl_dir, stem + ".txt")

            try:
                class_id = classes.index(class_name)
            except ValueError:
                class_id = 0

            with open(tmp_lbl_path, "w", encoding="utf-8") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

            for attempt in range(3):
                try:
                    dst_project.upload(
                        image_path=img_path,
                        annotation_path=tmp_lbl_path,
                        annotation_labelmap=labelmap,
                        split="train",
                        num_retry_uploads=2,
                        batch_name=f"v6_clahe_{src_name}_{class_name}",
                    )
                    done.add(img_id)
                    total_uploaded += 1

                    if total_uploaded % 100 == 0:
                        save_done(done)
                        log(f"    checkpoint — uploaded: {total_uploaded} | failed: {total_failed} | skipped: {total_skipped}")

                    time.sleep(0.1)
                    break

                except Exception as e:
                    if attempt == 2:
                        total_failed += 1
                        log(f"    FAIL [{attempt+1}/3]: {fname} — {e}")
                    else:
                        time.sleep(2)

    return total_uploaded, total_skipped, total_failed


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Generate version + train
# ═════════════════════════════════════════════════════════════════════════════

def phase3_generate_and_train(dst_project):
    log("\n" + "=" * 60)
    log("PHASE 3 — Generate dataset version")
    log("=" * 60)

    settings = {
        "augmentation": {
            "bbFlipX":    {"percent": 0.5},
            "crop":       {"min": 0.0, "max": 0.15},
            "rotation":   {"degrees": 15},
            "brightness": {"brighten": True, "darken": True},
            "blur":       {"pixels": 1.5},
            "noise":      {"percent": 1},
        },
        "preprocessing": {
            "auto-orient": True,
            "resize": {"width": 640, "height": 640, "format": "Stretch to"},
        },
    }

    try:
        log("  Calling generate_version()...")
        new_ver = dst_project.generate_version(settings=settings)
        log(f"  New version generated: v{new_ver.version}")
    except Exception as e:
        log(f"  generate_version FAILED: {e}")
        log(traceback.format_exc())
        log("  >>> MANUAL: Roboflow dashboard → Generate Version → Train")
        return

    log("\n" + "=" * 60)
    log("PHASE 4 — Start training")
    log("=" * 60)

    model_candidates = ["rf-detr-l", "rf-detr-large", "rf-detr-medium", "yolov11x"]
    trained = False
    for model in model_candidates:
        try:
            log(f"  Attempting train with model: {model}")
            new_ver.train(model_type=model, speed="accurate")
            log(f"  Training started with {model} — check Roboflow dashboard!")
            trained = True
            break
        except Exception as e:
            log(f"  {model} failed: {e}, trying next...")

    if not trained:
        log("  All model candidates failed. Start training manually from the Roboflow dashboard.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(CLAHE_OUT_DIR, exist_ok=True)

    log("=" * 60)
    log("BrainScope AI — v6 CLAHE Build STARTED")
    log(f"  Base dir  : {BASE_DIR}")
    log(f"  CLAHE out : {CLAHE_OUT_DIR}")
    log(f"  Done file : {DONE_FILE}")
    log(f"  Log file  : {LOG_FILE}")
    log("=" * 60)

    # ── Phase 1: CLAHE preprocessing ──────────────────────────────────────────
    try:
        phase1_clahe_yolo()
    except Exception as e:
        log(f"PHASE 1a ERROR: {e}")
        log(traceback.format_exc())

    try:
        phase1_clahe_classification()
    except Exception as e:
        log(f"PHASE 1b ERROR: {e}")
        log(traceback.format_exc())

    # ── Phase 2: Roboflow upload ───────────────────────────────────────────────
    log("\n  Connecting to Roboflow...")
    try:
        rf          = Roboflow(api_key=API_KEY)
        dst_project = rf.workspace(DST_WS).project(DST_PROJ)
        log(f"  Connected: {DST_WS}/{DST_PROJ}")
    except Exception as e:
        log(f"  Roboflow connection FAILED: {e}")
        log(traceback.format_exc())
        log("  Cannot proceed with upload/train. Exiting.")
        return

    done = load_done()
    log(f"  Already uploaded (resume): {len(done)} files")

    total_up = total_sk = total_fa = 0

    try:
        u, s, f = phase2_upload_yolo(dst_project, done)
        total_up += u; total_sk += s; total_fa += f
    except Exception as e:
        log(f"PHASE 2a ERROR: {e}")
        log(traceback.format_exc())

    try:
        u, s, f = phase2_upload_classification(dst_project, done)
        total_up += u; total_sk += s; total_fa += f
    except Exception as e:
        log(f"PHASE 2b ERROR: {e}")
        log(traceback.format_exc())

    save_done(done)

    log("\n" + "=" * 60)
    log(f"UPLOAD COMPLETE — uploaded: {total_up} | skipped: {total_sk} | failed: {total_fa}")
    log("=" * 60)

    # ── Phase 3 & 4: Generate version + train ─────────────────────────────────
    try:
        phase3_generate_and_train(dst_project)
    except Exception as e:
        log(f"PHASE 3/4 ERROR: {e}")
        log(traceback.format_exc())

    log("\n" + "=" * 60)
    log("v6 BUILD FINISHED — starting watch_v6.py to auto-deploy when training completes.")
    log("=" * 60)

    # Auto-start watcher to deploy when training completes
    import subprocess as _sp
    _sp.Popen(
        ["python", os.path.join(BASE_DIR, "watch_v6.py")],
        stdout=open(os.path.join(BASE_DIR, "watch_v6_stdout.txt"), "w"),
        stderr=open(os.path.join(BASE_DIR, "watch_v6_stderr.txt"), "w"),
        creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0)
    )
    log("watch_v6.py started — will auto-deploy v6 when Roboflow training is done.")


if __name__ == "__main__":
    main()
