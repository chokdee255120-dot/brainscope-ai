"""
BrainScope AI — Overnight Dataset Builder
Runs unattended: downloads, maps classes, uploads, then auto-trains.
Progress is saved to overnight_log.txt — check it in the morning.
"""
import os, glob, time, yaml, json, traceback
from datetime import datetime
from roboflow import Roboflow

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY  = 'CXCNe93GGPiEeF2XrcYE'
DST_WS   = 'fire-cjxu1'
DST_PROJ = 'find-glioma-and-brainscope-ai'
BASE_DIR = r'C:\Users\Acer NItro\Desktop\claude work\brainscope-ai\datasets'
LOG_FILE = r'C:\Users\Acer NItro\Desktop\claude work\brainscope-ai\overnight_log.txt'
DONE_FILE= r'C:\Users\Acer NItro\Desktop\claude work\brainscope-ai\uploaded_files.json'

# ── Class name normaliser ────────────────────────────────────────────────────
# Maps raw class names from any dataset → project class name
NAME_MAP = {
    "glioma":       "glioma",
    "meningioma":   "meningioma",
    "pituitary":    "pituitary",
    "no tumor":     "normal",
    "no_tumor":     "normal",
    "notumor":      "normal",
    "normal":       "normal",
    "healthy":      "normal",
    "alzheimer":    "alzheimer",
    "alzheimers":   "alzheimer",
    "ad":           "alzheimer",
    "mci":          "alzheimer",
    "non demented": "normal",
    "nondemented":  "normal",
    "very mild dementia": "alzheimer",
    "mild dementia":      "alzheimer",
    "moderate dementia":  "alzheimer",
    "milddemented":       "alzheimer",
    "moderatedemented":   "alzheimer",
    "severedemented":     "alzheimer",
    "verymilddemented":   "alzheimer",
    "nondemented":        "normal",
    "mild_demented":      "alzheimer",
    "moderate_demented":  "alzheimer",
    "severe_demented":    "alzheimer",
    "very_mild_demented": "alzheimer",
    "non_demented":       "normal",
    "label0":             "glioma",
    "label1":             "glioma",
    "label2":             "glioma",
    "tumor":        "glioma",
    "brain tumor":  "glioma",
    "positive":     "glioma",
    "negative":     "normal",
    "bleeding":     "bleeding",
    "hemorrhage":   "bleeding",
    "ischemia":     "ischemia",
    "stroke":       "ischemia",
}

# ── Datasets: (workspace, project, version) ──────────────────────────────────
SOURCES = [
    ("ali-rostami",  "labeled-mri-brain-tumor-dataset", 1),
    ("roboflow-100", "brain-tumor-m2pbp",               2),
    ("astrid",       "alzheimer-s-disease",             3),
    ("fcis-oudzj",   "alzheimer-s-detection",           2),
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_done():
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE) as f:
            return set(json.load(f))
    return set()

def save_done(done_set):
    with open(DONE_FILE, "w") as f:
        json.dump(list(done_set), f)

def build_labelmap(yaml_path):
    """Read data.yaml and return {class_id: project_class_name}."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    labelmap = {}
    for i, raw in enumerate(names):
        key = raw.lower().strip()
        mapped = NAME_MAP.get(key)
        if mapped:
            labelmap[i] = mapped
        else:
            # fallback: keep original lowercase
            labelmap[i] = key
    log(f"  Class map: {dict(zip(names, labelmap.values()))}")
    return labelmap

# ── Main ─────────────────────────────────────────────────────────────────────
os.makedirs(BASE_DIR, exist_ok=True)
log("=" * 60)
log("OVERNIGHT BUILD STARTED")
log("=" * 60)

rf  = Roboflow(api_key=API_KEY)
dst = rf.workspace(DST_WS).project(DST_PROJ)
done = load_done()
log(f"Already uploaded (resuming): {len(done)} files")

total_uploaded = 0
total_skipped  = 0
total_failed   = 0

for ws, proj_name, ver in SOURCES:
    log(f"\n{'='*60}")
    log(f"DATASET: {ws}/{proj_name}  v{ver}")

    dl_dir = os.path.join(BASE_DIR, f"{ws}__{proj_name}")

    # ── Download (skip if already exists) ──
    yaml_path = os.path.join(dl_dir, "data.yaml")
    if os.path.exists(yaml_path):
        log(f"  Already downloaded — skipping download")
    else:
        try:
            log(f"  Downloading...")
            src = rf.workspace(ws).project(proj_name).version(ver)
            src.download("yolov8", location=dl_dir)
            log(f"  Download complete")
        except Exception as e:
            log(f"  DOWNLOAD FAILED: {e}")
            continue

    # ── Build class map ──
    if not os.path.exists(yaml_path):
        log(f"  No data.yaml found — skipping")
        continue
    labelmap = build_labelmap(yaml_path)

    # ── Upload ──
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dl_dir, split, "images")
        lbl_dir = os.path.join(dl_dir, split, "labels")
        if not os.path.exists(img_dir):
            continue

        images = (glob.glob(os.path.join(img_dir, "*.jpg")) +
                  glob.glob(os.path.join(img_dir, "*.png")) +
                  glob.glob(os.path.join(img_dir, "*.jpeg")))
        if not images:
            continue

        log(f"  [{split}] {len(images)} images found")

        for img_path in images:
            img_id = f"{proj_name}|{split}|{os.path.basename(img_path)}"

            if img_id in done:
                total_skipped += 1
                continue

            stem     = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")

            for attempt in range(3):
                try:
                    kwargs = dict(split=split, num_retry_uploads=2,
                                  batch_name=f"overnight_{proj_name}")
                    if os.path.exists(lbl_path):
                        kwargs["annotation_path"]     = lbl_path
                        kwargs["annotation_labelmap"] = labelmap

                    dst.upload(image_path=img_path, **kwargs)
                    done.add(img_id)
                    total_uploaded += 1

                    if total_uploaded % 200 == 0:
                        save_done(done)
                        log(f"  ✓ {total_uploaded} uploaded | {total_failed} failed | {total_skipped} skipped")

                    time.sleep(0.1)
                    break

                except Exception as e:
                    if attempt == 2:
                        total_failed += 1
                        if total_failed <= 10:
                            log(f"  FAIL: {os.path.basename(img_path)} — {e}")
                    else:
                        time.sleep(2)

save_done(done)

log(f"\n{'='*60}")
log(f"UPLOAD DONE: {total_uploaded} uploaded | {total_failed} failed | {total_skipped} skipped")
log('='*60)

# ── Auto generate version + train ────────────────────────────────────────────
log("\nGenerating new dataset version...")
try:
    new_ver = dst.generate_version(settings={
        "augmentation": {
            "bbFlipX":    {"percent": 0.5},
            "brightness": {"percent": 0.15, "type": "static"},
            "rotation":   {"degrees": 15},
            "crop":       {"min": 0.0, "max": 0.2}
        },
        "preprocessing": {
            "auto-orient": True,
            "resize": {"width": 640, "height": 640, "format": "Stretch to"}
        }
    })
    log(f"New version: v{new_ver.version}")
    log("Starting training...")
    new_ver.train(speed="fast")
    log("TRAINING STARTED — check Roboflow dashboard in the morning!")
except Exception as e:
    log(f"Auto-train failed: {e}")
    log(">>> MANUAL: Roboflow → project → Generate Version → Train")

log("\nDONE. Good morning! Check overnight_log.txt for results.")
