"""
BrainScope AI — Dataset Builder
Downloads multiple brain imaging datasets from Roboflow Universe,
maps class names correctly, uploads to project, then triggers training.
"""
import os, glob, time, yaml
from roboflow import Roboflow

API_KEY = 'CXCNe93GGPiEeF2XrcYE'
DST_WS   = 'fire-cjxu1'
DST_PROJ = 'find-glioma-and-brainscope-ai'
BASE_DIR = r'C:\Users\Acer NItro\Desktop\claude work\brainscope-ai\datasets'

rf = Roboflow(api_key=API_KEY)

# ── Dataset sources + class label maps ──────────────────────────────────────
# Each entry: (workspace, project, version, labelmap)
# labelmap = {yolo_class_id: "project_class_name"}
# "No Tumor" and similar → mapped to "normal"
SOURCES = [
    # 2,443 images — Glioma / Meningioma / Normal / Pituitary
    ("ali-rostami", "labeled-mri-brain-tumor-dataset", 1,
     {0: "glioma", 1: "meningioma", 2: "normal", 3: "pituitary"}),

    # 9,900 images — brain tumor (roboflow-100)
    ("roboflow-100", "brain-tumor-m2pbp", 2,
     {0: "glioma"}),

    # ~1,364 images — Alzheimer's
    ("astrid", "alzheimer-s-disease", 2,
     {0: "alzheimer", 1: "alzheimer"}),

    # ~1,977 images — Alzheimer's (second source)
    ("fcis-oudzj", "alzheimer-s-detection", 2,
     {0: "alzheimer", 1: "alzheimer", 2: "normal"}),
]

os.makedirs(BASE_DIR, exist_ok=True)
dst = rf.workspace(DST_WS).project(DST_PROJ)

total_uploaded = 0
total_failed   = 0

for ws, proj_name, ver, labelmap in SOURCES:
    print(f"\n{'='*60}")
    print(f"Dataset: {ws}/{proj_name} v{ver}")
    print(f"Class map: {labelmap}")
    print('='*60)

    # ── Download ──
    dl_dir = os.path.join(BASE_DIR, f"{ws}__{proj_name}")
    try:
        src = rf.workspace(ws).project(proj_name).version(ver)
        src.download("yolov8", location=dl_dir)
        print(f"Downloaded to: {dl_dir}")
    except Exception as e:
        print(f"  SKIP — download failed: {e}")
        continue

    # ── Show classes from data.yaml ──
    yaml_path = os.path.join(dl_dir, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        print(f"  Classes in dataset: {data.get('names', [])}")

    # ── Upload each split ──
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

        print(f"\n  [{split}] uploading {len(images)} images...")

        for img_path in images:
            stem     = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")

            try:
                kwargs = dict(split=split, num_retry_uploads=3,
                              batch_name=f"{proj_name}_{split}")
                if os.path.exists(lbl_path):
                    kwargs["annotation_path"]     = lbl_path
                    kwargs["annotation_labelmap"] = labelmap

                dst.upload(image_path=img_path, **kwargs)
                total_uploaded += 1

                if total_uploaded % 100 == 0:
                    print(f"    ✓ {total_uploaded} total uploaded so far")
                time.sleep(0.08)

            except Exception as e:
                total_failed += 1
                if total_failed <= 3:
                    print(f"    WARN: {os.path.basename(img_path)} — {e}")

print(f"\n{'='*60}")
print(f"UPLOAD COMPLETE: {total_uploaded} uploaded, {total_failed} failed")
print('='*60)

# ── Try to generate new version and train ───────────────────────────────────
print("\nAttempting to generate new dataset version and start training...")
try:
    new_version = dst.generate_version(settings={
        "augmentation": {
            "bbFlipX": {"percent": 0.5},
            "bbFlipY": {"percent": 0},
            "brightness": {"percent": 0.15, "type": "static"},
            "rotation": {"degrees": 15}
        },
        "preprocessing": {
            "auto-orient": True,
            "resize": {"width": 640, "height": 640, "format": "Stretch to"}
        }
    })
    print(f"New version generated: v{new_version.version}")

    print("Starting training (YOLOv8n fast)...")
    new_version.train(speed="fast")
    print("Training started! Check Roboflow dashboard for progress.")

except Exception as e:
    print(f"Auto-train failed: {e}")
    print("\n>>> MANUAL STEP: Go to Roboflow → project → Generate Version → Train")
