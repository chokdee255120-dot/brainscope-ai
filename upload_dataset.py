"""
Download brain tumor dataset from Roboflow Universe
and upload to the brainscope-ai project.
"""
import os, glob, time
from roboflow import Roboflow

API_KEY      = 'CXCNe93GGPiEeF2XrcYE'
SRC_WS       = 'ali-rostami'
SRC_PROJ     = 'labeled-mri-brain-tumor-dataset'
SRC_VER      = 1
DST_WS       = 'fire-cjxu1'
DST_PROJ     = 'find-glioma-and-brainscope-ai'
DOWNLOAD_DIR = r'C:\Users\Acer NItro\Desktop\claude work\brainscope-ai\universe_dataset'

rf = Roboflow(api_key=API_KEY)

# ── 1. Download from Universe ──
print("=== Downloading from Roboflow Universe ===")
src = rf.workspace(SRC_WS).project(SRC_PROJ).version(SRC_VER)
dataset = src.download("yolov8", location=DOWNLOAD_DIR)
print(f"Downloaded to: {DOWNLOAD_DIR}")

# ── 2. Check classes in downloaded dataset ──
yaml_path = os.path.join(DOWNLOAD_DIR, "data.yaml")
if os.path.exists(yaml_path):
    with open(yaml_path) as f:
        print("\ndata.yaml contents:")
        print(f.read())

# ── 3. Upload images + annotations to destination project ──
print("\n=== Uploading to brainscope-ai project ===")
dst = rf.workspace(DST_WS).project(DST_PROJ)

uploaded = 0
failed   = 0

for split in ["train", "valid", "test"]:
    img_dir = os.path.join(DOWNLOAD_DIR, split, "images")
    lbl_dir = os.path.join(DOWNLOAD_DIR, split, "labels")
    if not os.path.exists(img_dir):
        continue

    images = glob.glob(os.path.join(img_dir, "*.jpg")) + \
             glob.glob(os.path.join(img_dir, "*.png")) + \
             glob.glob(os.path.join(img_dir, "*.jpeg"))

    print(f"\n[{split}] {len(images)} images")

    for img_path in images:
        lbl_path = os.path.join(lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        try:
            if os.path.exists(lbl_path):
                dst.upload(image_path=img_path, annotation_path=lbl_path,
                           annotation_labelmap=None, split=split,
                           num_retry_uploads=3, batch_name=f"universe_{split}")
            else:
                dst.upload(image_path=img_path, split=split,
                           num_retry_uploads=3, batch_name=f"universe_{split}")
            uploaded += 1
            if uploaded % 50 == 0:
                print(f"  Uploaded {uploaded} images so far...")
            time.sleep(0.1)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  WARN: {os.path.basename(img_path)} — {e}")

print(f"\n=== Done: {uploaded} uploaded, {failed} failed ===")
print("Go to Roboflow > Generate New Version > Train to start retraining.")
