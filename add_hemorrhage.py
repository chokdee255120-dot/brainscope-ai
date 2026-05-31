"""
Add brain-hemorrhage dataset (834 images) to current project.
CLAHE-process and upload in parallel with v6 build.
"""
import os, glob, base64, time, json, cv2, numpy as np
from datetime import datetime
from roboflow import Roboflow

API_KEY = 'CXCNe93GGPiEeF2XrcYE'
BASE_DIR = r'C:\Users\Acer NItro\Desktop\claude work\brainscope-ai'
LOG_FILE = os.path.join(BASE_DIR, 'hemorrhage_upload_log.txt')
DONE_FILE = os.path.join(BASE_DIR, 'hemorrhage_done.json')

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def apply_clahe(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return rgb

def load_done():
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE) as f:
            return set(json.load(f))
    return set()

def save_done(done):
    with open(DONE_FILE, 'w') as f:
        json.dump(list(done), f)

rf = Roboflow(api_key=API_KEY)

# Step 1: Download hemorrhage dataset
dl_dir = os.path.join(BASE_DIR, 'datasets', 'brain-hemorrhage__brain-hemorrhage-detection')
log('Downloading brain-hemorrhage dataset...')
if not os.path.exists(dl_dir) or not os.listdir(dl_dir):
    src = rf.workspace('brain-hemorrhage').project('brain-hemorrhage-detection').version(1)
    src.download('multiclass', location=dl_dir)
    log('Download complete')
else:
    log('Already downloaded, skipping')

# Classification dataset: no data.yaml, folders are class names
classes = [d for d in os.listdir(dl_dir)
           if os.path.isdir(os.path.join(dl_dir, d))
           and d not in ('train', 'valid', 'test')]
log(f'Classes found: {classes}')
# Map all classes to "bleeding"
labelmap = {cls: 'bleeding' for cls in classes}
log(f'Labelmap: {labelmap}')

# Step 2: CLAHE process + upload
dst = rf.workspace('fire-cjxu1').project('find-glioma-and-brainscope-ai')
done = load_done()
log(f'Already uploaded: {len(done)}')

total_uploaded = 0
total_failed = 0

clahe_dir = os.path.join(BASE_DIR, 'clahe_datasets', 'brain-hemorrhage__brain-hemorrhage-detection')

# Classification format: images are in class-named subfolders
# Create full-image YOLO bbox for each image
tmp_lbl_dir = os.path.join(BASE_DIR, 'tmp_hemorrhage_labels')
os.makedirs(tmp_lbl_dir, exist_ok=True)

# Get all images from all class subfolders (all are "bleeding")
all_images = []
for cls_folder in os.listdir(dl_dir):
    folder_path = os.path.join(dl_dir, cls_folder)
    if not os.path.isdir(folder_path):
        continue
    imgs = (glob.glob(os.path.join(folder_path, '*.jpg')) +
            glob.glob(os.path.join(folder_path, '*.png')))
    all_images.extend(imgs)

log(f'Total hemorrhage images: {len(all_images)}')

# Create a full-image bounding box label for each image
# YOLO format: class_id cx cy w h (normalized), full image = 0 0.5 0.5 1.0 1.0
BLEEDING_CLASS_ID = 0  # will use labelmap when uploading

for img_path in all_images:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(tmp_lbl_dir, stem + '.txt')
    if not os.path.exists(lbl_path):
        with open(lbl_path, 'w') as f:
            f.write(f'{BLEEDING_CLASS_ID} 0.5 0.5 1.0 1.0\n')

for img_path in all_images:
    img_id = f'hemorrhage|{os.path.basename(img_path)}'
    if img_id in done:
        continue

    # CLAHE process
    os.makedirs(clahe_dir, exist_ok=True)
    out_path = os.path.join(clahe_dir, os.path.basename(img_path))
    if not os.path.exists(out_path):
        enhanced = apply_clahe(img_path)
        if enhanced is not None:
            cv2.imwrite(out_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            out_path = img_path

    stem = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(tmp_lbl_dir, stem + '.txt')

    for attempt in range(3):
        try:
            dst.upload(
                image_path=out_path,
                annotation_path=lbl_path,
                annotation_labelmap={BLEEDING_CLASS_ID: 'bleeding'},
                split='train',
                num_retry_uploads=2,
                batch_name='hemorrhage_clahe'
            )
            done.add(img_id)
            total_uploaded += 1
            if total_uploaded % 100 == 0:
                save_done(done)
                log(f'Uploaded {total_uploaded} hemorrhage images')
            time.sleep(0.1)
            break
        except Exception as e:
            if attempt == 2:
                total_failed += 1
                if total_failed <= 5:
                    log(f'FAIL: {os.path.basename(img_path)} — {e}')
            else:
                time.sleep(2)

save_done(done)
log(f'DONE: {total_uploaded} uploaded, {total_failed} failed')
log('Hemorrhage data added to project. v6 version generation will include it.')
