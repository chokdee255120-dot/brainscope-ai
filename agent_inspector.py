"""
Inspector Agent — BrainScope AI
Checks dataset quality: finds mislabeled images, class imbalance,
blur/quality issues, and generates a quality report.
"""
import os, json, glob, base64, time, requests
from datetime import datetime
import cv2
import numpy as np

API_KEY  = 'CXCNe93GGPiEeF2XrcYE'
BASE_DIR = r'C:\Users\Acer NItro\Desktop\claude work\brainscope-ai'
LOG_FILE = os.path.join(BASE_DIR, 'inspector_log.txt')
REPORT_FILE = os.path.join(BASE_DIR, 'inspector_report.json')

TEST_DIR = os.path.join(BASE_DIR, 'test_images', 'test')
CLASSES = ['bleeding', 'glioma', 'ischemia', 'meningioma', 'normal', 'pituitary']
MODEL_VERSION = 4

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def measure_blur(img_path):
    """Laplacian variance — lower = more blurry."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

def check_class_balance():
    """Report image count per class."""
    log('=== Class Balance Check ===')
    counts = {}
    for cls in CLASSES:
        folder = os.path.join(TEST_DIR, cls)
        imgs = glob.glob(os.path.join(folder, '*.jpg')) + glob.glob(os.path.join(folder, '*.png'))
        counts[cls] = len(imgs)
        log(f'  {cls}: {len(imgs)} images')

    min_cls = min(counts, key=counts.get)
    max_cls = max(counts, key=counts.get)
    ratio = counts[max_cls] / counts[min_cls] if counts[min_cls] > 0 else 999
    log(f'  Imbalance ratio: {ratio:.1f}x ({min_cls} vs {max_cls})')
    if ratio > 5:
        log(f'  WARNING: High imbalance! Consider adding more {min_cls} images.')
    return counts

def check_image_quality(n_per_class=20):
    """Check blur and quality of images per class."""
    log('=== Image Quality Check ===')
    quality_report = {}
    blurry_images = []

    for cls in CLASSES:
        folder = os.path.join(TEST_DIR, cls)
        imgs = glob.glob(os.path.join(folder, '*.jpg'))[:n_per_class]
        if not imgs:
            continue

        scores = [measure_blur(p) for p in imgs]
        avg_blur = np.mean(scores)
        min_blur = np.min(scores)
        blurry = [imgs[i] for i, s in enumerate(scores) if s < 50]

        quality_report[cls] = {
            'avg_sharpness': round(float(avg_blur), 1),
            'blurry_count': len(blurry),
            'blurry_files': [os.path.basename(b) for b in blurry]
        }

        status = 'OK' if avg_blur > 100 else 'WARN'
        log(f'  {cls}: avg_sharpness={avg_blur:.0f} [{status}], blurry={len(blurry)}/{len(imgs)}')
        blurry_images.extend(blurry)

    log(f'  Total blurry images found: {len(blurry_images)}')
    return quality_report, blurry_images

def check_mislabeled(n_per_class=10, confidence_threshold=0.5):
    """
    Use current model to check if images are labeled correctly.
    Flags images where model disagrees with label.
    """
    log(f'=== Mislabel Check (v{MODEL_VERSION}, {n_per_class} per class) ===')
    suspicious = []

    for cls in CLASSES:
        folder = os.path.join(TEST_DIR, cls)
        imgs = glob.glob(os.path.join(folder, '*.jpg'))[:n_per_class]
        wrong = 0

        for img_path in imgs:
            with open(img_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            try:
                url = f'https://detect.roboflow.com/find-glioma-and-brainscope-ai/{MODEL_VERSION}?api_key={API_KEY}'
                r = requests.post(url, data=b64,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=15)
                preds = r.json().get('predictions', [])

                if preds:
                    top = max(preds, key=lambda p: p['confidence'])
                    pred_cls = (top.get('class') or '').lower()
                    conf = top['confidence']

                    if pred_cls != cls and conf >= confidence_threshold:
                        suspicious.append({
                            'file': os.path.basename(img_path),
                            'true_label': cls,
                            'predicted': pred_cls,
                            'confidence': round(conf, 3)
                        })
                        wrong += 1
                time.sleep(0.1)
            except Exception:
                pass

        log(f'  {cls}: {wrong}/{len(imgs)} possible mislabels')

    if suspicious:
        log(f'\n  Suspicious images ({len(suspicious)}):')
        for s in suspicious[:10]:
            log(f'    {s["file"]}: labeled={s["true_label"]}, predicted={s["predicted"]} ({s["confidence"]:.0%})')
    else:
        log('  No obvious mislabeled images found')

    return suspicious

def run_full_inspection():
    log('=' * 60)
    log('Inspector Agent Running')
    log('=' * 60)

    report = {
        'timestamp': datetime.now().isoformat(),
        'model_version': MODEL_VERSION,
    }

    # 1. Class balance
    report['class_balance'] = check_class_balance()

    # 2. Image quality
    quality, blurry = check_image_quality(n_per_class=30)
    report['image_quality'] = quality
    report['blurry_count'] = len(blurry)

    # 3. Mislabeled check
    report['suspicious_labels'] = check_mislabeled(n_per_class=10)

    # 4. Summary
    log('\n=== INSPECTION SUMMARY ===')
    min_class = min(report['class_balance'], key=report['class_balance'].get)
    log(f'  Weakest class: {min_class} ({report["class_balance"][min_class]} images) — needs more data')
    log(f'  Blurry images: {report["blurry_count"]}')
    log(f'  Suspicious labels: {len(report["suspicious_labels"])}')

    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    log(f'\nFull report saved to: {REPORT_FILE}')

    return report

if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'full'

    if cmd == 'balance':
        check_class_balance()
    elif cmd == 'quality':
        check_image_quality()
    elif cmd == 'mislabeled':
        check_mislabeled()
    else:
        run_full_inspection()
