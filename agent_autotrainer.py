"""
AutoTrainer Agent — BrainScope AI
Monitors model performance, compares versions,
auto-deploys the best model, and updates the frontend.
"""
import os, json, time, subprocess, requests, base64
from datetime import datetime
from roboflow import Roboflow

API_KEY  = 'CXCNe93GGPiEeF2XrcYE'
BASE_DIR = r'C:\Users\Acer NItro\Desktop\claude work\brainscope-ai'
LOG_FILE = os.path.join(BASE_DIR, 'autotrainer_log.txt')
STATE_FILE = os.path.join(BASE_DIR, 'autotrainer_state.json')

TEST_IMAGES = {
    'glioma':    os.path.join(BASE_DIR, r'test_images\test\glioma'),
    'bleeding':  os.path.join(BASE_DIR, r'test_images\test\bleeding'),
    'ischemia':  os.path.join(BASE_DIR, r'test_images\test\ischemia'),
    'normal':    os.path.join(BASE_DIR, r'test_images\test\normal'),
    'meningioma':os.path.join(BASE_DIR, r'test_images\test\meningioma'),
    'pituitary': os.path.join(BASE_DIR, r'test_images\test\pituitary'),
}

FRONTEND_FILE = os.path.join(BASE_DIR, 'frontend', 'index.html')

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {'best_version': 4, 'best_map': 96.0, 'deployed_version': 4}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def test_model_accuracy(version, n_per_class=5):
    """Test model accuracy on local test images. Returns per-class accuracy."""
    import glob
    results = {}
    total_correct = 0
    total_tested = 0

    for cls, folder in TEST_IMAGES.items():
        if not os.path.exists(folder):
            continue
        images = glob.glob(os.path.join(folder, '*.jpg'))[:n_per_class]
        correct = 0
        for img_path in images:
            with open(img_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            try:
                url = f'https://detect.roboflow.com/find-glioma-and-brainscope-ai/{version}?api_key={API_KEY}'
                r = requests.post(url, data=b64,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=15)
                preds = r.json().get('predictions', [])
                if preds:
                    top = max(preds, key=lambda p: p['confidence'])
                    pred_class = (top.get('class') or '').lower()
                    if pred_class == cls or pred_class == cls.replace('_', ''):
                        correct += 1
                time.sleep(0.1)
            except Exception:
                pass

        acc = correct / len(images) * 100 if images else 0
        results[cls] = {'correct': correct, 'total': len(images), 'acc': round(acc, 1)}
        total_correct += correct
        total_tested += len(images)

    overall = total_correct / total_tested * 100 if total_tested else 0
    results['_overall'] = round(overall, 1)
    return results

def get_latest_version():
    """Get the latest trained version from Roboflow."""
    rf = Roboflow(api_key=API_KEY)
    proj = rf.workspace('fire-cjxu1').project('find-glioma-and-brainscope-ai')
    versions = proj.versions()
    trained = []
    for v in versions:
        try:
            model = v.model
            if model and model.get('id'):
                trained.append(v.version)
        except Exception:
            pass
    return max(trained) if trained else None

def update_frontend(version):
    """Update frontend INFER_URL to new model version."""
    import re
    with open(FRONTEND_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = re.sub(
        r"find-glioma-and-brainscope-ai/\d+",
        f"find-glioma-and-brainscope-ai/{version}",
        content
    )

    with open(FRONTEND_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)

    # Also update root index.html
    root_html = os.path.join(BASE_DIR, 'index.html')
    with open(root_html, 'w', encoding='utf-8') as f:
        f.write(new_content)

    log(f'Frontend updated to v{version}')

def deploy_if_better():
    """Compare latest version with deployed version, deploy if better."""
    state = load_state()
    log(f'Current best: v{state["best_version"]} ({state["best_map"]}% mAP)')
    log(f'Currently deployed: v{state["deployed_version"]}')

    latest = get_latest_version()
    if not latest:
        log('No trained versions found')
        return

    if latest <= state['best_version']:
        log(f'Latest v{latest} is not newer than best v{state["best_version"]}')
        return

    log(f'Testing new v{latest}...')
    new_results = test_model_accuracy(latest)
    new_overall = new_results.get('_overall', 0)

    log(f'v{latest} overall accuracy: {new_overall}%')
    for cls, r in new_results.items():
        if cls != '_overall':
            log(f'  {cls}: {r["acc"]}% ({r["correct"]}/{r["total"]})')

    if new_overall > state['best_map']:
        log(f'v{latest} ({new_overall}%) > v{state["best_version"]} ({state["best_map"]}%) — DEPLOYING')
        update_frontend(latest)

        # Git commit and push
        try:
            subprocess.run(['git', '-C', BASE_DIR, 'add', 'frontend/index.html', 'index.html'], check=True)
            subprocess.run(['git', '-C', BASE_DIR, 'commit', '-m',
                f'deploy: auto-update to v{latest} ({new_overall}% accuracy)'], check=True)
            subprocess.run(['git', '-C', BASE_DIR, 'push', 'origin', 'main'], check=True)
            log('Pushed to GitHub Pages')
        except Exception as e:
            log(f'Git push failed: {e}')

        state['best_version'] = latest
        state['best_map'] = new_overall
        state['deployed_version'] = latest
        save_state(state)
        log(f'Deployed v{latest} successfully')
    else:
        log(f'v{latest} ({new_overall}%) did NOT beat v{state["best_version"]} ({state["best_map"]}%) — keeping current')

def check_v6_training():
    """Check if v6 training is complete."""
    try:
        rf = Roboflow(api_key=API_KEY)
        proj = rf.workspace('fire-cjxu1').project('find-glioma-and-brainscope-ai')
        v6 = proj.version(6)
        model = v6.model
        if model and model.get('id'):
            # Try a test inference
            test_img = None
            for folder in TEST_IMAGES.values():
                import glob
                imgs = glob.glob(os.path.join(folder, '*.jpg'))
                if imgs:
                    test_img = imgs[0]
                    break
            if test_img:
                with open(test_img, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
                url = f'https://detect.roboflow.com/find-glioma-and-brainscope-ai/6?api_key={API_KEY}'
                r = requests.post(url, data=b64,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=15)
                if r.status_code == 200:
                    return True
    except Exception:
        pass
    return False

if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'

    if cmd == 'deploy':
        log('Running deploy check...')
        deploy_if_better()

    elif cmd == 'test':
        ver = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        log(f'Testing v{ver}...')
        results = test_model_accuracy(ver, n_per_class=3)
        for cls, r in results.items():
            print(f'  {cls}: {r}')

    elif cmd == 'check-v6':
        ready = check_v6_training()
        print('v6 ready:', ready)

    elif cmd == 'status':
        state = load_state()
        latest = get_latest_version()
        print(f'Best version: v{state["best_version"]} ({state["best_map"]}% mAP)')
        print(f'Deployed: v{state["deployed_version"]}')
        print(f'Latest trained: v{latest}')
        print('Run with "deploy" to auto-deploy best version')
        print('Run with "check-v6" to check if v6 training is done')
