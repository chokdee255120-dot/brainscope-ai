"""
Watch v6 training and auto-deploy when complete.
Run this once v6_build.py finishes uploading.
Polls Roboflow every 5 minutes until v6 is trained and ready.
"""
import time, sys, os, subprocess, requests, base64, glob
from datetime import datetime

API_KEY  = 'CXCNe93GGPiEeF2XrcYE'
BASE_DIR = r'C:\Users\Acer NItro\Desktop\claude work\brainscope-ai'
LOG = os.path.join(BASE_DIR, 'watch_v6_log.txt')
FRONTEND = os.path.join(BASE_DIR, 'frontend', 'index.html')

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def test_v6():
    """Return True if v6 API responds with 200."""
    test_img = None
    for folder in ['glioma', 'normal', 'meningioma']:
        imgs = glob.glob(os.path.join(BASE_DIR, 'test_images', 'test', folder, '*.jpg'))
        if imgs:
            test_img = imgs[0]
            break
    if not test_img:
        return False
    try:
        with open(test_img, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        r = requests.post(
            f'https://detect.roboflow.com/find-glioma-and-brainscope-ai/6?api_key={API_KEY}',
            data=b64, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=15)
        return r.status_code == 200
    except Exception:
        return False

def deploy_v6():
    """Update frontend to v6 and push to GitHub."""
    import re
    with open(FRONTEND, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update version URL
    new_content = re.sub(
        r"find-glioma-and-brainscope-ai/\d+",
        "find-glioma-and-brainscope-ai/6",
        content
    )
    # Update CLASS_MAP for v6 (alphabetical, 7 classes)
    new_content = re.sub(
        r"const CLASS_MAP = \{[^}]+\};",
        """const CLASS_MAP = {
    'alzheimer':'Alzheimer','bleeding':'Bleeding','glioma':'Glioma',
    'ischemia':'Ischemia','meningioma':'Meningioma','normal':'Normal','pituitary':'Pituitary',
    '0':'Alzheimer','1':'Bleeding','2':'Glioma',
    '3':'Ischemia','4':'Meningioma','5':'Normal','6':'Pituitary'
  };""",
        new_content
    )

    with open(FRONTEND, 'w', encoding='utf-8') as f:
        f.write(new_content)
    # Copy to root
    import shutil
    shutil.copy(FRONTEND, os.path.join(BASE_DIR, 'index.html'))
    log('Frontend updated to v6')

    try:
        subprocess.run(['git', '-C', BASE_DIR, 'add', 'frontend/index.html', 'index.html'], check=True)
        subprocess.run(['git', '-C', BASE_DIR, 'commit', '-m',
            'deploy: BrainScope AI v6 - CLAHE dataset, RF-DETR Large, 7 classes'], check=True)
        subprocess.run(['git', '-C', BASE_DIR, 'push', 'origin', 'main'], check=True)
        log('Deployed to GitHub Pages!')
        return True
    except Exception as e:
        log(f'Git push failed: {e}')
        return False

log('=== watch_v6.py started - polling every 5 min ===')
check_count = 0

while True:
    check_count += 1
    log(f'Check #{check_count}: testing v6 availability...')
    ready = test_v6()

    if ready:
        log('v6 is READY! Deploying...')
        success = deploy_v6()
        if success:
            log('SUCCESS: v6 deployed to GitHub Pages!')
            log('URL: https://chokdee255120-dot.github.io/brainscope-ai/')
        else:
            log('Deploy failed - run manually: python agent_autotrainer.py deploy')
        break
    else:
        log('v6 not ready yet, waiting 5 minutes...')
        time.sleep(300)

log('watch_v6.py done.')
