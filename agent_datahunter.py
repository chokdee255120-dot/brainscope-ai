"""
DataHunter Agent — BrainScope AI
Searches Roboflow Universe for brain imaging datasets,
prioritises underrepresented classes (bleeding, ischemia),
downloads and prepares them for v6+.
"""
import os, json, time
from datetime import datetime
from roboflow import Roboflow

API_KEY   = 'CXCNe93GGPiEeF2XrcYE'
BASE_DIR  = r'C:\Users\Acer NItro\Desktop\claude work\brainscope-ai'
HUNT_LOG  = os.path.join(BASE_DIR, 'datahunter_log.txt')
HUNT_RESULTS = os.path.join(BASE_DIR, 'datahunter_results.json')

# Classes that need more data (under 500 images in current dataset)
PRIORITY_CLASSES = ['bleeding', 'hemorrhage', 'ischemia', 'stroke']

# Search queries for Roboflow Universe
SEARCH_QUERIES = [
    'brain hemorrhage CT',
    'intracranial bleeding',
    'brain stroke detection',
    'ischemic stroke CT',
    'brain CT scan detection',
    'brain MRI tumor detection',
]

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(HUNT_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def check_dataset_quality(workspace, project_name):
    """Check if a public Roboflow dataset is accessible and get details."""
    try:
        rf = Roboflow(api_key=API_KEY)
        proj = rf.workspace(workspace).project(project_name)
        versions = proj.versions()
        if not versions:
            return None
        v = versions[0]
        return {
            'workspace': workspace,
            'project': project_name,
            'version': v.version,
            'images': v.images,
            'type': getattr(proj, 'type', 'unknown'),
        }
    except Exception as e:
        return None

def search_universe():
    """Search Roboflow Universe for relevant datasets."""
    log('=' * 60)
    log('DataHunter Agent Starting')
    log('Priority classes: ' + ', '.join(PRIORITY_CLASSES))
    log('=' * 60)

    # Known high-quality public datasets for underrepresented classes
    KNOWN_DATASETS = [
        # Hemorrhage/Bleeding
        ('brain-hemorrhage', 'brain-hemorrhage-detection', 'hemorrhage/bleeding'),
        ('medical-ai', 'intracranial-hemorrhage', 'hemorrhage/bleeding'),
        ('roboflow-100', 'brain-tumor-m2pbp', 'tumor (check classes)'),
        # Stroke/Ischemia
        ('stroke-detection', 'brain-stroke', 'stroke/ischemia'),
        ('medical-imaging', 'brain-ct-stroke', 'stroke/ischemia'),
        # General brain CT
        ('diagnoct', 'brain-djpb5', 'original dataset - all classes'),
        ('brain-mri', 'brain-tumor-mri', 'tumor classes'),
    ]

    results = []
    log(f'\nChecking {len(KNOWN_DATASETS)} known datasets...')

    for ws, proj, desc in KNOWN_DATASETS:
        log(f'  Checking {ws}/{proj} ({desc})...')
        info = check_dataset_quality(ws, proj)
        if info:
            log(f'  FOUND: {info["images"]} images, v{info["version"]}, type={info["type"]}')
            results.append({**info, 'description': desc, 'status': 'available'})
        else:
            log(f'  Not accessible')
        time.sleep(0.5)

    # Save results
    with open(HUNT_RESULTS, 'w') as f:
        json.dump(results, f, indent=2)

    log(f'\n{"=" * 60}')
    log(f'Found {len(results)} accessible datasets')
    log(f'Results saved to: {HUNT_RESULTS}')

    # Recommend best datasets to download
    log('\nRECOMMENDATIONS:')
    for r in sorted(results, key=lambda x: x.get('images', 0), reverse=True):
        log(f'  {r["workspace"]}/{r["project"]} — {r["images"]} images — {r["description"]}')

    return results

def download_best_datasets(results, min_images=500):
    """Download the most valuable datasets."""
    from overnight_build import NAME_MAP

    rf = Roboflow(api_key=API_KEY)
    downloaded = []

    for r in results:
        if r.get('images', 0) < min_images:
            log(f'Skip {r["project"]} — too few images ({r["images"]})')
            continue

        out_dir = os.path.join(BASE_DIR, 'datasets', f'{r["workspace"]}__{r["project"]}')
        if os.path.exists(os.path.join(out_dir, 'data.yaml')):
            log(f'Already downloaded: {r["project"]}')
            downloaded.append((r, out_dir))
            continue

        log(f'Downloading {r["workspace"]}/{r["project"]} v{r["version"]}...')
        try:
            proj = rf.workspace(r['workspace']).project(r['project'])
            v = proj.version(r['version'])
            v.download('yolov8', location=out_dir)
            log(f'  Downloaded to {out_dir}')
            downloaded.append((r, out_dir))
        except Exception as e:
            log(f'  Download failed: {e}')

    return downloaded

if __name__ == '__main__':
    results = search_universe()
    print(f'\n{len(results)} datasets found. Check {HUNT_RESULTS} for details.')
    print('Run with --download flag to download the best ones.')

    import sys
    if '--download' in sys.argv:
        downloaded = download_best_datasets(results)
        print(f'Downloaded {len(downloaded)} datasets.')
        print('Run v6_build.py to CLAHE-process and upload them.')
