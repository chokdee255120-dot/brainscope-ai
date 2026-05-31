import base64, requests, os

API_KEY = 'CXCNe93GGPiEeF2XrcYE'
BASE = r'C:\Users\Acer NItro\Desktop\claude work\brainscope-ai\test_images\test'

tests = {
    'glioma':   os.path.join(BASE, 'glioma',    'gg-332-_jpg.rf.918f85e684d806a0045ebf409f34d042.jpg'),
    'bleeding': os.path.join(BASE, 'bleeding',  '10146_png.rf.a7ac294461c933405765297c6480fdda.jpg'),
    'ischemia': os.path.join(BASE, 'ischemia',  next(iter(os.listdir(os.path.join(BASE,'ischemia'))))),
    'normal':   os.path.join(BASE, 'normal',    next(iter(os.listdir(os.path.join(BASE,'normal'))))),
}

print('=== v4 class ID test ===')
for label, path in tests.items():
    if not os.path.exists(path):
        print(f'[{label}] file not found'); continue
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    r = requests.post(
        f'https://detect.roboflow.com/find-glioma-and-brainscope-ai/4?api_key={API_KEY}',
        data=b64, headers={'Content-Type': 'application/x-www-form-urlencoded'})
    preds = r.json().get('predictions', [])
    if preds:
        for p in preds:
            print(f'  [{label}] class="{p["class"]}" id={p.get("class_id","?")} conf={p["confidence"]:.2f}')
    else:
        print(f'  [{label}] no detection')
