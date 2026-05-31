import base64, requests, json

API_KEY = 'CXCNe93GGPiEeF2XrcYE'
IMG = r'C:\Users\Acer NItro\Desktop\claude work\brainscope-ai\test_images\test\glioma\gg-332-_jpg.rf.918f85e684d806a0045ebf409f34d042.jpg'

with open(IMG, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

for ver in [2, 5]:
    url = f'https://detect.roboflow.com/find-glioma-and-brainscope-ai/{ver}?api_key={API_KEY}'
    resp = requests.post(url, data=img_b64, headers={'Content-Type': 'application/x-www-form-urlencoded'})
    data = resp.json()
    if resp.status_code == 200:
        preds = data.get('predictions', [])
        print(f'v{ver} OK — {len(preds)} predictions')
        for p in preds:
            print(f'  class={p["class"]} class_id={p.get("class_id","?")} conf={p["confidence"]:.2f}')
    else:
        print(f'v{ver} {resp.status_code} — {data.get("message", "")}')
