import base64, pathlib, httpx

BACKEND = "http://localhost:8000/api/analyze"
TEST_DIR = pathlib.Path(__file__).parent / "test_images" / "test"
CLASSES  = ["bleeding", "glioma", "ischemia", "meningioma", "normal", "pituitary"]

correct = 0
total   = 0

with httpx.Client(timeout=30) as client:
    for cls in CLASSES:
        imgs = list((TEST_DIR / cls).glob("*.jpg"))[:3]
        for p in imgs:
            b64 = base64.b64encode(p.read_bytes()).decode()
            r   = client.post(BACKEND, json={"image": b64})
            pred = r.json().get("top", "?")
            conf = r.json().get("confidence", 0)
            ok   = "OK" if pred == cls else "XX"
            print(f"{ok} [{cls}]  -> predicted: {pred}  ({conf*100:.1f}%)")
            if pred == cls: correct += 1
            total += 1

print(f"\nQuick test: {correct}/{total} correct ({correct/total*100:.0f}%)")
