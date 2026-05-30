"""
BrainScope AI — Batch Test Script
Runs every image in test_images/test/ against the backend API and reports accuracy.
Requires: backend running on localhost:8000  (start_backend.bat)
"""

import base64, pathlib, httpx, json
from collections import defaultdict

BACKEND = "http://localhost:8000/api/analyze"
TEST_DIR = pathlib.Path(__file__).parent / "test_images" / "test"
CLASSES  = ["bleeding", "glioma", "ischemia", "meningioma", "normal", "pituitary"]


def encode(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def run():
    results = defaultdict(lambda: {"correct": 0, "total": 0, "wrong": []})
    overall = {"correct": 0, "total": 0}

    with httpx.Client(timeout=30) as client:
        for cls in CLASSES:
            cls_dir = TEST_DIR / cls
            if not cls_dir.exists():
                continue

            images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
            print(f"\n[{cls.upper()}] Testing {len(images)} images...")

            for img_path in images:
                try:
                    resp = client.post(BACKEND, json={"image": encode(img_path)})
                    resp.raise_for_status()
                    data = resp.json()

                    # Classification response: data["top"] is the predicted class
                    predicted = (data.get("top") or "").lower()
                    confidence = data.get("confidence", 0)

                    results[cls]["total"] += 1
                    overall["total"] += 1

                    if predicted == cls.lower():
                        results[cls]["correct"] += 1
                        overall["correct"] += 1
                    else:
                        results[cls]["wrong"].append({
                            "file": img_path.name,
                            "predicted": predicted,
                            "confidence": round(confidence * 100, 1)
                        })

                except Exception as e:
                    print(f"  ERROR {img_path.name}: {e}")

            acc = results[cls]["correct"] / results[cls]["total"] * 100 if results[cls]["total"] else 0
            print(f"  Accuracy: {results[cls]['correct']}/{results[cls]['total']} = {acc:.1f}%")

    # ── Summary ──
    print("\n" + "=" * 50)
    print("OVERALL RESULTS")
    print("=" * 50)
    for cls in CLASSES:
        r = results[cls]
        if r["total"] == 0:
            continue
        acc = r["correct"] / r["total"] * 100
        bar = "#" * int(acc / 5) + "." * (20 - int(acc / 5))
        print(f"  {cls:<12} {bar}  {acc:5.1f}%  ({r['correct']}/{r['total']})")

    total_acc = overall["correct"] / overall["total"] * 100 if overall["total"] else 0
    print(f"\n  Overall accuracy: {total_acc:.1f}%  ({overall['correct']}/{overall['total']})")

    # Save JSON report
    report_path = pathlib.Path(__file__).parent / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"overall_accuracy": round(total_acc, 2), "per_class": dict(results)}, f, indent=2, default=str)
    print(f"\n  Full report saved: {report_path}")


if __name__ == "__main__":
    run()
