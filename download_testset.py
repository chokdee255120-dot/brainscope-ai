from roboflow import Roboflow
import os, shutil, pathlib

API_KEY   = "cBv6UPv7FRabrn1eN6iT"
WORKSPACE = "diagnoct"
PROJECT   = "brain-djpb5"
VERSION   = 3
OUT_DIR   = pathlib.Path(__file__).parent / "test_images"

print("Connecting to Roboflow...")
rf      = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(VERSION).download("folder", location=str(OUT_DIR))

print(f"\nDownloaded to: {OUT_DIR}")

# Count per class in test split
test_dir = OUT_DIR / "test"
if test_dir.exists():
    for cls_dir in sorted(test_dir.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*")))
            print(f"  {cls_dir.name}: {count} images")
