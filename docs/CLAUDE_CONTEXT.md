# BrainScope AI — Claude Context File
> Read this first when starting a new terminal session. This file replaces lost memory.

---

## 1. Project Identity

| Key | Value |
|-----|-------|
| Project | BrainScope AI — Brain CT Scan Multi-Disease Detector |
| GitHub | https://github.com/chokdee255120-dot/brainscope-ai |
| GitHub Pages (live) | https://chokdee255120-dot.github.io/brainscope-ai/ |
| Roboflow workspace | `fire-cjxu1` |
| Roboflow project | `find-glioma-and-brainscope-ai` |
| Roboflow API key | `CXCNe93GGPiEeF2XrcYE` |
| Local path | `C:\Users\Acer NItro\Desktop\claude work\brainscope-ai` |
| Git config user | `chokdee255120-dot` / `totoolclaude@gmail.com` |

---

## 2. Detected Disease Classes (7)

| ID | Class | Notes |
|----|-------|-------|
| 0 | alzheimer | Added in v4+, from astrid + fcis datasets |
| 1 | bleeding | Underrepresented (~107 test images) |
| 2 | glioma | Primary tumor, well represented |
| 3 | ischemia | Underrepresented (~113 test images) |
| 4 | meningioma | Benign tumor |
| 5 | normal | No finding |
| 6 | pituitary | Pituitary gland tumor |

**Class mapping in frontend:** v4 returns string names (e.g. `"glioma"`), not numeric IDs. CLASS_MAP handles both.

---

## 3. File Structure

```
brainscope-ai/
├── frontend/index.html          # Main web app (ALWAYS copy to root index.html before push)
├── index.html                   # Root copy for GitHub Pages (must match frontend/)
├── backend/
│   ├── main.py                  # FastAPI server (serves index.html locally only)
│   ├── .env                     # ROBOFLOW_API_KEY, ROBOFLOW_MODEL
│   └── requirements.txt
├── docs/
│   ├── CLAUDE_CONTEXT.md        # THIS FILE — context for new Claude sessions
│   └── brainscope_report.html   # Full technical report for human presentation
├── agents/
│   ├── agent_datahunter.py      # Searches Roboflow Universe for new datasets
│   ├── agent_autotrainer.py     # Monitors + auto-deploys best model version
│   ├── agent_inspector.py       # Checks dataset quality, class balance, mislabels
│   └── (ReportWriter is embedded in frontend/index.html as JS)
├── v6_build.py                  # CLAHE preprocessing + upload + version gen + train
├── overnight_build.py           # Older overnight uploader (v4 era, has resume)
├── watch_v6.py                  # Polls Roboflow every 5min, auto-deploys v6 when ready
├── gen_v6_now.py                # One-shot: generate version + trigger training
├── add_hemorrhage.py            # Attempts to add hemorrhage-specific dataset
├── check_model.py               # Test API response for any version
├── test_v4.py                   # Test v4 class IDs with local images
├── datasets/                    # Downloaded Roboflow Universe datasets
│   ├── ali-rostami__labeled-mri-brain-tumor-dataset/   (2,443 images, YOLO format)
│   ├── astrid__alzheimer-s-disease/                     (3,262 images)
│   └── fcis-oudzj__alzheimer-s-detection/               (4,473 images)
├── clahe_datasets/              # CLAHE-processed versions of above (for v6)
├── test_images/test/            # Local test set, 6 class subfolders
│   └── {bleeding,glioma,ischemia,meningioma,normal,pituitary}/
├── uploaded_files.json          # Resume tracker for overnight_build.py
├── uploaded_clahe.json          # Resume tracker for v6_build.py
├── v6_log.txt                   # v6 build log (check for progress)
├── overnight_log.txt            # Overnight build log
└── autotrainer_state.json       # Tracks best deployed version + mAP
```

---

## 4. Model Version History

| Version | Date | Architecture | mAP@50 | Precision | Recall | F1 | Notes |
|---------|------|-------------|--------|-----------|--------|-----|-------|
| v2 | May 26 | RF-DETR Nano | 97.1% | 98.0% | 91.9% | 94.9% | Original, 6 classes |
| v3 | May 31 01:03 | RF-DETR Nano | 95.6% | 95.2% | 93.9% | 94.5% | Auto-generated overnight |
| v4 | May 31 02:02 | RF-DETR Nano | 96.0% | 96.6% | 93.9% | 95.2% | **Currently deployed** |
| v5 | May 31 13:38 | RF 3.0 Fast | 44.2% | 56.7% | 63.8% | 60.0% | Failed — wrong arch + label pollution |
| v6 | May 31 | RF-DETR Nano | pending | - | - | - | CLAHE+augmentation, training now |

**Currently deployed:** v4 at `https://detect.roboflow.com/find-glioma-and-brainscope-ai/4`

---

## 5. Frontend Architecture

The entire app is a **single-file SPA** (`frontend/index.html` ~1600 lines).

### Key JavaScript components:
```javascript
// API config (line ~1273)
const INFER_URL = 'https://detect.roboflow.com/find-glioma-and-brainscope-ai/4';
const ROBOFLOW_KEY = 'CXCNe93GGPiEeF2XrcYE';
const CLASS_MAP = { 'alzheimer':'Alzheimer', 'bleeding':'Bleeding', ... };

// DICOM loader (line ~1300+)
async function loadDicom(file) { /* dicom-parser → canvas → JPEG */ }

// Analysis (line ~1285+)
async function runAnalysis() { /* POST base64 → Roboflow API → displayResults */ }

// ReportWriter Agent (line ~1540+)
function generateReport() { /* predictions → urgency → specialist rec → HTML */ }
```

### CDN Dependencies:
- `https://unpkg.com/dicom-parser@1.8.21/dist/dicomParser.min.js` — DICOM parsing
- Google Fonts (Sarabun, IBM Plex Sans Thai)

### Supported input formats:
- JPG, PNG (standard)
- DICOM `.dcm` — parsed with dicom-parser, window/level auto-detected from header (defaults: WC=40, WW=80 for brain CT)

---

## 6. Agent System

### ReportWriter (embedded in frontend)
- Triggered by "Generate Report" button (visible after analysis)
- Maps predictions → `DISEASE_DB` (urgency, specialist, description, action)
- Urgency levels: `CRITICAL > HIGH > MODERATE > LOW > NONE`
- Outputs: HTML modal + Print/PDF via `window.print()`
- CRITICAL findings: bleeding → Neurosurgeon EMERGENCY, ischemia → Stroke Team EMERGENCY

### DataHunter (`agent_datahunter.py`)
```bash
python agent_datahunter.py           # Search only
python agent_datahunter.py --download # Search + download
```
- Checks 7 known Roboflow Universe datasets
- Prioritises: bleeding, hemorrhage, ischemia, stroke
- Saves results to `datahunter_results.json`

### AutoTrainer (`agent_autotrainer.py`)
```bash
python agent_autotrainer.py status      # Show current state
python agent_autotrainer.py test 6      # Test accuracy of v6
python agent_autotrainer.py deploy      # Compare versions, deploy if better
python agent_autotrainer.py check-v6   # Check if v6 is ready
```
- Compares new model vs `autotrainer_state.json` best
- Auto-updates `frontend/index.html` + `index.html`, commits, pushes

### Inspector (`agent_inspector.py`)
```bash
python agent_inspector.py full        # Full inspection
python agent_inspector.py balance     # Class count per folder
python agent_inspector.py quality     # Blur detection (Laplacian variance)
python agent_inspector.py mislabeled  # Model vs label disagreement check
```
- Laplacian variance < 50 = blurry
- Flags images where model confidence ≥ 50% but predicts wrong class

---

## 7. Data Pipeline

```
Raw Dataset Sources
    ↓
Download (Roboflow SDK / Kaggle)
    ↓
CLAHE Preprocessing (cv2.createCLAHE clipLimit=4.0, tileGridSize=8×8)
    ↓
Upload to Roboflow Project (fire-cjxu1/find-glioma-and-brainscope-ai)
    ↓
Generate Dataset Version (augmentation: flip50%, crop15%, rot15°, brightness, blur, noise)
    ↓
Train (RF-DETR Nano, speed=fast)
    ↓
Compare mAP with previous best (autotrainer_state.json)
    ↓
Auto-deploy if better → git push → GitHub Pages
```

---

## 8. How to Continue Work (New Session Checklist)

### A. Check v6 status
```bash
python agent_autotrainer.py check-v6
```
If ready:
```bash
python agent_autotrainer.py deploy
```

### B. Push any frontend change
```bash
# After editing frontend/index.html:
Copy-Item frontend/index.html index.html
git add index.html frontend/index.html
git commit -m "your message"
git push origin main
```

### C. Resume CLAHE upload (v6 dataset enhancement)
```bash
python v6_build.py   # Resumes from uploaded_clahe.json
```

### D. Run full dataset inspection
```bash
python agent_inspector.py full
```

### E. Start local backend (optional, for testing without GitHub Pages)
```bash
start_backend.bat    # Starts FastAPI on localhost:8000
```

---

## 9. Pending Work

| Priority | Task | Command/Action |
|----------|------|----------------|
| HIGH | Deploy v6 when training completes | `python agent_autotrainer.py deploy` |
| HIGH | Verify v6 class IDs match CLASS_MAP | Check API response after deploy |
| MEDIUM | Add more bleeding/ischemia images | Fix `add_hemorrhage.py` (SDK format issue) |
| MEDIUM | Run Inspector full report | `python agent_inspector.py full` |
| LOW | Train v7 with CLAHE dataset (uploading now) | After v6_build.py finishes |
| LOW | Improve ReportWriter with more disease info | Edit DISEASE_DB in frontend |

---

## 10. Known Issues

| Issue | Description | Status |
|-------|-------------|--------|
| label0/1/2 pollution | roboflow-100 uploaded with wrong class names | In Roboflow project, doesn't affect v4 much |
| bleeding/ischemia underrepresented | Only ~107/113 test images each | Needs more data |
| `add_hemorrhage.py` fails | Brain-hemorrhage dataset is multilabel-classification, SDK download broken | Workaround needed |
| RF-DETR Large blocked | Free plan only allows `speed='fast'` | Upgrade plan or use YOLOv11x locally |
| v6 CLAHE still uploading | Started 15:02 May 31, ~9h total | Running in background PID 35956 |
