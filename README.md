# GI Endoscopy Classifier

A portfolio project demonstrating end-to-end ML engineering: local model training with PyTorch, a FastAPI inference backend, and a live interactive demo deployed on Google Cloud Run.

**Model**: EfficientNet-B0 fine-tuned on [Kvasir-v2](https://datasets.simula.no/kvasir/)  
**Task**: 8-class GI endoscopy image classification (polyps, esophagitis, ulcerative colitis, and normal findings)  
**Dataset**: ~8,000 labeled colonoscopy/endoscopy images, freely available for research

---

## Project Structure

```
├── training/           # Local model training
│   ├── download_dataset.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── export_samples.py
├── app/                # FastAPI app deployed to Cloud Run
│   ├── main.py
│   ├── inference.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── static/         # Frontend (vanilla HTML/CSS/JS)
│       ├── index.html
│       ├── style.css
│       ├── app.js
│       └── samples/    # Demo images (populated by export_samples.py)
├── models/             # Trained weights — gitignored, lives locally
├── data/               # Dataset — gitignored, lives locally
└── deploy/
    └── deploy.sh
```

---

## Quickstart

### 1 — Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
```

### 2 — Download the dataset (~400 MB)

```bash
cd training
python download_dataset.py --dest ../data
```

### 3 — Train the model

```bash
# ~30 min on Apple Silicon MPS, ~2 h on CPU
python train.py --num-workers 0
```

> **macOS note**: `--num-workers 0` avoids a deadlock between MPS and Python multiprocessing. Key flags: `--epochs 15`, `--unfreeze-epoch 5`, `--batch-size 32`.

Weights are saved to `models/best_model.pt`.

### 4 — Evaluate and save metrics

```bash
python evaluate_metrics.py   # writes app/static/metrics.json
```

Computes accuracy, macro F1, weighted F1, per-class precision/recall/F1/AUC-ROC, and an 8×8 confusion matrix on the validation set. The JSON is served as a static file and rendered in the app's Model Performance section.

### 5 — Export sample images for the demo gallery

```bash
python export_samples.py   # writes 8 images/class → app/static/samples/
```

### 6 — Run the app locally

```bash
cd app
pip install -r requirements.txt
MODEL_PATH=../models/best_model.pt uvicorn main:app --reload
# Open http://localhost:8000
```

---

## Deploy to Google Cloud Run

### Prerequisites

```bash
gcloud auth login
gcloud auth configure-docker
gcloud services enable run.googleapis.com containerregistry.googleapis.com
```

### Deploy

```bash
GCP_PROJECT_ID=your-project-id ./deploy/deploy.sh
```

The script:
1. Copies `models/best_model.pt` into `app/model/`
2. Builds a CPU-only Docker image (~1.2 GB, no CUDA)
3. Pushes to Google Container Registry
4. Deploys to Cloud Run (1 vCPU, 1 GB RAM, scales to zero)

---

## Classes

| Class | Description |
|---|---|
| **polyps** | Abnormal tissue growths — potential precancerous lesions |
| **dyed-lifted-polyps** | Polyp injected and lifted for endoscopic removal |
| **dyed-resection-margins** | Margins after removal, dyed to verify completeness |
| **esophagitis** | Inflammation of the esophageal lining |
| **ulcerative-colitis** | IBD causing ulceration of the colon |
| **normal-cecum** | Healthy cecum — junction of small and large intestine |
| **normal-pylorus** | Healthy pyloric valve between stomach and small intestine |
| **normal-z-line** | Normal esophagus–stomach boundary |

---

## Architecture

```
User browser
    │  GET /           → index.html
    │  GET /samples    → gallery image URLs
    │  POST /predict   → { predictions: [{class, probability}] }
    ▼
FastAPI (Cloud Run)
    │  lifespan: loads EfficientNet-B0 weights at startup
    │  /predict: PIL decode → normalize → model forward → softmax
    ▼
EfficientNet-B0 (CPU, ~20 MB weights)
```

## License

Dataset: [Kvasir-v2 license](https://datasets.simula.no/kvasir/) — free for research use.  
Code: MIT.
