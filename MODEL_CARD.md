# Model Card — GI Endoscopy Classifier

## Model Details

| Field | Value |
|---|---|
| Architecture | EfficientNet-B0 |
| Framework | PyTorch 2.2 |
| Task | Multi-class image classification (8 classes) |
| Input | RGB image, resized/center-cropped to 224 × 224 |
| Output | Softmax probability distribution over 8 GI endoscopy classes |
| Parameters | ~5.3 M (pretrained backbone) + 8-way linear head |
| Training hardware | Apple M-series (MPS), ~12 min for 15 epochs |

## Intended Use

This model was built as a **portfolio demonstration** of end-to-end ML engineering on a medical imaging dataset. It is not validated for clinical use and must not be used to inform patient care decisions.

Appropriate uses:
- Demonstrating transfer learning and fine-tuning on a small medical dataset
- Exploring Grad-CAM explainability for image classification
- Benchmarking inference serving on Cloud Run

## Training Data

**Dataset:** [Kvasir-v2](https://datasets.simula.no/kvasir/)  
**Source:** Simula Research Laboratory  
**Size:** 8,000 images — 1,000 per class, perfectly balanced  
**Split:** 80% train / 20% validation, stratified, random seed 42

| Class | Description |
|---|---|
| dyed-lifted-polyps | Polyps lifted with dye injection prior to resection |
| dyed-resection-margins | Post-resection margins stained for completeness assessment |
| esophagitis | Inflammation of the esophageal lining |
| normal-cecum | Healthy cecum (junction of small and large intestine) |
| normal-pylorus | Healthy pyloric valve (stomach exit) |
| normal-z-line | Healthy gastroesophageal junction |
| polyps | Colonic polyps, potential cancer precursors |
| ulcerative-colitis | Inflammatory bowel disease of the colon |

**Preprocessing (train):** RandomResizedCrop(224), RandomHorizontalFlip, RandomVerticalFlip, ColorJitter(0.2, 0.2, 0.2, 0.1), ToTensor, ImageNet normalize  
**Preprocessing (val):** Resize(256), CenterCrop(224), ToTensor, ImageNet normalize

## Training Procedure

Two-phase transfer learning from ImageNet-pretrained weights:

**Phase 1 — backbone frozen (epochs 1–5)**
- Only the classification head is trained
- Learning rate: 1e-3, cosine annealing
- Optimizer: AdamW, weight decay 1e-4
- Batch size: 32

**Phase 2 — full fine-tune (epochs 6–15)**
- All layers unfrozen
- Learning rate: 1e-4, cosine annealing
- Same optimizer and batch size

Best checkpoint selected by lowest validation loss.

### Training curve

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 1.0918 | 66.3% | 0.5942 | 81.3% |
| 5 | — | — | — | — |
| 10 | — | — | — | — |
| 15 | 0.2353 | 90.7% | 0.1982 | 92.7% |

## Evaluation Results

Evaluated on the held-out 20% validation split (1,600 images).

| Metric | Value |
|---|---|
| Accuracy | **92.9%** |
| Macro F1 | **92.9%** |
| Weighted F1 | **92.9%** |

### Per-class results

| Class | Precision | Recall | F1 | AUC-ROC | Support |
|---|---|---|---|---|---|
| dyed-lifted-polyps | 89.4% | 95.9% | 92.5% | 0.996 | 219 |
| dyed-resection-margins | 95.9% | 89.5% | 92.6% | 0.996 | 209 |
| esophagitis | 91.6% | 80.8% | 85.9% | 0.991 | 203 |
| normal-cecum | 94.4% | 100.0% | 97.1% | 0.999 | 201 |
| normal-pylorus | 97.8% | 100.0% | 98.9% | 1.000 | 177 |
| normal-z-line | 80.7% | 91.6% | 85.8% | 0.990 | 178 |
| polyps | 97.4% | 92.6% | 94.9% | 0.996 | 202 |
| ulcerative-colitis | 97.5% | 93.8% | 95.7% | 0.998 | 211 |

**Weakest classes:** esophagitis and normal-z-line (F1 ~85.8–85.9%), likely due to visual similarity at the gastroesophageal junction.  
**Strongest classes:** normal-pylorus and normal-cecum (F1 ~97–99%), which have highly distinctive appearances.

## Limitations

**Equipment color bias.** All images were captured at a single hospital with the same endoscopy system. The model may have learned color signatures from the camera hardware rather than tissue morphology alone. Performance on images from different equipment is unknown.

**Potential patient leakage.** Kvasir-v2 does not publish patient IDs. If multiple images from the same patient appear in both train and val, validation accuracy is partially inflated. A proper clinical split would be stratified by patient, not by image.

**Idealized class balance.** Real colonoscopy datasets are heavily imbalanced (polyps appear in ~20–30% of procedures). The 1,000-images-per-class balance does not reflect clinical prevalence.

**Classification only — no localization.** The model outputs a single class label for the whole image. It cannot indicate where in the frame a finding is located. Global average pooling discards all spatial information before the classifier.

**Not clinically validated.** This model has not been evaluated on multi-site data, has not undergone regulatory review, and must not be used in any clinical workflow.

## Out-of-Scope Uses

- Clinical diagnosis or triage
- Screening in any patient population
- Deployment without re-evaluation on target-site data and equipment
- Any application where a false negative has patient safety implications

## Citation

If you use the Kvasir-v2 dataset, please cite the original paper:

```
Pogorelov et al. (2017). KVASIR: A Multi-Class Image Dataset for Computer Aided
Gastrointestinal Disease Detection. In Proceedings of the 8th ACM on Multimedia
Systems Conference (MMSys'17). ACM, New York, NY, USA.
```
