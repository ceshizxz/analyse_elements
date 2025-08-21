# Study of Early Printing Techniques Using Computer Vision and Machine Learning to Analyse Euclid’s Elementa

## 📌 Overview
End‑to‑end pipeline to **extract, preprocess, recall, compare, and visualise geometric figures** from historical prints.  
It mixes **classical CV** and **deep learning**, and includes **synthetic data generation** for detector training.

---

## 📂 Repository Structure (current)
```
.vscode/
  settings.json
images/
  pages/          # sample book page images (10 provided)
  backgrounds/    # 2 blank backgrounds for synthetic generation
  shapes/         # 5 geometric shape templates (PNG)
model/
  __init__.py
  extract.py
other/
  convert.py
  split_pdf.py
utils/
  compare.py
  generate_regions.py
  overlay.py
  preprocess.py
  rcnn.py
  recall_hu_orb.py
README.md
```
> All paths in the scripts are set via constants at the top of each file. Edit them for your environment before running.

---

## ⚙️ Installation
**Python 3.9+** and:
- torch, torchvision
- opencv-python
- scikit-image
- faiss-cpu *(optional, speeds up recall)*
- pymupdf (`fitz`)
- pillow, numpy, pandas, tqdm

```bash
pip install torch torchvision opencv-python scikit-image faiss-cpu pymupdf pillow numpy pandas tqdm
```

---

## 🚀 Workflow (from project root)

> **Note:** All scripts contain hard-coded example paths (e.g. `pdf_path`, `output_folder`, `input_dir`, `output_dir`, `model_path`).  
> **Before running, replace them with your own local paths** (e.g. `/path/to/your/book.pdf`, `/path/to/output/`).  

1. **PDF → Images (optional)**  
   If you have a book in PDF format, first convert it to page images:
   ```bash
   python other/split_pdf.py
   ```
   For quick testing, you can skip this step and directly use the **10 sample page images** in `images/pages/`.

2. **Synthetic Training Data (images + XML)**  
   ```bash
   python utils/generate_regions.py
   ```  
   Uses **2 backgrounds** (`images/backgrounds/`) and **5 shape templates** (`images/shapes/`)  
   to generate composite training images with XML annotations.

3. **Train Detector (Faster R‑CNN)**  
   ```bash
   python model/extract.py
   ```  
   Trains a 2‑class model (background + shape) using the generated dataset.  
   Saves model weights as `fasterrcnn_shapes.pth`.

4. **Detect on Pages & Crop Shapes**  
   ```bash
   python utils/rcnn.py
   ```  
   Runs the trained detector on sample pages (`images/pages/`):  
   - Saves detection visualisations (`pred_*`)  
   - Crops detected objects  
   - Outputs Pascal‑VOC XML annotations

5. **Preprocess Cropped Shapes**  
   ```bash
   python utils/preprocess.py
   ```  
   Applies binarisation, deskewing, resizing, edge and skeleton extraction.  
   Produces `_gray`, `_bin`, `_denoise`, `_deskew`, `_norm`, `_edge`, `_skeleton`.

6. **Stage A – Candidate Recall**  
   ```bash
   python utils/recall_hu_orb.py
   ```  
   Computes similarity candidates with Hu moments + ORB + Chamfer(+SSIM).  
   Saves results to `stageA_candidates.csv`.

7. **Stage B – Deep Similarity**  
   ```bash
   python utils/compare.py
   ```  
   Uses ResNet‑18 embeddings (L2‑normalised cosine similarity).  
   Saves `stageB_results.csv` and montage comparisons.

8. **Visual Verification (Overlay Triplets)**  
   ```bash
   python utils/overlay.py
   ```  
   Produces triplets: **[Image A | Image B | Overlay]** with colour‑coded edges.  

---

## 🧩 Utilities
- `other/convert.py` – simple B/W thresholding helper.
- Constants to edit in key scripts:
  - `utils/preprocess.py`: `input_dir`, `output_dir` …
  - `utils/recall_hu_orb.py`: `input_dir`, `output_csv`, weights, ECC params …
  - `utils/compare.py`: `candidates_csv`, `output_csv`, `montage_dir`, `expected_size` …
  - `utils/overlay.py`: `results_csv`, `candidates_csv`, `preprocessed_dir`, `overlay_dir` …
  - `model/extract.py`: training dirs, epochs, LR/scheduler …
  - `utils/rcnn.py`: model path, I/O folders, `confidence_threshold` …

---

## 🔍 Notes
- Keep image **canvas size consistent** (`target_size` in preprocessing).
- If FAISS is unavailable, recall falls back to **brute‑force** (still works).
- Use overlay triplets to manually filter borderline cases.
- Tune `tau_similarity` (Stage B) for your precision/recall needs.
```mermaid
flowchart LR
  A[PDF pages] --> B[Detector (rcnn.py)]
  B --> C[Crop shapes]
  C --> D[Preprocess (preprocess.py)]
  D --> E[Recall (recall_hu_orb.py)]
  E --> F[Deep compare (compare.py)]
  F --> G[Overlay triplets (overlay.py)]
```
