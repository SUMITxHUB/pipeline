# 🏀 Basketball CV Pipeline

The core computer vision inference pipeline for the [NBA Player Tracker](https://github.com/SUMITxHUB/Basketball-system) project. Contains end-to-end notebooks for player detection, tracking, jersey number recognition, and team classification on basketball match footage.

---

## Notebooks

### `pipeline.ipynb` — Full inference pipeline
End-to-end pipeline that takes raw basketball video footage and runs:
- **Player detection** using fine-tuned RF-DETR-Small
- **Segmentation & tracking** across frames using SAM2
- **Jersey number OCR** using SmolVLM2
- **Team classification** via SigLIP embeddings + K-means clustering
- **Structured output export** with player IDs, positions, and team labels

### `basketball.ipynb` — Exploratory analysis & prototyping
Prototyping notebook used to experiment with individual model components, test detection accuracy, and validate the tracking logic before integrating into the full pipeline.

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | RF-DETR-Small (fine-tuned) |
| Segmentation & Tracking | SAM2 |
| OCR | SmolVLM2 |
| Visual Embeddings | SigLIP |
| Team Clustering | K-means |
| Language | Python |

---

## Related Repos

- **[rf-detr](https://github.com/SUMITxHUB/rf-detr)** — RF-DETR fine-tuning notebook and model training
- **[Basketball-system](https://github.com/SUMITxHUB/Basketball-system)** — Streamlit web interface for this pipeline

---

## Setup

```bash
git clone https://github.com/SUMITxHUB/pipeline.git
cd pipeline
pip install -r requirements.txt  # or install dependencies listed in notebook
jupyter notebook
```

Open `pipeline.ipynb` to run the full inference pipeline, or `basketball.ipynb` for individual component experiments.

---

## Results

- Detects and tracks **10+ players per frame** in live and recorded match footage
- Trained on **500+ labeled video frames**
- Achieved **15% improvement** in tracking accuracy through model parameter tuning
- Optimized for **Knicks vs Celtics 2025** broadcast footage

---

## Author

**Sumit Kumar** — [GitHub](https://github.com/SUMITxHUB) · [LinkedIn](https://linkedin.com/in/sumit16)
B.Tech CSE, KIIT University (2022–2026)
