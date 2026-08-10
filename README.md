# Multi-Task Banana Plant Analysis 

A multi-task deep learning system that simultaneously classifies **banana variety**, **disease**, and **nutrient deficiency** from a single field image — built for real-world deployment on edge devices in Assam, India.

> Research project exploring hard parameter sharing for agricultural computer vision under real deployment constraints: limited GPU memory, noisy field imagery, and the need to run offline on mobile hardware.

---

## Reasoning for Multi-Task Learning

Farmers in the field need fast, offline answers to three questions from one photo: *What variety is this? Is it diseased? Is it nutrient-deficient?* Training three separate models is wasteful — this project uses a **shared backbone with task-specific heads**, so the network learns general visual features once and specializes only at the final layers. This cuts inference cost for on-device deployment while letting related tasks (e.g. deficiency symptoms and disease symptoms often look visually similar) reinforce each other during training.

```
- **Sharing strategy**: Hard parameter sharing — forces the network to learn generalizable visual features across all three tasks through a shared bottleneck
- **Training**: Two-phase — frozen backbone feature extraction, then selective fine-tuning with BatchNorm layers kept frozen to avoid corrupting running statistics on small batches
- **Loss weighting**: Data-volume-based heuristic (inverse frequency of labeled samples per task), with uncertainty weighting (Kendall et al., CVPR 2018) under evaluation

## Dataset

Combines three sources into a unified label schema using an `UNKNOWN` sentinel for missing/partial labels, since not every source has annotations for every task:

| Source | Variety | Disease | Deficiency |
|---|:---:|:---:|:---:|
| PSFD-Musa (Assam) | ✅ | ✅ | ✅ |
| Karnataka Nutrient Deficiency Dataset | — | — | ✅ |
| Kaggle Banana Disease Recognition | — | ✅ | — |

This lets every image contribute to training even when it's only labeled for one or two of the three tasks — the loss function masks out unknown labels per sample rather than discarding the image.

*Note: raw datasets are not included in this repo — see [Setup](#setup) for sourcing them.*

## Engineering Highlights

A few of the harder problems solved along the way:

- **Dict-keyed sample weights failing silently in Keras 3.x** — `tf.data` pipelines with 3-tuple `(image, labels, weights)` structures needed weights passed positionally rather than by dict key to avoid a `KeyError` deep in the training loop
- **Dead-code bug in the variety head** — a `Dropout` layer was silently bypassing the head's `Dense` layer due to a tensor reference mix-up, invisible until inspecting per-head validation metrics
- **GPU memory management on a 4GB laptop GPU** — training a 3-headed model on EfficientNetB0 required `cuda_malloc_async`, reduced batch sizes, and image resolution tuning to fit within VRAM without sacrificing too much accuracy
- **Per-task Dense heads** — diagnosed that the disease and deficiency heads lacked dedicated task-specific layers before their output layer, likely limiting their capacity relative to the variety head

## Setup

```bash
git clone <repo-url>
cd banana-mtl
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dataset sources (not redistributed here — check each source's license before use):
- PSFD-Musa: [link]
- Karnataka Nutrient Deficiency Dataset: [link]
- Kaggle Banana Disease Recognition: [link]

Place downloaded datasets under `data/` (already gitignored) and run the preprocessing cells in `banana.ipynb` to generate the merged label CSV.
## Tech Stack

TensorFlow / Keras 3 · EfficientNetB0 · Jupyter · scikit-learn · WSL2 + CUDA

## References

- Kendall et al., *Multi-Task Learning Using Uncertainty to Weigh Losses*, CVPR 2018
- Sue-Chayapak et al., *A Novel Multi-Task Deep Learning for Plant Classification and Disease Diagnosis from Leaf Images*, ICPEI 2024
- Philippini et al., *Enhancing Crop Clinic Assistance: Empowering Plant Disease Detection through Multi-task Machine Learning*, LA-CCI 2023
---

*This is an active research project — architecture and results are still evolving. See commit history for progress.*
