# Text-to-Vehicle Retrieval

Find a specific vehicle in road CCTV footage using nothing but a natural-language description, for example *"a black SUV turning left at the intersection"*.

The system detects and tracks every vehicle in a video, embeds each track and your text query into a shared space, ranks the tracks by similarity, and renders a copy of the video with the best match highlighted.

- **Pretrained weights:** [Takayanagi-sensei/Text-to-Vehicle-Retrieval-Model](https://huggingface.co/Takayanagi-sensei/Text-to-Vehicle-Retrieval-Model) on Hugging Face
- **Task:** text-based vehicle retrieval / re-identification (the same task as AI City Challenge 2023, Track 2)
- **Intended use:** research and education (see [License and data restrictions](#license-and-data-restrictions))

---

## How it works

```
CCTV video ──► YOLO26 detection ──► ByteTrack tracking ──► one track per vehicle
                                                                  │
                                                       Visual encoder (ResNet-50)
                                                                  │
                                                                  ▼
Text query ──► CLIP text encoder + projection ──────►  shared 256-d embedding space
                                                                  │
                                                       cosine-similarity ranking
                                                                  │
                                              optional attribute re-ranking (color / type)
                                                                  │
                                                                  ▼
                                               highlighted output video + ranked list
```

### 1. Detection and tracking
[YOLO26](https://github.com/ultralytics/ultralytics) detects cars, motorcycles, buses and trucks in every frame. [ByteTrack](https://github.com/ifzhang/ByteTrack) (via `supervision`) links those detections across frames, so each physical vehicle becomes a single track with a stable ID. A track is the unit that gets matched against your text.

### 2. Visual encoder (vehicle Re-ID)
A ResNet-50 with a BNNeck head, pretrained on ImageNet and then trained for vehicle re-identification on [VeRi-776](https://github.com/JDAI-CV/VeRidataset) using identity classification plus batch-hard triplet loss. This teaches the backbone to tell visually similar vehicles apart, which a generic image model does not do well. The same backbone is then fine-tuned further in the joint stage below. Each track is represented by the averaged embedding of several sampled crops, which is more robust than relying on a single frame.

### 3. Text encoder
The text tower of CLIP (ViT-B/32) followed by a small trained projection head. The CLIP backbone stays frozen; only the projection is learned.

### 4. Joint training (contrastive)
Vehicle crops and their natural-language descriptions from CityFlow-NL are trained with a symmetric InfoNCE loss, the same objective CLIP uses. Matching (crop, description) pairs are pulled together in the shared space and non-matching pairs are pushed apart.

### 5. Attribute re-ranking (optional)
Pure embedding similarity can return a close but wrong vehicle. A lightweight second stage parses color and vehicle type out of the query (for example "white", "sedan") and checks them against zero-shot CLIP classification of each top candidate. Candidates whose attributes agree with the query get a score bonus. This grounds the ranking in attributes the model can actually see.

### 6. Rendering
The matched track is drawn with a thick red box and a label; all other vehicles are drawn as thin gray boxes. Detections are cached from the first pass, so rendering does not re-run the detector.

---

## Why it is useful

Camera footage has no searchable metadata. Finding one vehicle normally means a person scrubbing through hours of video. A text interface lets you describe what you remember and jump straight to candidates.

Example scenarios this kind of system is built for:

- **Traffic and incident review:** locate a vehicle from a witness description such as "a dark red SUV going straight through the intersection".
- **Smart-city and ITS research:** a baseline for cross-modal retrieval over traffic cameras.
- **Dataset exploration:** quickly search large amounts of road video by description instead of by timestamp.
- **Teaching and benchmarking:** a complete, readable reference pipeline covering detection, tracking, metric learning and contrastive vision-language training, with all the free-GPU workarounds documented.

This repository is a research prototype. See [Limitations](#limitations) before relying on it for anything beyond experimentation.

---

## Repository structure

```
├── src/
│   ├── models/
│   │   ├── reid_model.py            # ResNet-50 + BNNeck Re-ID backbone
│   │   └── joint_model.py           # visual / text projection heads
│   ├── data/
│   │   ├── veri_dataset.py          # VeRi-776 dataset + PK sampler
│   │   ├── cityflow_nl_dataset.py   # CityFlow-NL (crop, text) pairs
│   │   └── extract_crops.py         # storage-efficient crop extraction
│   ├── training/
│   │   ├── losses.py                # triplet and InfoNCE losses
│   │   ├── checkpoint_utils.py      # resume + keep-latest checkpointing
│   │   ├── train_reid.py            # Stage 1: Re-ID on VeRi-776
│   │   └── train_joint.py           # Stage 2: contrastive text-vehicle training
│   ├── pipeline/
│   │   ├── detect_track.py          # YOLO26 + ByteTrack
│   │   ├── attribute_rerank.py      # color/type parsing + CLIP zero-shot re-rank
│   │   └── inference.py             # end-to-end: video + query -> highlighted video
│   └── eval_metrics.py              # mAP/CMC and MRR/Recall@k
├── app/gradio_app.py                # interactive demo
├── scripts/extract_cityflow_crops.py
├── configs/config.yaml              # all paths and hyperparameters
├── notebooks/                       # original phase-by-phase Kaggle notebooks
├── notebooks_combined/              # 01_finetuning.ipynb and 02_inference.ipynb
├── docs/gpu_notes.md                # free-tier GPU lessons learned
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/Takayanagi-sensei/Text-to-Vehicle-Retrieval.git
cd Text-to-Vehicle-Retrieval
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended. Everything was developed on free Kaggle T4/P100 GPUs.

---

## Quick start: run inference with the pretrained weights

You do **not** need either dataset to run inference on your own video.

**1. Download the checkpoint** from the [Hugging Face model page](https://huggingface.co/Takayanagi-sensei/Text-to-Vehicle-Retrieval-Model) into a `checkpoints/` folder:

```bash
pip install -U huggingface_hub
huggingface-cli download Takayanagi-sensei/Text-to-Vehicle-Retrieval-Model --local-dir checkpoints
```

**2a. Launch the interactive demo:**

```bash
python app/gradio_app.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/<your-checkpoint-file>.pt \
    --share
```

Open the printed local (or public, with `--share`) URL, upload a video, type a query, and press **Find the vehicle**.

**2b. Or call it from Python:**

```python
from src.pipeline.inference import run_pipeline

output_path, ranked, summary = run_pipeline(
    video_path="path/to/video.mp4",
    query_text="a white sedan driving straight",
    checkpoint_path="checkpoints/<your-checkpoint-file>.pt",
)
print(summary)   # top-5 tracks with similarity scores
```

For many queries in a row, instantiate `VehicleTextRetriever` once and call `.run(...)` repeatedly so the models load only once.

### Writing good queries

The model learned from CityFlow-NL annotations, which describe **color, coarse vehicle type, and motion**. Queries in that style work best:

| Works well | Does not work well |
|---|---|
| `a white sedan driving straight` | `a Jeep Liberty KJ, 2002-2007` |
| `a black SUV turning left at the intersection` | `a car with a dented rear bumper` |
| `a red pickup truck following another car` | `license plate ABC-1234` |

Fine-grained make, model, trim, or plate-level queries are outside what this model was trained to do.

---

## Training your own model

Training requires both datasets, which are **not** included in this repository.

### Datasets

| Dataset | Used for | How to get it |
|---|---|---|
| **VeRi-776** | Re-ID pretraining | Public research dataset: [official source](https://github.com/JDAI-CV/VeRidataset) or a Kaggle mirror. Expected layout: `image_train/`, `image_query/`, `image_test/` plus the label files. |
| **CityFlow-NL** | Text-vehicle contrastive training | Access-restricted. Request it from the [AI City Challenge Track 2 download page](https://www.aicitychallenge.org/2023-track2-download/). |

Point `configs/config.yaml` at wherever you place them (`data/veri776/`, `data/cityflow_nl/` by default).

### Pipeline

```bash
# 1. Extract only the vehicle crops CityFlow-NL's annotations reference
#    (never writes full frames, so it stays well under the free-tier disk quota)
python scripts/extract_cityflow_crops.py --config configs/config.yaml

# 2. Stage 1 - Re-ID pretraining on VeRi-776 (resumable)
python -m src.training.train_reid --config configs/config.yaml

# 3. Stage 2 - contrastive text-vehicle training on CityFlow-NL (resumable)
python -m src.training.train_joint --config configs/config.yaml
```

Both training scripts resume automatically from the latest checkpoint and keep only the most recent few to save disk space.

### Running on Kaggle

If you prefer notebooks, `notebooks_combined/` contains two self-contained notebooks:

- `01_finetuning.ipynb` - setup, VeRi-776 Re-ID training, CityFlow-NL crop extraction, joint contrastive training.
- `02_inference.ipynb` - load the checkpoint, run the pipeline on a video, Gradio demo, and formal evaluation.

Attach the datasets via **Add Input**, run top to bottom, and download the final `joint_model_epochN.pt`. See [`docs/gpu_notes.md`](docs/gpu_notes.md) for practical free-tier tips (disk quota, DataLoader worker crashes, mixed precision).

### Evaluation

- **Re-ID (VeRi-776):** mAP and Rank-1/5/10 following the dataset's query/gallery protocol.
- **Text retrieval (CityFlow-NL):** Mean Reciprocal Rank and Recall@5/@10. `test-tracks.json` carries no labels, so the notebooks hold out a slice of the training tracks for local MRR, and also write the official-format `submission.json`.

---

## Results

| Metric | Value |
|---|---|
| VeRi-776 mAP | _fill in_ |
| VeRi-776 Rank-1 | _fill in_ |
| CityFlow-NL held-out MRR | _fill in_ |
| CityFlow-NL held-out Recall@5 / @10 | _fill in_ |

---

## Limitations

- **Coarse attributes only.** It understands color, rough body type, and motion described in CityFlow-NL's style, not specific makes, models, or generations.
- **Single-camera inference.** The demo ranks tracks within one video. Cross-camera association is not implemented.
- **Detector dependence.** If YOLO misses or fragments a vehicle, that vehicle cannot be retrieved. Heavily occluded or very distant vehicles are the usual failures.
- **Short clips by default.** The demo processes the first 600 frames unless you raise `max_frames`.
- **Research prototype.** Not validated for real-world surveillance, enforcement, or any decision affecting people.

---

## License and data restrictions

The **code** in this repository is released under the MIT License.

That license does **not** cover the datasets or models trained on them:

- **VeRi-776** has its own license. Check the [official source](https://github.com/JDAI-CV/VeRidataset); it is intended for research use only.
- **CityFlow-NL** is distributed under NVIDIA's Dataset License Agreement (v. January 12, 2023). Among other terms:
  - **Non-commercial, academic use only**, for the dataset *and* any model trained on it. Do not use it or derived models for commercial or production purposes.
  - **No redistribution.** No CityFlow-NL video, images, or annotations are included in this repository or in the linked model weights, only trained model parameters. To retrain or evaluate on CityFlow-NL yourself, request access directly from the AI City Challenge organizers.
  - **No unredacted faces or license plates** may be reproduced or displayed in anything derived from the dataset. Blur them before sharing demo videos or screenshots.
  - **No implied NVIDIA endorsement** of this project.
- Per the challenge rules, no model pretrained on the CityFlow benchmark is used. The Re-ID backbone is pretrained on the independent VeRi-776 dataset instead.

---

## Citation

If you build on this work, please cite the datasets it relies on:

```bibtex
@article{feng2021cityflownl,
  title   = {CityFlow-NL: Tracking and Retrieval of Vehicles at City Scale by Natural Language Descriptions},
  author  = {Feng, Qi and others},
  journal = {arXiv preprint arXiv:2101.04741},
  year    = {2021}
}

@inproceedings{tang2019cityflow,
  title     = {CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification},
  author    = {Tang, Zheng and others},
  booktitle = {CVPR},
  year      = {2019}
}

@inproceedings{liu2016veri,
  title     = {Large-scale vehicle re-identification in urban surveillance videos},
  author    = {Liu, Xinchen and Liu, Wu and Ma, Huadong and Fu, Huiyuan},
  booktitle = {IEEE International Conference on Multimedia and Expo},
  year      = {2016}
}
```

## Acknowledgments

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [ByteTrack](https://github.com/ifzhang/ByteTrack), [supervision](https://github.com/roboflow/supervision), [OpenCLIP](https://github.com/mlfoundations/open_clip), [timm](https://github.com/huggingface/pytorch-image-models), and the organizers of the [AI City Challenge](https://www.aicitychallenge.org/).
