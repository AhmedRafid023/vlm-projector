```markdown
# Medical VLM Alignment Experiment: MedGemma & LLaVA-Med

This repository contains a framework for evaluating and fine-tuning Vision-Language Models (VLMs) for medical image classification. We compare the performance of foundation models in a **Zero-Shot** setting against two supervised fine-tuning strategies: **Vision-Only** (Head tuning) and **Vision + Projector** (Alignment tuning).


---

## 📂 Project Structure

```text
.
├── configs/
│   ├── train.yaml          # Configuration for Vision-Only (Head) training
│   ├── vision_proj.yaml    # Configuration for Projector + Head training
│   ├── test.yaml           # Parameters for checkpoint evaluation
│   └── zero_shot.yaml      # Prompting config for MedGemma/LLaVA Zero-Shot
├── data/
│   └── iu-chest/           # Local data root (Images + generated CSVs)
├── src/
│   ├── models.py           # MedicalVLMExperiment (LightningModule)
│   ├── dataset_iu.py       # IU-Chest Multi-label Dataset & DataModule
│   └── utils.py            # Evaluation metrics and cleanup scripts
├── results/                # Directory for inference output CSVs
├── checkpoints/            # Saved model weights (.ckpt)
├── Dockerfile              # Environment with PyTorch 2.x & CUDA 12.1
├── process_data.py         # Data pipeline: merging, MeSH mapping, and splitting
├── train.py                # Main training script
├── test.py                 # Checkpoint testing script
├── zero_shot.py            # Zero-shot inference baseline script
└── README.md               # Project documentation

```

---

## 🧬 Experiment Overview

This research explores the impact of the multimodal projector in aligning medical visual features with the language model's embedding space.

### 1. Zero-Shot Baseline

Direct inference using `google/medgemma-4b-it`. We use structured prompting ("Is there [Pathology] in this X-ray? Answer 1 or 0") to extract binary predictions without any weight updates.

### 2. Situation: Vision Only

We freeze the entire VLM backbone (Vision Tower and Projector). Only a linear classification head is trained on the vision encoder's pooled features. This benchmarks the raw medical representation power of the encoder.

### 3. Situation: Vision + Projector

We unfreeze the **Multimodal Projector** while keeping the backbones frozen. This allows the model to learn a specialized "translation" layer between medical imagery and text embeddings before reaching the classifier.

---

## 🛠️ Execution Guide

### 1. Environment Setup

Build the Docker image to ensure all dependencies and CUDA drivers are correctly configured for the NVIDIA L4 GPU.

```bash
docker build -t proj-exp .

# Run container with GPU access and environment variables
docker run --gpus all -it --rm -v $(pwd):/app --env-file .env proj-exp /bin/bash

```

### 2. Data Preparation

Prepare the IU-Chest dataset. This script handles the merge between clinical reports and projections, maps MeSH tags to binary labels, and generates the 100-sample test split.

```bash
python process_data.py

```

### 3. Running Experiments

**Zero-Shot Inference:**

```bash
docker run --gpus all --ipc=host --rm --env-file .env -v $(pwd):/app proj-exp \
    python zero_shot.py --config configs/zero_shot.yaml

```

**Training: Vision Encoder Only:**

```bash
docker run --gpus all --ipc=host --rm --env-file .env -v $(pwd):/app proj-exp \
    python train.py --config configs/train.yaml

```

**Training: Vision + Projector Alignment:**

```bash
docker run --gpus all --ipc=host --rm --env-file .env -v $(pwd):/app proj-exp \
    python train.py --config configs/vision_proj.yaml

```

### 4. Evaluation

To run inference on the test set using a saved checkpoint:

```bash
docker run --gpus all --ipc=host --rm --env-file .env -v $(pwd):/app proj-exp \
    python test.py --config configs/test.yaml

```

---
