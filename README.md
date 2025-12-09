# Reproducing and Extending the Segment Anything Model (SAM)

**Module:** EN4554 Final Project  
**Team:** TwoQbits  
**Members:**
* R.M.K.C. Jayathissa (210258J)
* L.A.S. Liyanaarachchi (210341H)

---

## 📌 Project Overview
This project aims to reproduce the core capabilities of Meta AI's **Segment Anything Model (SAM)** and extend its functionality to address specific limitations in the official release.

We successfully:
1.  **Reproduced** the "Promptable Segmentation" task (Points, Boxes, and Automatic Mode) using the official ViT-H model.
2.  **Optimized** the model via Parameter-Efficient Fine-Tuning (PEFT) on the COCO dataset to improve the efficiency-accuracy trade-off of the lightweight ViT-B model.
3.  **Extended** the architecture with a **"Grounded-SAM"** pipeline, enabling text-based prompting (e.g., "segment the wheels") by integrating GroundingDINO.

---

## 📂 Repository Structure

The project is organized into three main notebooks:

| Notebook | Description |
| :--- | :--- |
| `01_SAM_Reproduction.ipynb` | **Core Reproduction:** Verifies Point prompts, Box prompts, Ambiguity awareness (multi-mask output), and Zero-Shot "Segment Everything" mode. Includes interactive Gradio demos. |
| `02_SAM_FineTuning.ipynb` | **Optimization:** Implements a training loop to fine-tune the lightweight `ViT-B` decoder on a subset of the COCO 2017 dataset, improving segmentation quality on complex objects. |
| `03_SAM_Extension_Text.ipynb` | **Extension (Novelty):** Implements the **Text-to-Mask** pipeline. Connects Hugging Face's `GroundingDINO` (Semantic Detector) with SAM (Geometric Segmenter) to allow natural language queries. |

---

## 🛠️ Installation & Setup

This project is designed to run on **Google Colab** (T4 GPU recommended).

### Prerequisites
* Python 3.8+
* PyTorch (CUDA supported)
* Google Drive (for hosting model checkpoints)

### Required Libraries
```bash
pip install git+[https://github.com/facebookresearch/segment-anything.git](https://github.com/facebookresearch/segment-anything.git)
pip install transformers accelerate supervision gradio jupyter_bbox_widget
pip install opencv-python pycocotools matplotlib onnxruntime
