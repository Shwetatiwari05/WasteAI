# ♻️ WasteAI — Smart Waste Classifier

A deep learning web app that classifies waste into 6 categories in real-time using **MobileNetV2 transfer learning**, built with TensorFlow and Streamlit.

🔗 **Live Demo:** [huggingface.co/spaces/ShwetaTiwari05/waste-classifier-ai](https://huggingface.co/spaces/ShwetaTiwari05/waste-classifier-ai)

---

## 📌 Overview

WasteAI takes an image (uploaded or from webcam) and classifies it into one of 6 waste categories, along with recyclability status, confidence score, and eco tips. Built using transfer learning on MobileNetV2 pretrained on ImageNet, fine-tuned on the TrashNet dataset.
We also achieved an **Accuracy of 85.56%**

---

## 🚀 App Features

- 📤 **Image Upload** — Upload any JPG, JPEG, PNG, or WEBP image directly
- 📷 **Webcam Capture** — Take a live photo from your webcam for instant classification
- 📊 **Confidence Score** — Visual progress bar showing prediction confidence
- ♻️ **Recyclability Badge** — Instantly see if the item is recyclable or not
- 🌿 **Eco Tip** — Waste-specific disposal and recycling advice
- 📈 **All Class Probabilities** — See scores for all 6 categories

---

## 🗂️ Waste Categories

| Category | Recyclable |
|----------|------------|
| Cardboard | ✅ Yes |
| Glass | ✅ Yes |
| Metal | ✅ Yes |
| Paper | ✅ Yes |
| Plastic | ✅ Yes |
| Trash | ❌ No |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | MobileNetV2 (Transfer Learning) |
| Input Size | 224 × 224 px |
| **Overall Validation Accuracy** | **85.56%** |
| Best Val Accuracy (Phase 2) | 84.10% |
| Final Train Accuracy | 98.17% |
| Final Val Accuracy | 82.50% |
| Generalisation Gap | 0.1567 (Well-generalised) |
| Training Samples | 2,024 |
| Validation Samples | 503 |
| Dataset | TrashNet |

### Per-Class Accuracy
| Class | Accuracy |
|-------|----------|
| Cardboard | 85.36% ✅ |
| Glass | 81.84% ✅ |
| Metal | 68.05% ⚠️ |
| Paper | 94.78% ✅ |
| Plastic | 96.89% ✅ |
| Trash | 72.26% ⚠️ |

---

## 🧠 Model Architecture & Training

**Base Model:** MobileNetV2 pretrained on ImageNet

### Phase 1 — Frozen Base (Head Training)
- Base model frozen, only classification head trained
- Learning rate: 3e-4
- Epochs: 20 (with EarlyStopping)
- Augmentation: rotation, zoom, horizontal flip, width/height shift

### Phase 2 — Fine-Tuning
- Top 19/154 base layers unfrozen (BatchNorm layers kept frozen)
- Learning rate: 2e-5 (slow to preserve pretrained weights)
- Epochs: 25 (EarlyStopping triggered at epoch 14)
- Best val accuracy achieved: 84.10%

### Callbacks Used
- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau

---

## 🗃️ Dataset

- **Name:** TrashNet
- **Total Images:** 2,527
- **Split:** 80% train / 20% validation
- **Classes:** 6 (cardboard, glass, metal, paper, plastic, trash)
- **Download:** [Kaggle - TrashNet](https://www.kaggle.com/datasets/fedesoriano/trashnet)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Model training |
| MobileNetV2 | Transfer learning backbone |
| Streamlit | Web app frontend |
| Hugging Face Spaces | Deployment |
| Docker | Containerization |
| Python 3.11 | Runtime |

---

## 📁 Project Structure

```
waste-classifier-ai/
├── app.py                              # Streamlit web app
├── waste_model_v2.h5                   # Trained model weights
├── class_indices_v2.json               # Class index mapping
├── requirements.txt                    # Python dependencies
├── runtime.txt                         # Python version
├── Dockerfile                          # Docker config for HF Spaces
└── Waste_Classification_Using_CNN.ipynb  # Full training notebook
```

---

## ⚙️ Local Setup

```bash
# Clone the repo
git clone https://huggingface.co/spaces/ShwetaTiwari05/waste-classifier-ai
cd waste-classifier-ai

# Create conda environment
conda create -n wasteai python=3.11
conda activate wasteai

# Install dependencies
pip install streamlit==1.38.0 tensorflow==2.13.0 keras==2.13.1 numpy==1.24.3 Pillow

# Run the app
python -m streamlit run app.py
```

App will open at `http://localhost:8501`

---

## 📓 Training Notebook

`Waste_Classification_Using_CNN.ipynb` contains the complete ML pipeline:

1. Dataset loading and extraction (TrashNet)
2. Hyperparameter configuration
3. Data augmentation with ImageDataGenerator
4. MobileNetV2 model building
5. Phase 1 training (frozen base)
6. Phase 2 fine-tuning (unfrozen top layers)
7. Training curves and loss plots
8. Confusion matrix and classification report
9. Per-class accuracy evaluation
10. Model saving (`.h5` and SavedModel format)

---

## 👩‍💻 Author

**Shweta Tiwari**
[Hugging Face](https://huggingface.co/ShwetaTiwari05)

---

*Built with TensorFlow · MobileNetV2 · Streamlit · TrashNet Dataset*
