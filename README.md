# ♻️ WasteAI — Smart Waste Classifier

A deep learning-based waste classification web app that identifies waste type from an image in real-time using **MobileNetV2** and **Streamlit**.

🔗 **Live Demo:** [waste-classifier-ai on Hugging Face](https://huggingface.co/spaces/ShwetaTiwari05/waste-classifier-ai)

---

## 📌 Overview

WasteAI classifies waste images into 6 categories and tells whether the item is recyclable or not. It uses transfer learning with MobileNetV2 trained on the TrashNet dataset.

We achieved an **Accuracy of 85.56%**

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
| Overall Val Accuracy | ~80% |
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

## 🧠 Model Architecture

- **Base Model:** MobileNetV2 (pretrained on ImageNet, frozen initially)
- **Training Strategy:** Two-phase transfer learning
  - Phase 1 — Train custom classification head (frozen base, lr=3e-4, 20 epochs)
  - Phase 2 — Fine-tune top layers of base model (lr=2e-5, 25 epochs)
- **Augmentation:** Rotation, zoom, horizontal flip, width/height shift
- **Callbacks:** EarlyStopping, ModelCheckpoint

---

## 🗃️ Dataset

- **Name:** TrashNet
- **Total Images:** 2,527
- **Split:** 80% train / 20% validation
- **Classes:** 6 (cardboard, glass, metal, paper, plastic, trash)

---

## 🚀 Features

- 📤 Upload image for classification
- 📷 Webcam live capture
- 📊 Confidence score with progress bar
- ♻️ Recyclable / Non-recyclable badge
- 🌿 Eco tip for each waste category
- 📈 All class probabilities shown

---

## 🛠️ Tech Stack

- **Model:** TensorFlow / Keras — MobileNetV2
- **Frontend:** Streamlit
- **Deployment:** Hugging Face Spaces (Docker)
- **Language:** Python 3.11

---

## 📁 Project Structure

```
waste-classifier-ai/
├── app.py                  # Streamlit web app
├── waste_model_v2.h5       # Trained model
├── class_indices_v2.json   # Class index mapping
├── requirements.txt        # Dependencies
├── runtime.txt             # Python version
├── Dockerfile              # Docker config for HF Spaces
└── Waste_Classification_Using_CNN.ipynb  # Training notebook
```

---

## ⚙️ Local Setup

```bash
# Clone the repo
git clone https://huggingface.co/spaces/ShwetaTiwari05/waste-classifier-ai
cd waste-classifier-ai

# Create environment
conda create -n wasteai python=3.11
conda activate wasteai

# Install dependencies
pip install streamlit==1.38.0 tensorflow==2.13.0 keras==2.13.1 numpy==1.24.3 Pillow

# Run the app
python -m streamlit run app.py
```

---

## 📓 Training Notebook

The Jupyter notebook `Waste_Classification_Using_CNN.ipynb` contains the full training pipeline:
- Dataset loading and preprocessing
- Data augmentation
- MobileNetV2 transfer learning (Phase 1 + Phase 2)
- Training curves and confusion matrix
- Per-class accuracy evaluation
- Model saving

---

## 👩‍💻 Author

**Shweta Tiwari**  
[Hugging Face](https://huggingface.co/ShwetaTiwari05)

---

*Built with TensorFlow · MobileNetV2 · Streamlit · TrashNet Dataset*
