# Meme Emotion Classification 🧠🎭

This project focuses on **classifying the emotional tone of memes** — using image, text, and multimodal (image + text) representations.  
It supports both **independent modality classification** and **fusion-based models** for richer emotion prediction.

---

## 📁 Project Structure

```
meme-emotion/
├── data/                    # Placeholder for datasets (not tracked in Git)
│   ├── processed/
│   └── raw/
├── docs/                    # Documentation, figures, or references
├── models/                  # Saved model checkpoints (ignored in git)
├── notebooks/               # Jupyter notebooks for experiments
├── scripts/                 # Training and inference scripts
│   ├── predict_sentiment.py
│   ├── predict_text_sentiment.py
│   ├── train_multi_outputs.py
│   ├── train_text_multi_output.py
│   ├── train_text_classifier.py
│   ├── train_resnet.py
│   └── train_svm.py
├── src/                     # Core package code
│   └── meme_emotion/
│       ├── __init__.py
│       ├── data.py
│       ├── utils.py
│       └── models/
│           ├── image_classifier.py
│           ├── text_classifier.py
│           └── multimodal.py
├── pyproject.toml
└── README.md
```

---

## ⚙️ Setup & Installation

### 1️⃣ Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate       # on Linux/Mac
# or
.\.venv\Scripts\activate        # on Windows
```

### 2️⃣ Install dependencies
```bash
pip install -e .
```

You can also install manually:
```bash
pip install numpy pandas scikit-learn torch torchvision
```

---

## 🚀 Usage

### Training
Train multimodal emotion classifier:
```bash
python scripts/train_multi_outputs.py
```

Train text-only classifier:
```bash
python scripts/train_text_classifier.py
```

Train ResNet-based image classifier:
```bash
python scripts/train_resnet.py
```

Train SVM baseline:
```bash
python scripts/train_svm.py
```

### Prediction
Predict emotions for new memes:
```bash
python scripts/predict_sentiment.py --image_path <path_to_image> --text "example caption"
```

---

## 🧩 Features
- Multimodal emotion classification (image + text fusion)
- SVM and deep learning baselines
- Configurable training & evaluation scripts
- Modular code for easy experimentation

---

## 🧠 Future Work
- Add transformer-based text embeddings (e.g., BERT)
- Integrate CLIP for multimodal representations
- Streamlit demo for interactive emotion prediction

---

## 🪪 License
MIT License © 2025 Chhavi Sharma
