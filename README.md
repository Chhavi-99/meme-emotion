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

## 🔖 Citation

If you use this repository, data, or model in your research or publications, please cite the following paper:

> **Chhavi Sharma**, Deepesh Bhageria, William Scott, Srinivas PYKL, Amitava Das, Tanmoy Chakraborty, Viswanath Pulabaigari, and Björn Gambäck.  
> *SemEval-2020 Task 8: Memotion Analysis — The Visuo-Lingual Metaphor!*  
> In *Proceedings of the Fourteenth Workshop on Semantic Evaluation (SemEval-2020)*, Barcelona (online), December 2020.  
> [[Paper Link]](https://aclanthology.org/2020.semeval-1.99/)  
> DOI: [10.18653/v1/2020.semeval-1.99](https://doi.org/10.18653/v1/2020.semeval-1.99)

#### 📚 BibTeX

```bibtex
@inproceedings{sharma-etal-2020-semeval,
    title = "{S}em{E}val-2020 Task 8: Memotion Analysis- the Visuo-Lingual Metaphor!",
    author = {Sharma, Chhavi  and
      Bhageria, Deepesh  and
      Scott, William  and
      PYKL, Srinivas  and
      Das, Amitava  and
      Chakraborty, Tanmoy  and
      Pulabaigari, Viswanath  and
      Gamb{\"a}ck, Bj{\"o}rn},
    editor = "Herbelot, Aurelie  and
      Zhu, Xiaodan  and
      Palmer, Alexis  and
      Schneider, Nathan  and
      May, Jonathan  and
      Shutova, Ekaterina",
    booktitle = "Proceedings of the Fourteenth Workshop on Semantic Evaluation",
    month = dec,
    year = "2020",
    address = "Barcelona (online)",
    publisher = "International Committee for Computational Linguistics",
    url = "https://aclanthology.org/2020.semeval-1.99/",
    doi = "10.18653/v1/2020.semeval-1.99",
    pages = "759--773"
}

---

## 🪪 License
MIT License © 2025 Chhavi Sharma
