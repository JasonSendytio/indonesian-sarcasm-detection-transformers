# Indonesian Sarcasm Detection using Fine-Tuned IndoBERT and DistilBERT

An experimental Natural Language Processing (NLP) study evaluating the classification performance and inference efficiency trade-offs between **IndoBERT** (base model) and **DistilBERT** (knowledge-distilled model) on Indonesian social media texts.

---

## 📌 Project Overview
Sarcasm introduces severe figurative nuances and polarity flips, presenting a major bottleneck for standard social media sentiment analysis. This project fine-tunes and benchmarks Transformer-based architectures on combined benchmark datasets from **IndoNLU (SmSA)** and **IdSarcasm** across three target classes:
* `0`: Literal Negative (`negative`)
* `1`: Literal Positive (`positive`)
* `2`: Sarcasm (`sarcasm`)

---

## 🚀 Key Experimental Findings
Both architectures were trained under identical hyperparameter conditions (Batch Size: 16, Warmup Steps: 500, Weight Decay: 0.01, Early Stopping on Validation Loss).

| Metric | IndoBERT-base-p1 | DistilBERT-base-uncased | Trade-off / Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **91.94%** | 89.56% | -2.38% |
| **Macro F1-Score** | **0.8998** | 0.8712 | -0.0286 |
| **Inference Time (per sample)** | 8.94 ms | **4.73 ms** | **1.89x Faster** |

> **Key Takeaway:** IndoBERT demonstrates superior semantic contextualization for figurative nuances. However, DistilBERT achieves an **89% computational speedup (1.89x faster)** with only an acceptable 2.38% drop in overall accuracy, rendering it highly practical for low-latency production pipelines.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch, Hugging Face Transformers
* **Evaluation & Metrics:** Scikit-Learn (Macro-averaged Precision, Recall, F1-Score, Confusion Matrix)
* **Data Manipulation & Visuals:** Pandas, NumPy, Matplotlib, Seaborn

---

## 📂 Repository Structure
```text
├── data/              <- Data directory (IndoNLU & IdSarcasm inputs, combined CSVs)
├── notebooks/         <- End-to-end preprocessing, fine-tuning, & evaluation notebooks
├── assets/            <- Visual plots (Loss curves and confusion matrices)
├── LICENSE            <- MIT License
└── README.md          <- Project documentation and findings