# 🧠 Hybrid Skin Type Classification Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25-success)
![Status](https://img.shields.io/badge/Status-Under%20Review-orange)

---

### 📘 Course: Technical Answers for Real World Problems (CBS1901)  
**Student:** Atharwa Vatsyayan  
**Register Number:** 22BBS0103  
**Faculty Mentor:** Dr. Santhi K  
**Semester:** Fall 2025-26  

---

## 🩺 Project Overview

This project presents a **Hybrid Machine Learning Model** that classifies human **skin types** (Dry, Normal, Oily, Combination) using **ResNet-50 (CNN)** and combines results with questionnaire-based data for **personalized skincare recommendations**.

The research work is currently **under review** for publication as:  
> **“A Hybrid Machine Learning Approach for Personalized Skincare Recommendations Using Questionnaire Data and Real-Time Skin Analysis.”**

---

## 🎯 Objective

To develop a personalized skincare system that:
1. Classifies skin type from facial images using **transfer learning**.  
2. Integrates **questionnaire data** for accurate recommendations.  
3. Provides **real-time AI-based skincare feedback**.

---

## ⚙️ Features

✅ Four skin types: Dry, Normal, Oily, Combination  
✅ ResNet50 transfer learning with 20-epoch fine-tuning  
✅ Real-time classification and probability visualization  
✅ Data augmentation for better generalization  
✅ Easy model saving and loading for reuse  

---

## 🧩 Dataset

final_self_skin_dataset/
├── train/
│ ├── dry/
│ ├── normal/
│ ├── oily/
│ └── combination/
├── valid/
└── test/


*(You can use any dataset following this folder structure.)*

---

## 🧠 Model Architecture

| Component | Details |
|:-----------|:---------|
| Base Model | ResNet50 (pretrained on ImageNet) |
| Final Layer | Linear (in_features, 4 outputs) |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam (lr=0.0001) |
| Scheduler | ReduceLROnPlateau |
| Epochs | 20 |
| Batch Size | 32 |

---

## 🧪 Training Performance

| Metric | Train | Validation |
|:------:|:------:|:-----------:|
| Accuracy | 100% | 95% |
| Loss | 0.015 | 0.208 |

**Test Accuracy:** 85%  
**Macro F1-Score:** 0.85  

---

## 📊 Visualizations

- 📉 Training & Validation Loss Curve  
- 📈 Training & Validation Accuracy Curve  
- 🔷 Confusion Matrix  
- 🖼️ Prediction Visualization on Test Images  

---

## 🖥️ Requirements

Install dependencies with:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm pillow


The dataset consists of categorized facial skin images organized as:

