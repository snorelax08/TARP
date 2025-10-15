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

The dataset consists of categorized facial skin images organized as:

final_self_skin_dataset/
├── train/
│ ├── dry/
│ ├── normal/
│ ├── oily/
│ └── combination/
├── valid/
└── test/

ruby
Copy code

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
🚀 How to Run
Clone this repository

bash
Copy code
git clone https://github.com/<your-username>/hybrid-skin-type-classification.git
cd hybrid-skin-type-classification
Run the main script

bash
Copy code
python Main.py
(Optional) Classify a custom image

python
Copy code
from Main import classify_custom_image, load_model
model, mapping = load_model("skin_classification_4_types_model.pth")
result = classify_custom_image(model, "sample.jpg", transform)
print(result)
💾 Example Output
vbnet
Copy code
Predicted class: oily
Confidence: 0.9321

Probabilities:
dry: 0.0214
normal: 0.0133
oily: 0.9321
combination: 0.0332
📈 Evaluation Results
Skin Type	Precision	Recall	F1-score
Dry	0.70	0.78	0.74
Normal	0.90	0.90	0.90
Oily	0.90	0.75	0.82
Combination	0.90	1.00	0.95

Weighted Average Accuracy: 85%

🧠 Core Functions
Function	Description
create_df(base)	Loads dataset images and labels
CloudDS(Dataset)	Custom PyTorch dataset class
train_model()	Trains and validates the model
evaluate_model()	Evaluates model performance
classify_custom_image()	Predicts class for a custom image
save_model() / load_model()	Save and reload model weights

🧩 Repository Structure
bash
Copy code
├── Main.py
├── README.md
├── skin_classification_4_types_model.pth
├── /final_self_skin_dataset
│   ├── train/
│   ├── valid/
│   └── test/
└── requirements.txt
🎥 Demonstration Video
📹 Google Drive Link to Demo Video (Add your actual link here)

🧩 Future Enhancements
Integrate questionnaire data for hybrid recommendation.

Develop a web-based interface using Flask or Streamlit.

Add live webcam input for real-time skin type detection.

Suggest personalized skincare products based on results.

🏁 Conclusion
This project demonstrates a robust AI-powered skincare analysis tool that combines computer vision and machine learning for personalized skin type prediction. The system performs with high accuracy and serves as a foundation for next-generation skincare recommendation platforms.

⭐ Author: Atharwa Vatsyayan
📧 Contact: [your.email@example.com]
📍 VIT Vellore

yaml
Copy code

---

Would you like me to also generate a **`requirements.txt`** file based on your imports (for GitHub upload)? It’ll make your repository setup fully complete.






You said:
give all of this In a single README Code


ChatGPT said:
Here’s your complete README.md — ready to upload directly to GitHub as one single code block.
Just copy this and paste it into your repository’s README.md file 👇

markdown
Copy code
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

The dataset consists of categorized facial skin images organized as:

final_self_skin_dataset/
├── train/
│ ├── dry/
│ ├── normal/
│ ├── oily/
│ └── combination/
├── valid/
└── test/

ruby
Copy code

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
🚀 How to Run
Clone this repository

bash
Copy code
git clone https://github.com/<your-username>/hybrid-skin-type-classification.git
cd hybrid-skin-type-classification
Run the main script

bash
Copy code
python Main.py
(Optional) Classify a custom image

python
Copy code
from Main import classify_custom_image, load_model
model, mapping = load_model("skin_classification_4_types_model.pth")
result = classify_custom_image(model, "sample.jpg", transform)
print(result)
💾 Example Output
vbnet
Copy code
Predicted class: oily
Confidence: 0.9321

Probabilities:
dry: 0.0214
normal: 0.0133
oily: 0.9321
combination: 0.0332
📈 Evaluation Results
Skin Type	Precision	Recall	F1-score
Dry	0.70	0.78	0.74
Normal	0.90	0.90	0.90
Oily	0.90	0.75	0.82
Combination	0.90	1.00	0.95

Weighted Average Accuracy: 85%

🧠 Core Functions
Function	Description
create_df(base)	Loads dataset images and labels
CloudDS(Dataset)	Custom PyTorch dataset class
train_model()	Trains and validates the model
evaluate_model()	Evaluates model performance
classify_custom_image()	Predicts class for a custom image
save_model() / load_model()	Save and reload model weights

🧩 Repository Structure
bash
Copy code
├── Main.py
├── README.md
├── skin_classification_4_types_model.pth
├── /final_self_skin_dataset
│   ├── train/
│   ├── valid/
│   └── test/
└── requirements.txt
🎥 Demonstration Video
📹 Google Drive Link to Demo Video (Add your actual link here)

🧩 Future Enhancements
Integrate questionnaire data for hybrid recommendation.

Develop a web-based interface using Flask or Streamlit.

Add live webcam input for real-time skin type detection.

Suggest personalized skincare products based on results.

🏁 Conclusion
This project demonstrates a robust AI-powered skincare analysis tool that combines computer vision and machine learning for personalized skin type prediction. The system performs with high accuracy and serves as a foundation for next-generation skincare recommendation platforms.
