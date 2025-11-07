# Medical Image Classification using CNNs: Detecting Pneumonia from Chest X-rays

## Project Overview

This project applies deep learning to medical imaging by training a Convolutional Neural Network (CNN) to classify chest X-ray scans as either _Normal_ or _Pneumonia_. Using TensorFlow/Keras and a public dataset from Kaggle, this notebook demonstrates how AI can assist in early disease detection.

## Key Objectives

- Load and preprocess medical X-ray images
- Build a CNN from scratch using TensorFlow/Keras
- Train and evaluate the model on a binary classification task
- Visualize training performance and confusion matrix
- Discuss model interpretability and future extensions

## Dataset

**Dataset:** [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
**Classes:** Normal, Pneumonia  
**Format:** JPEG images (~5,800 total)

Download and unzip the dataset into the `data/` folder.

## Tools & Libraries

- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- OpenCV

## Folder Structure

```
medical-imaging-pneumonia/
├── README.md
├── pneumonia_cnn.ipynb
├── data/
│   ├── train/
│   └── test/
└── results/
    ├── training_curves.png
    ├── confusion_matrix.png
    └── gradcam_heatmap.png
```

## Learning Outcomes

- Understand CNNs and their role in image classification
- Gain hands-on experience with TensorFlow/Keras
- Learn medical data preprocessing & augmentation
- Develop an interpretable AI model for healthcare tasks

## Future Work

- Add Grad-CAM for model interpretability
- Apply transfer learning (e.g., ResNet, MobileNet)
- Expand dataset to multi-class classification (e.g., COVID-19)

## Author

**Ayaz Ali**  
Bioinformatics | Computational Biology | AI Research  
_Project designed for academic portfolio._
