# Malaria Detection Using CNN with Grad-CAM

This project was developed for **AI for Engineers (UCS321)** at **Thapar Institute of Engineering and Technology, Patiala** and focuses on automated malaria detection using Convolutional Neural Networks (CNNs) with Grad-CAM based explainability.

Developed by:

* Vanshika (1024030163)
* Vanshika (1024030166)
* Nandini Garg (1024030169)
* Aishwarya Verma (1024030170)
* Diya Jain (1024031154)

---

## Project Overview

The project aims to classify red blood cell images as **Parasitized** or **Uninfected** using a deep learning model trained on the NIH Malaria Dataset.

A CNN architecture was implemented for feature extraction and classification, while Grad-CAM (Gradient-weighted Class Activation Mapping) was used to generate heatmaps highlighting the regions influencing the model’s predictions.

The project demonstrates the application of explainable AI in medical image analysis.

---

## Innovation

A key innovation in this project is the integration of **confidence analysis** alongside prediction results.

In addition to classifying images, the model provides a **confidence score** indicating how certain it is about each prediction. This improves transparency and reliability by helping users identify uncertain cases that may require further medical review.

---

## Dataset

The model was trained on the **NIH Malaria Dataset**, containing:

* 27,558 cell images
* 13,779 parasitized images
* 13,779 uninfected images

---

## Features Included

* CNN-based malaria cell classification
* Image preprocessing and normalization
* Data augmentation for improved generalization
* Grad-CAM visualization for explainable predictions
* Confidence analysis for prediction reliability
* Performance evaluation using multiple metrics

---

## Preprocessing Techniques

The following preprocessing steps were applied:

* Image resizing to 64×64
* RGB conversion
* Pixel normalization

Data augmentation included:

* Rotation
* Horizontal and vertical flipping
* Zooming
* Width/height shifting

---

## Model Architecture

The CNN model consists of:

* Convolutional Layers
* ReLU Activation
* Batch Normalization
* Max Pooling Layers
* Dropout Layer
* Dense Layers
* Sigmoid Output Layer

---

## Evaluation Metrics

Model performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The final model achieved approx. **95% validation accuracy**.

---

## Grad-CAM Explainability

Grad-CAM was integrated to visualize the important regions contributing to the model’s predictions. The generated heatmaps help improve interpretability and trust in the diagnostic process.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Seaborn

---

## Project Structure

```bash
├── malaria_app/
│   ├── README.md
│   ├── app.py
│   ├── malaria_cnn_model.keras
│   └── requirements.txt
│
├── LINKS.md
├── Malaria_Detection_Using_CNN_with_GradCAM.ipynb
└── requirements.txt
```

---

## References

1. NIH Malaria Dataset
2. TensorFlow Documentation
3. Grad-CAM Research Paper
