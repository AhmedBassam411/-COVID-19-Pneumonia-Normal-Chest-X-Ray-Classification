# -COVID-19-Pneumonia-Normal-Chest-X-Ray-Classification
Explainable AI with CNN + Grad-CAM (Flask Deployment)
📌 Project Overview

This project is an end-to-end medical imaging AI system that classifies chest X-ray images into:

🦠 COVID-19

🫁 Pneumonia

🩺 Normal

The system uses a custom-trained Convolutional Neural Network (CNN) and provides explainable predictions using Grad-CAM heatmaps, deployed via a Flask web application.

The focus is on:

⚡ Efficiency (GPU-safe, lightweight CNN)

🧠 Explainability (medical-grade interpretability)

🚀 Deployment readiness

🎯 Why Custom CNN (Not Transfer Learning)?

After experimentation, custom CNN outperformed transfer learning models (e.g., EfficientNet) on this dataset:

Model    -        	Accuracy	  -    Stability     -   	    GPU Load
Transfer Learning-	 ❌ Unstable	-  ❌ Class collapse	  -  🔥 High

Custom CNN	   -    ✅ 82.45%	  -  ✅ Stable	        -  🟢 Low

✔ Better texture learning
✔ No ImageNet bias
✔ Faster training
✔ More reliable predictions

🧠 Model Performance (Test Set)
✅ Test Accuracy
82.45%

📊 Classification Report
              precision    recall  f1-score   support

     COVID19       0.95      0.98      0.97       116
      NORMAL       0.59      0.97      0.73       317
   PNEUMONIA       0.99      0.75      0.85       855

    accuracy                           0.82      1288
   macro avg       0.84      0.90      0.85
weighted avg       0.89      0.82      0.83

🔍 Confusion Matrix
[[114   2   0]
 [  4 306   7]
 [  2 211 642]]


✔ High COVID-19 recall (critical for healthcare)
✔ Conservative Pneumonia predictions (high precision)
✔ No class collapse

🔬 Explainable AI with Grad-CAM

Each prediction includes a Grad-CAM heatmap highlighting the lung regions that influenced the decision.

This makes the model:

✔ Clinically interpretable

✔ Trustworthy

✔ Suitable for medical AI demonstrations

🌐 Web Application (Flask)
Features:

📤 Upload chest X-ray image

📈 Class probability visualization

🔥 Grad-CAM heatmap overlay

🧠 Model inference using .keras format

🎨 Clean, professional UI

🗂 Project Structure
├── Data/
│   ├── train/
│   └── test/
│
├── static/
│   ├── uploads/
│   └── gradcam/
│
├── templates/
│   └── index.html
│
├── app.py
├── cnn_training.py
└── README.md
