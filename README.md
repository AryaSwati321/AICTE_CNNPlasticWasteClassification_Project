# AICTE_CNNPlasticWasteClassification_Project

🌍 CNN Model for Waste Management
A deep learning-based approach to classify waste into different categories using Convolutional Neural Networks (CNNs).

📝 Overview
This project aims to develop a CNN model that accurately classifies waste into categories such as:
♻️ Recyclable
🗑️ Non-recyclable
🍃 Organic, etc.

The model is trained and tested on a dataset containing images of various waste types.
✨ Features
📚 CNN model implemented using Keras and TensorFlow
🖼️ Dataset of images with various types of waste
⚙️ Scripts for training and testing
📊 Model evaluation using accuracy, precision, recall, and F1-score
🛠️ Requirements

Ensure you have the following installed:
🐍 Python 3.x
⚙️ Keras 2.x
🧠 TensorFlow 2.x
🔢 NumPy
📈 Matplotlib
📊 Scikit-learn
🚀 Installation

Clone the repository:
git clone https://github.com/AryaSwati321/AICTE_CNNPlasticWasteClassification_Project
Install dependencies:
pip install -r requirements.txt  

🏃‍♂️ Usage
Train the model:
python train.py  
Test the model:
python test.py  
Evaluate the model:
python evaluate.py  

🗂️ Dataset
The dataset used consists of images of various waste types, divided into:

🏋️‍♂️ Training set
🧪 Testing set

🏗️ Model Architecture
 Depicts the architectural model of a Convolutional Neural Network (CNN) designed for plastic waste management:
1️⃣ Input Layer: Takes images of waste (e.g., bottles, bags, organic material).
2️⃣ Conv2D Layers: Extracts features using filters and kernels to identify patterns in waste images.
3️⃣ MaxPooling2D Layers: Reduces spatial dimensions while preserving key features, improving computational efficiency.
4️⃣ Flatten Layer: Converts 2D feature maps into a 1D vector for further processing.
5️⃣ Dense Layers: Fully connected layers with ReLU activation to analyze and refine features.
6️⃣ Softmax Output Layer: Predicts the waste category (e.g., Recyclable, Non-Recyclable, Organic).
This architecture enables accurate waste classification, aiding in better waste management practices. 🚮✨
