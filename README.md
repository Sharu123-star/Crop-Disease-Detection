# 🌱 Crop Disease Detection and Classification Using Deep Learning
A deep learning-based web application that detects and classifies diseases in tomato leaves using MobileNetV2 transfer learning. Users can upload a tomato leaf image and receive a predicted disease class with confidence and class probabilities.

## 🚀 Features

- Tomato leaf disease detection
- Classification into four disease categories
- MobileNetV2 transfer learning
- Image preprocessing and resizing
- Confidence score for predictions
- Class probability visualization
- Disease information for the predicted class
- Interactive Streamlit web interface
- Separate training, validation, and test datasets

## 🦠 Disease Classes

The model classifies tomato leaf images into four categories:

| Class | Description |
|---|---|
| Bacterial Spot | A bacterial disease that can cause small dark spots on tomato leaves. |
| Early Blight | A fungal disease commonly associated with dark spots and concentric ring patterns. |
| Healthy | Tomato leaves showing no disease class detected by the model. |
| Late Blight | A disease that can cause dark, irregular lesions on tomato leaves. |

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Model:** MobileNetV2 with Transfer Learning
- **Data Processing:** NumPy, Pandas
- **Image Processing:** Pillow
- **Machine Learning Utilities:** Scikit-learn
- **Visualization:** Matplotlib
- **Web Application:** Streamlit
- **Version Control:** Git, GitHub

## 📊 Dataset

This project uses a subset of the **PlantVillage dataset**, focusing on four tomato leaf classes:

- Tomato___Bacterial_spot
- Tomato___Early_blight
- Tomato___healthy
- Tomato___Late_blight

The dataset contains **6,627 images** across these four classes.

The data was divided into:

- **Training set:** 5,301 images
- **Validation set:** 663 images
- **Test set:** 663 images

Stratified splitting was used to maintain the class distribution across the three sets.

## 🧠 Model Architecture

The project uses **MobileNetV2** with transfer learning.

### Architecture

1. **Input Image**
   - Tomato leaf image
   - Resized to `224 × 224 × 3`

2. **MobileNetV2**
   - Pre-trained on ImageNet
   - Used as the feature extraction base
   - Base model layers are frozen during training

3. **Global Average Pooling**
   - Converts extracted feature maps into a compact feature representation

4. **Dropout**
   - Dropout rate: `0.2`
   - Helps reduce overfitting

5. **Dense Output Layer**
   - 4 output neurons
   - Softmax activation
   - Produces probabilities for the four tomato leaf classes

### Training Configuration

- **Optimizer:** Adam
- **Learning Rate:** 0.0001
- **Loss Function:** Sparse Categorical Crossentropy
- **Batch Size:** 8
- **Epochs:** 5

## 🔄 Project Workflow

```text
Tomato Leaf Image
        ↓
Image Preprocessing
        ↓
Resize to 224 × 224
        ↓
MobileNetV2 Feature Extraction
        ↓
Global Average Pooling
        ↓
Dropout
        ↓
Dense + Softmax Layer
        ↓
Disease Classification
        ↓
Prediction + Confidence Score
        ↓
Streamlit Web Application

*Workflow Explanation*
The user uploads a tomato leaf image.
The image is converted to RGB format and resized to 224 × 224 pixels.
MobileNetV2 extracts useful visual features from the image.
The extracted features are passed through the classification layers.
The Softmax layer produces probabilities for all four classes.
The class with the highest probability is selected as the prediction.
The Streamlit application displays the predicted class, confidence, and class probabilities.

## 📈 Results

The trained MobileNetV2 model was evaluated on the test dataset.

- **Test Accuracy:** 88.84%
- **Test Loss:** 0.3277

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Bacterial Spot | 0.91 | 0.94 | 0.92 |
| Early Blight | 0.75 | 0.64 | 0.69 |
| Healthy | 0.95 | 0.96 | 0.96 |
| Late Blight | 0.88 | 0.90 | 0.89 |
| **Overall Accuracy** | | | **0.89** |

The model performed differently across the four classes, with the test results showing stronger classification performance for the Healthy and Bacterial Spot classes and comparatively lower performance for Early Blight.

## 📁 Project Structure

```text
Crop-Disease-Detection/
│
├── app.py
├── train_model.py
├── prepare_dataset.py
├── view_dataset.py
├── crop_disease_mobilenetv2.keras
├── requirements.txt
├── .gitignore
├── README.md
│
├── dataset/
│   └── tomato/
│       ├── Tomato___Bacterial_spot/
│       ├── Tomato___Early_blight/
│       ├── Tomato___healthy/
│       └── Tomato___Late_blight/
│
└── splits/
    ├── train.csv
    ├── validation.csv
    └── test.csv

 ## ▶️ How to Run Locally

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd Crop-Disease-Detection
### 2. Create a Virtual Environment

```bash
python -m venv venv

Then below that, add:

```markdown
### 3. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
### 4. Install Dependencies

```bash
pip install -r requirements.txt
### 5. Run the Streamlit Application

```bash
streamlit run app.py

## 🌐 Deployment

The Streamlit application can be deployed online using **Streamlit Community Cloud**.

After deployment, the application can be accessed through a public URL, allowing users to upload tomato leaf images and obtain disease predictions through a web browser.

## 🌐 Web Application

The project includes an interactive Streamlit web application where users can upload a tomato leaf image and receive a disease classification result.

### Application Features

- Upload tomato leaf images in JPG, JPEG, or PNG format
- Display the uploaded image
- Predict the disease class using the trained MobileNetV2 model
- Display prediction confidence
- Display confidence level
- Show information about the predicted disease
- Display probability scores for all four classes

## ⚠️ Disclaimer

This project is developed for educational and demonstration purposes.

The predictions generated by the model should not be considered a substitute for professional agricultural diagnosis or advice.

## 🌐 Live Demo

[🚀 Try the Crop Disease Detection App](https://crop-disease-detection-jpsyxttmy2dzqe7mhwgdox.streamlit.app/)

## 👩‍💻 Author

**Sharanya Naik**

B.Tech Computer Science and Engineering Student