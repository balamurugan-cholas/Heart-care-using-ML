# 🫀 HeartCare – Heart Disease Detection using Flask & Machine Learning

HeartCare is a Flask web application that predicts the presence of heart disease based on user inputs. It uses a deep learning model built with TensorFlow/Keras and is trained on the Heart Disease UCI dataset.

---

## 🚀 Features

- Web-based form to collect patient data
- Real-time heart disease prediction
- Model built using Keras Sequential API
- Displays prediction result and model accuracy
- Uses a neural network with dropout regularization

---

## 📦 Tech Stack

- **Backend:** Flask
- **ML Libraries:** scikit-learn, TensorFlow, Keras
- **Data Handling:** Pandas, LabelEncoder
- **Frontend:** HTML (home.html)
- **Dataset:** Heart Disease UCI dataset (`heart.csv`)

---

## 🧠 Model Details

- Layers:
  - Dense(64, relu) + Dropout(0.3)
  - Dense(32, relu)
  - Dense(1, sigmoid)
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Trained for 50 epochs with batch size 16

---

## 🛠️ Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/balamurugan-cholas/Heart-care-using-ML.git
cd Heart-care-using-ML
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Ensure the dataset is in place:**
```bash
Put heart.csv inside the static/files/ directory.
```

4. **Run the app:**
```bash
python app.py
```

5. **Visit in browser:**
```bash
http://localhost:3000/
```

6. **Input Features:**
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Max Heart Rate
- Exercise Induced Angina
- Oldpeak
- ST Slope

**Output**

- Result: Indicates if heart disease is detected or not.
- Accuracy: Displays model accuracy on test data.

**👤 Author**

## 👤 Author

Balamurugan  
Data Science Enthusiast 
[GitHub](https://github.com/balamurugan-cholas)


