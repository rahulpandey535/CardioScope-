# CardioScope-
# ❤️ CardioScope – Heart Disease Prediction

## 🧠 Overview
**CardioScope** is a machine learning-based system that predicts the risk of heart disease based on patient health indicators. The project utilizes clinical data and supervised learning algorithms to assist healthcare professionals in making informed decisions.

## 🎯 Objective
To develop a predictive model that accurately classifies whether an individual is at risk of heart disease using key medical attributes. This model can act as an early warning system, promoting preventive care.

## 📦 Dataset
- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Attributes:**
  - Age
  - Sex
  - Chest pain type (cp)
  - Resting blood pressure (trestbps)
  - Serum cholesterol (chol)
  - Fasting blood sugar (fbs)
  - Resting electrocardiographic results (restecg)
  - Maximum heart rate achieved (thalach)
  - Exercise-induced angina (exang)
  - ST depression (oldpeak)
  - Slope of peak exercise ST segment (slope)
  - Number of major vessels (ca)
  - Thalassemia (thal)
  - **Target:** Presence (1) or absence (0) of heart disease

## ⚙️ Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Environment:** Jupyter Notebook  

## 🧪 Workflow

### 1. Data Preprocessing
- Missing value handling
- One-hot encoding for categorical features
- Feature scaling using StandardScaler

### 2. Exploratory Data Analysis (EDA)
- Correlation matrix
- Outlier detection
- Distribution plots for feature understanding

### 3. Model Development
Implemented and compared multiple models:
- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- XGBoost (optional)

### 4. Model Evaluation
- Accuracy Score
- Precision, Recall, F1-Score
- ROC-AUC Curve
- Confusion Matrix

## 🏆 Results
| Model                 | Accuracy | AUC Score |
|----------------------|----------|-----------|
| Random Forest         | 86.0%    | 0.89      |
| Logistic Regression   | 83.1%    | 0.86      |
| K-Nearest Neighbors   | 81.7%    | 0.84      |

> ✅ **Random Forest Classifier** provided the best overall performance in terms of accuracy and AUC.

## 📁 Project Structure
CardioScope/ ├── data/ │ └── heart.csv ├── models/ │ └── rf_model.pkl ├── notebooks/ │ └── Heart_Disease_Prediction.ipynb ├── app/ │ └── streamlit_app.py (Optional) ├── requirements.txt └── README.md

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/CardioScope.git
cd CardioScope
2. Install Dependencies
pip install -r requirements.txt
3. Run the Jupyter Notebook
jupyter notebook notebooks/Heart_Disease_Prediction.ipynb
4. (Optional) Launch Streamlit App
streamlit run app/streamlit_app.py
📈 Future Enhancements

Deploy as a REST API using Flask or FastAPI
UI dashboard with Streamlit or Dash
SHAP/LIME integration for model explainability
Add patient data encryption for HIPAA compliance
👤 Author

Rahul Pandey
📧 rahulpandey02124@gmail.com
💼 Aspiring Data Scientist | iOS Developer | Cybersecurity Enthusiast
🌍 India

