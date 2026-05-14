# ⚡ AI Energy Forecasting System

## 🚀 Project Overview
The AI Energy Forecasting System is a Machine Learning-based project developed to predict energy consumption using time-based features such as hour and day.

This project uses an MLP Regressor model for forecasting and Flask for deploying the model with a simple web interface for real-time predictions.

---

## 🎯 Objectives
- Predict future energy consumption
- Understand Machine Learning workflow
- Deploy ML model using Flask
- Visualize Actual vs Predicted results

---

## 🧠 Features
✅ Energy consumption prediction  
✅ Machine Learning model training  
✅ Flask-based web application  
✅ Actual vs Predicted graph visualization  
✅ User-friendly prediction interface  

---

## 🛠️ Tech Stack
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib

---

## 📂 Project Structure

```bash
AI-Energy-Forecasting/
│
├── app3.py
├── test.py
│
├── templates/
│   └── index.html
│
├── data3/
│   └── energy.csv
│
├── src3/
│   └── main3.py
│
├── model3/
│   └── energy_model.pkl
│
├── output3/
│   └── actual_vs_pred.png
│
└── venv3/


⚙️ Installation & Setup
1️⃣ Clone Repository
git clone <your-github-repo-link>

2️⃣ Open Project Folder
cd AI-Energy-Forecasting

3️⃣ Create Virtual Environment
python -m venv venv3

4️⃣ Activate Virtual Environment
venv3\Scripts\activate

5️⃣ Install Dependencies
pip install pandas numpy matplotlib scikit-learn flask requests joblib

▶️ How to Run Project

Step 1: Train Model
cd src3
python main3.py

Step 2: Run Flask App
cd ..
python app3.py

Step 3: Open Browser
http://127.0.0.1:5000


📊 Output

Predicts energy consumption
Displays Actual vs Predicted graph
Web interface for user input
