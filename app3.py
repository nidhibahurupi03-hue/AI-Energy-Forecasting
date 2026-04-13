from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model3/energy_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_form', methods=['POST'])
def predict_form():
    hour = int(request.form['hour'])
    day = int(request.form['day'])

    features = np.array([[hour, day]])
    prediction = model.predict(features)

    return render_template('index.html', prediction_text=f"Predicted Energy: {prediction[0]}")

if __name__ == '__main__':
    app.run(debug=True)