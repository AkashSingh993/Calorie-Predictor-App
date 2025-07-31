from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load("calorie_predictor_pipeline.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            data = {
                "Sex": request.form["sex"],
                "Age": float(request.form["age"]),
                "Height": float(request.form["height"]),
                "Weight": float(request.form["weight"]),
                "Duration": float(request.form["duration"]),
                "Heart_Rate": float(request.form["heart_rate"]),
                "Body_Temp": float(request.form["body_temp"]),
            }

            input_df = pd.DataFrame([data])
            prediction = model.predict(input_df)[0]
            prediction = np.maximum(0, prediction)

            return render_template("index.html", prediction=f"{prediction:.2f} Calories Burned")

        except Exception as e:
            return render_template("index.html", prediction="Error: " + str(e))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
