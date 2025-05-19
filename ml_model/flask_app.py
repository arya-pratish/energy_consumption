import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    build_type = int(request.form['Building Type'])
    sqft = int(request.form['Square Footage'])
    occupants = int(request.form['Number of Occupants'])
    appliances = int(request.form['Appliances Used'])
    avg_temp = int(request.form['Average Temperature'])
    dow = 1 if request.form['Day of Week'] == "Weekend" else 0
    

    # Prepare features for prediction
    
    feature_names = ["build_type", "sqft", "occupants", "appliances", "avg_temp", "dow"]
    features = pd.DataFrame([[build_type, sqft, occupants, appliances, avg_temp, dow]], columns=feature_names)

    # Predict charges
    prediction = model.predict(features)
    #  Format to float and 2 decimal places
    formatted_prediction = f"The predicted value is ${round(float(prediction), 2)}"


    return render_template("result.html", prediction=formatted_prediction)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')