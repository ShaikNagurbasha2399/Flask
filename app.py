
from flask import Flask, request, jsonify
import numpy as np
import os
import joblib
from flask_cors import CORS


# Load trained model safely
model = joblib.load("model.pkl")


app = Flask(__name__)
CORS(app)  # enables CORS for all routes


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract required inputs from request body
        prev_speed = float(data.get("prev_speed"))
        prev_acc = float(data.get("prev_acc"))

        # Derived values
        desired_speed = prev_speed + 10
        desired_acc = 95
        time = 10  

        # Arrange features in the same order as used during training
        features = np.array([[prev_speed, prev_acc, desired_speed, desired_acc, time]])
        # Here "0" is placeholder for no_of_days (if model requires it but not provided)

        # Predict
        prediction = model.predict(features)

        return jsonify({"predicted_value": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

