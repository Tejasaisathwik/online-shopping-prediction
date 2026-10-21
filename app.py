import os
from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd
import joblib

app = Flask(__name__)

folder = Path(__file__).parent
model = joblib.load(folder / "shopping_model.pkl")

FEATURES = [
    "Age", "Gender", "MonthlyIncome", "WebsiteVisits",
    "PagesViewed", "TimeSpentMinutes", "PreviousPurchases",
    "CartItems", "DiscountUsed", "DeviceType", "Membership",
    "CustomerRating", "PurchaseAmount"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "online-shopping-prediction",
        "status": "ok"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send valid JSON"}), 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing features", "features": missing}), 400

    try:
        row = pd.DataFrame([data])[FEATURES]
        result = int(model.predict(row)[0])

        return jsonify({
            "prediction": result,
            "label": "Purchased" if result == 1 else "Not Purchased"
        })
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)