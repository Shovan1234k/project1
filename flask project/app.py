from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form input
    area = float(request.form["area"])
    prediction = model.predict([[area]])[0]

    return render_template("index.html", 
                           prediction_text=f"Predicted Price = {prediction:.2f}k")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    area = data["area"]
    prediction = model.predict([[area]])[0]
    return jsonify({"predicted_price": prediction})

if __name__ == "__main__":
    app.run(debug=True)
