from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        data = {
            "longitude": float(request.form["longitude"]),
            "latitude": float(request.form["latitude"]),
            "housing_median_age": float(request.form["housing_median_age"]),
            "total_rooms": float(request.form["total_rooms"]),
            "total_bedrooms": float(request.form["total_bedrooms"]),
            "population": float(request.form["population"]),
            "households": float(request.form["households"]),
            "median_income": float(request.form["median_income"]),
            "ocean_proximity": request.form["ocean_proximity"]
        }

        df = pd.DataFrame([data])
        prepared_data = pipeline.transform(df)
        prediction = model.predict(prepared_data)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
