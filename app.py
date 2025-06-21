# app.py (Flask version)
from flask import Flask, request, render_template
import joblib
import cloudpickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack

app = Flask(__name__)

# Load models
with open("model_rf.pkl", "rb") as f:
    model = cloudpickle.load(f)
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uraian = request.form.get("uraian")
        pelaksana = request.form.get("pelaksana")
        nilai = float(request.form.get("nilai"))
        waktu = float(request.form.get("waktu"))

        uraian_vec = tfidf.transform([uraian])
        pelaksana_enc = le.transform([pelaksana])[0] if pelaksana in le.classes_ else -1
        fitur = np.array([[nilai, pelaksana_enc, waktu]])
        df_fitur = pd.DataFrame(fitur, columns=["nilai_kontrak", "pelaksana", "jangka_waktu"])
        X_input = hstack([uraian_vec, df_fitur])
        pred = model.predict(X_input)[0]

        kategori = {0: "Kecil", 1: "Menengah", 2: "Besar"}
        return render_template("index.html", hasil=kategori[pred])

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
