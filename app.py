from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
import numpy as np
import cv2
import base64
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "mysecretkey123"

# ================= PREDICTION HISTORY =================
prediction_history = []

# ================= LOAD MODEL =================
try:
    model = tf.keras.models.load_model("covid_cnn_model.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)

# ================= PREPROCESS IMAGE =================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not loaded!")
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))
    return img

# ================= HOME =================
@app.route("/")
def home():
    return render_template("new.html", logged_in=session.get("logged_in"))

# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "123":
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error=True)

    return render_template("login.html")

# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template("dashboard.html", logged_in=True)

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    file = request.files.get("file")
    if not file or file.filename == "":
        return redirect(url_for("dashboard"))

    # Save image
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filepath = f"uploads/{timestamp}_{file.filename}"
    file.save(filepath)

    # Prediction
    try:
        img = preprocess_image(filepath)
        pred = model.predict(img)
        predicted_class = "COVID-19" if pred[0][0] > 0.5 else "Normal"
    except:
        predicted_class = "Prediction failed"

    # Convert to base64
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    # Save history
    scan_number = len(prediction_history) + 1
    prediction_data = {
        "scan_number": scan_number,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": encoded,
        "result": predicted_class,
        "name": request.form["name"],
        "age": request.form["age"],
        "gender": request.form["gender"],
        "doctor": request.form["doctor"],
        "scan_date": request.form["scan_date"],
        "hospital_id": request.form["hospital_id"]
    }
    prediction_history.append(prediction_data)

    return render_template(
        "result.html",
        predicted_class=predicted_class,
        image=encoded,
        name=request.form["name"],
        age=request.form["age"],
        gender=request.form["gender"],
        doctor=request.form["doctor"],
        scan_date=request.form["scan_date"],
        hospital_id=request.form["hospital_id"],
        logged_in=True
    )


# ================= HISTORY (PUBLIC) =================
@app.route("/history")
def history():
    return render_template("history.html", history=prediction_history, logged_in=session.get("logged_in"))

# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ================= OTHER PAGES =================
@app.route('/about')
def about():
    return render_template('about.html', logged_in=session.get("logged_in"))

@app.route('/details')
def details():
    return render_template('details.html', logged_in=session.get("logged_in"))

@app.route('/how-works')
def how_it_works():
    return render_template('how-works.html', logged_in=session.get("logged_in"))

@app.route('/help')
def help_page():
    return render_template('help.html', logged_in=session.get("logged_in"))

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
