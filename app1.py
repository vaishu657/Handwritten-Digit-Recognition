from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
from locker_system import verify_login
from PIL import Image

app = Flask(__name__)
app.secret_key = "digitsecret"
model = tf.keras.models.load_model("digit_model.h5")

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["username"]
        pwd = request.form["password"]
        if verify_login(user, pwd):
            session["user"] = user
            return redirect("/predict")
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect("/")
    
    if request.method == "POST":
        file = request.files["image"]
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(path)

            img = Image.open(path).convert('L').resize((28, 28))
            img = np.array(img)
            img = img / 255.0
            img = img.reshape(1, 28, 28)

            prediction = model.predict(img)
            digit = np.argmax(prediction)

            return render_template("result.html", digit=digit, image=path)

    return render_template("index1.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
