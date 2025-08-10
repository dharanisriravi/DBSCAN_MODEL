import os
import uuid
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import pandas as pd
from model.clustering import run_dbscan_and_prepare
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8MB

app = Flask(__name__)
app.secret_key = "replace-with-a-strong-random-string"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # file
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(path)

            # get params from form
            try:
                eps = float(request.form.get("eps", 0.5))
                min_samples = int(request.form.get("min_samples", 5))
            except:
                eps = 0.5
                min_samples = 5

            # run clustering
            try:
                result = run_dbscan_and_prepare(path, eps=eps, min_samples=min_samples)
            except Exception as e:
                flash(f"Error processing file: {e}")
                return redirect(request.url)

            # pass result to template
            return render_template(
                "result.html",
                clusters_json=json.dumps(result["clusters_for_chart"]),
                summary=result["summary_table"].to_dict(orient="records"),
                params={"eps": eps, "min_samples": min_samples},
                raw_preview=result["preview_html"],
                file_info={"name": filename},
            )
        else:
            flash("Invalid file type. Only CSV allowed.")
            return redirect(request.url)

    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
