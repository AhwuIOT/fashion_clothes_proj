from flask import Flask, render_template, request
import os, uuid
import numpy as np
import cv2
from predict import predict_embedding
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

import shutil

# 複製推薦圖片到 static 資料夾內供瀏覽器顯示



# Load precomputed embeddings
embeddings = np.load("embeddings.npy")
image_paths = np.load("image_paths.npy")

@app.route("/", methods=["GET", "POST"])
def index():
    static_rec_paths = []
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = str(uuid.uuid4()) + ".jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            query_emb = predict_embedding(filepath).reshape(1, -1)
            sims = cosine_similarity(query_emb, embeddings)[0]
            top_indices = np.argsort(sims)[-6:-1][::-1]
            rec_imgs = [image_paths[i] for i in top_indices]
            rec_scores = [round(sims[i], 2) for i in top_indices]
            for i, img_path in enumerate(rec_imgs):
                filename = os.path.basename(img_path)
                static_path = os.path.join("static", "recommend", filename)
                shutil.copyfile(img_path, static_path)
                static_rec_paths.append(static_path)
            recommendations = list(zip(static_rec_paths, rec_scores))
            
            return render_template("index.html",
                       query_img=filepath,
                       recommendations=recommendations)
            
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
