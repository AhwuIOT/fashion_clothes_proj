from flask import Flask, render_template, request
import os, uuid, shutil
import numpy as np
import cv2
from predict import predict_embedding
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
RECOMMEND_FOLDER = 'static/recommend'
IMAGE_DIR = 'static/images'  # 容器內的掛載路徑
TOP_N = 5

# Load embeddings
embeddings = np.load("embeddings.npy")
image_paths = np.load("image_paths.npy")  # 這裡應為 images/xxx.jpg，相對容器內部

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    query_img_path = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = f"{uuid.uuid4()}.jpg"
            saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(saved_path)

            query_emb = predict_embedding(saved_path).reshape(1, -1)
            sims = cosine_similarity(query_emb, embeddings)[0]
            top_indices = np.argsort(sims)[-TOP_N-1:-1][::-1]

            for idx in top_indices:
                img_path = image_paths[idx]  # e.g., images/123.jpg
                abs_path = os.path.join(IMAGE_DIR, os.path.basename(img_path))

                if not os.path.exists(abs_path):
                    print(f"⚠️ Image not found: {abs_path}")
                    continue

                try:
                    dst_path = os.path.join(RECOMMEND_FOLDER, os.path.basename(img_path))
                    shutil.copyfile(abs_path, dst_path)
                    score = round(sims[idx], 3)
                    recommendations.append((dst_path, score))
                except Exception as e:
                    print(f"❌ Failed to copy {img_path}: {e}")
                    continue

            query_img_path = saved_path

    return render_template("index.html",
                           query_img=query_img_path,
                           recommendations=recommendations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
