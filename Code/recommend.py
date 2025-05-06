import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
from predict import predict_embedding  # 從你剛剛的 predict.py 匯入功能
from sklearn.metrics.pairwise import cosine_similarity

# 載入事先算好的所有 embedding
embeddings = np.load("embeddings.npy")
image_paths = np.load("image_paths.npy")

def recommend(img_path, top_n=5):
    query_emb = predict_embedding(img_path).reshape(1, -1)
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(sims)[-top_n-1:-1][::-1]

    # 顯示參考圖
    ref_img = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    plt.title("Query Image")
    plt.axis("off")
    plt.show()

    # 顯示推薦圖
    fig, axes = plt.subplots(1, top_n, figsize=(15, 5))
    for i, idx in enumerate(top_indices):
        rec_img = cv2.imread(image_paths[idx])
        axes[i].imshow(cv2.cvtColor(rec_img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Score: {sims[idx]:.2f}")
        axes[i].axis("off")
    plt.suptitle("Top {} Recommendations".format(top_n))
    plt.show()

# 主程式執行
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("請提供圖片路徑：python recommend.py ./images/123.jpg")
        sys.exit()
    img_path = sys.argv[1]
    recommend(img_path, top_n=5)