import os
import json
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

LANGUAGE = 'ko'

# 1) Load embeddings
emb_path = f'data/processed/nat/clustering/{LANGUAGE}_embeddings.pkl'
with open(emb_path, 'rb') as f:
    embeddings = pickle.load(f)

# 2) Load pre-computed clusters
clustered_path = f'data/processed/nat/clustering/{LANGUAGE}_clustered.json'
with open(clustered_path, 'r', encoding='utf-8') as f:
    clustered = json.load(f)

# build word→cluster map
word_to_cluster = {}
for cluster in clustered:
    cid = cluster['cluster_id']
    for item in cluster['words']:
        word_to_cluster[item['word']] = cid

# prepare data & labels
X = []
labels = []
for item in embeddings:
    w = item['word']
    if w in word_to_cluster:
        X.append(item['embedding'])
        labels.append(word_to_cluster[w])
X = np.vstack(X)
labels = np.array(labels)

# 3) PCA로 2차원으로 차원 축소
pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X)

# 4) Scatter plot
plt.figure(figsize=(8, 8))
scatter = plt.scatter(
    X2[:, 0], X2[:, 1],
    c=labels, cmap='tab20', s=20, alpha=0.7, edgecolors='none'
)
plt.colorbar(scatter, label='Cluster ID')
plt.title(f'{LANGUAGE.upper()} Embedding Clusters (k={len(set(labels))})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.tight_layout()

# 5) 결과 저장 및 출력
out_png = f'results/plots/clustering/{LANGUAGE}_kmeans_scatter.png'
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.savefig(out_png, dpi=300)
plt.show()