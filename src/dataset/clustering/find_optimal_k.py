# python src/dataset/clustering/find_optimal_k.py -l ko -k 650

import argparse
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Get optimal k using silhouette method')
parser.add_argument('-l', '--language', type=str, default='ko', help='Language code (default: ko)', choices=['ko', 'en', 'ja', 'fr', 'all'])
parser.add_argument('-k', '--max-k', type=int, default=600, help='Max k for silhouette method (default: 600)')
args = parser.parse_args()

LANGUAGE = args.language
MAX_K = args.max_k
print(f"Language: {LANGUAGE}")

with open(f'data/processed/nat/clustering/{LANGUAGE}_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# extract only the embedding vectors
if LANGUAGE == 'all':
    embedding_vectors = np.array([item['en_embedding'] for item in embeddings])
else:
    embedding_vectors = np.array([item['embedding'] for item in embeddings])

# Silhouette Method
k_range = range(2, MAX_K)
silhouette_scores = []

print("Calculating silhouette scores for different k values...")
for k in tqdm(k_range):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embedding_vectors)
    silhouette_avg = silhouette_score(embedding_vectors, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plotting the Silhouette Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title(f'Silhouette Method for Optimal k ({LANGUAGE})')
# indicate the optimal k on the plot
optimal_k = k_range[np.argmax(silhouette_scores)]
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k: {optimal_k}')
plt.legend()
plt.grid(True)
plt.savefig(f'results/plots/find_optimal_k/silhouette_method/silhouette_method_{LANGUAGE}.png')
plt.show()

print(f"Recommended optimal k based on silhouette score: {optimal_k}")