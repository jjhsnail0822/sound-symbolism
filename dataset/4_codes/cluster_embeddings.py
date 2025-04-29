import os
import json
import pickle
import argparse
import numpy as np
from sklearn.cluster import KMeans

def parse_args():
    parser = argparse.ArgumentParser(description='K-means clustering for word embeddings')
    parser.add_argument('-l', '--language', type=str, default='ko', 
                        help='Language code (default: ko)', choices=['ko', 'en', 'ja', 'fr'])
    parser.add_argument('-k', '--num-clusters', type=int, default=435, 
                        help='Number of clusters (default: 435)')
    parser.add_argument('-o', '--output-dir', type=str, default='dataset/1_preprocess/nat',
                        help='Output directory for clustered data')
    return parser.parse_args()

# Parse command line arguments
args = parse_args()

# Load embeddings
embeddings_path = f'dataset/1_preprocess/nat/{args.language}_embeddings.pkl'
print(f"Loading embeddings from {embeddings_path}")
with open(embeddings_path, 'rb') as f:
    embeddings = pickle.load(f)

# Load original data
data_path = f'dataset/1_preprocess/nat/{args.language}.json'
print(f"Loading original data from {data_path}")
with open(data_path, 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# Extract embedding vectors
embedding_vectors = np.array([item['embedding'] for item in embeddings])

# Apply K-means clustering
print(f"Clustering with K={args.num_clusters}")
kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embedding_vectors)

# Merge cluster information with original data
print("Merging cluster information with original data")

# Create a mapping from word to cluster
word_to_cluster = {}
for i, item in enumerate(embeddings):
    word_to_cluster[item['word']] = int(clusters[i])

# Create a new data structure to hold the clustered data
new_data = []
for i in range(args.num_clusters):
    new_data.append({'cluster_id': i, 'words': []})

# Add words to their corresponding clusters
for item in original_data:
    if item.get('found', False) == False:
        continue    
    word = item['word']
    if word in word_to_cluster:
        cluster_id = word_to_cluster[word]
        new_data[cluster_id]['words'].append(item)

# Save clustered data
output_path = os.path.join(args.output_dir, f'{args.language}_clustered.json')
print(f"Saving clustered data to {output_path}")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
print("Clustering complete.")