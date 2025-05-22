import json
import argparse
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import networkx as nx
from tqdm import tqdm
import time
from typing import Dict, List, Any, Optional, Tuple

# --- Constants ---
DEFAULT_INPUT_DIR = "dataset/1_preprocess/nat"
DEFAULT_OUTPUT_DIR = "dataset/1_preprocess/nat/crosslingual_dbscan_clusters"
DEFAULT_MIN_SAMPLES = 5
DEFAULT_EPS = 0.5
TARGET_LANGUAGES = ["en", "fr", "ja", "ko"]
EMBEDDING_KEY = "en_embedding"

# --- Helper Functions ---
def load_json(filepath: str) -> Optional[Any]:
    """Loads JSON data from a file with error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {filepath}. Check file format. Details: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
        return None

def save_json(data: Any, filepath: str):
    """Saves data to a JSON file with error handling."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved data to {filepath}")
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")

def load_pickle(filepath: str) -> Optional[Any]:
    """Loads data from a pickle file with error handling."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
        return None

# --- DBSCAN-based Clustering Logic ---
def perform_dbscan_clustering(
    languages: List[str],
    eps: float,
    min_samples: int,
    input_dir: str,
    output_dir: str,
    embedding_key: str
) -> Tuple[Optional[str], int, int]:
    """
    Performs DBSCAN-based clustering on word embeddings from multiple languages.
    Returns the output file path, the count of clusters (excluding noise),
    and the count of clusters containing all target languages.
    """
    print("\n--- DBSCAN-based Cross-lingual Clustering ---")
    print(f"Parameters: eps={eps}, min_samples={min_samples}")
    print(f"Target languages for this run: {languages}")
    start_time = time.time()

    all_word_items: List[Dict[str, Any]] = []
    word_identifier_set = set()

    # Load Data and Embeddings for all languages
    print("--- Loading data and embeddings ---")
    for lang_idx, lang in enumerate(languages):
        print(f"Processing language: {lang} ({lang_idx+1}/{len(languages)})")
        embeddings_path = os.path.join(input_dir, f'{lang}_embeddings.pkl')
        embeddings_data = load_pickle(embeddings_path)
        if embeddings_data is None:
            print(f"Skipping language {lang} due to missing embeddings file.")
            continue

        data_path = os.path.join(input_dir, f'{lang}.json')
        original_data_list = load_json(data_path)
        if original_data_list is None:
            print(f"Skipping language {lang} due to missing original data file.")
            continue

        word_to_original_item = {item.get('word'): item for item in original_data_list if isinstance(item, dict)}

        for embed_item in embeddings_data:
            word = embed_item.get('word')
            embedding_vector = embed_item.get(embedding_key)
            original_item = word_to_original_item.get(word)
            identifier = (lang, word)

            if word and embedding_vector is not None and original_item and identifier not in word_identifier_set:
                original_item_copy = original_item.copy()
                original_item_copy['language'] = lang
                all_word_items.append({
                    'word': word,
                    'language': lang,
                    'original_data': original_item_copy,
                    'embedding': np.array(embedding_vector, dtype=np.float32),
                    'id': len(all_word_items)
                })
                word_identifier_set.add(identifier)

    if not all_word_items:
        print("Error: No valid embeddings found across languages.")
        return None, 0, 0

    print(f"\nLoaded a total of {len(all_word_items)} words from {len(languages)} languages.")

    embeddings_matrix = np.array([item['embedding'] for item in all_word_items])
    if embeddings_matrix.shape[0] == 0:
        print("Error: Embeddings matrix is empty.")
        return None, 0, 0

    # Normalize embeddings
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normalized_embeddings_matrix = embeddings_matrix / norms

    # Perform DBSCAN Clustering
    print(f"\n--- Performing DBSCAN clustering (eps={eps}, min_samples={min_samples}) ---")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
    cluster_labels = dbscan.fit_predict(normalized_embeddings_matrix)

    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise_points = np.sum(cluster_labels == -1)
    print(f"DBSCAN found {num_clusters} clusters and {num_noise_points} noise points.")

    # Build the final clustered data structure
    clustered_data_structure = []
    if num_clusters > 0:
        for i in range(num_clusters):
            cluster_id = i
            words_in_cluster = []
            for item_idx, label in enumerate(cluster_labels):
                if label == cluster_id:
                    word_data = all_word_items[item_idx]['original_data'].copy()
                    word_data['cluster_id'] = cluster_id
                    words_in_cluster.append(word_data)
            
            if words_in_cluster:
                words_in_cluster.sort(key=lambda x: (x.get('language', ''), x.get('word', '')))
                clustered_data_structure.append({'cluster_id': cluster_id, 'words': words_in_cluster})

    # Calculate number of clusters containing all target languages
    complete_clusters_count = 0
    if clustered_data_structure:
        for cluster_info in clustered_data_structure:
            langs_in_cluster = set(word_item['language'] for word_item in cluster_info['words'])
            if all(req_lang in langs_in_cluster for req_lang in languages):
                complete_clusters_count += 1

    print(f"Found {complete_clusters_count} clusters containing all {len(languages)} specified target languages.")
    print(f"Total non-noise clusters formed: {len(clustered_data_structure)}")

    # Save Clustered Data
    output_filename = f'crosslingual_dbscan_eps{str(eps).replace(".", "p")}_ms{min_samples}_clustered.json'
    output_path = os.path.join(output_dir, output_filename)
    save_json(clustered_data_structure, output_path)

    end_time = time.time()
    return output_path, len(clustered_data_structure), complete_clusters_count

# --- Hyperparameter Exploration ---
def explore_hyperparameters(
    base_languages: List[str],
    base_input_dir: str,
    base_output_dir: str,
    base_embedding_key: str,
    eps_values: List[float],
    min_samples_values: List[int]
):
    print("\n--- Starting Hyperparameter Exploration for DBSCAN ---")
    print(f"Target languages for 'complete' clusters: {base_languages} (count: {len(base_languages)})")
    if not base_languages:
        print("Error: No base languages specified for exploration.")
        return

    best_eps = -1.0
    best_min_samples = -1
    max_complete_clusters_found = -1
    best_config_output_path = None
    best_total_clusters_for_best_config = -1

    results_summary = []

    for eps_val in eps_values:
        for ms_val in min_samples_values:
            run_start_time = time.time()
            print(f"\nRunning DBSCAN with eps={eps_val}, min_samples={ms_val}")

            output_path, total_clusters, complete_clusters_count = perform_dbscan_clustering(
                languages=base_languages,
                eps=eps_val,
                min_samples=ms_val,
                input_dir=base_input_dir,
                output_dir=base_output_dir,
                embedding_key=base_embedding_key
            )
            
            run_duration = time.time() - run_start_time
            print(f"Run eps={eps_val}, min_samples={ms_val} completed in {run_duration:.2f}s. Found {complete_clusters_count} complete clusters out of {total_clusters} total clusters.")
            
            results_summary.append({
                "eps": eps_val, "min_samples": ms_val,
                "total_clusters": total_clusters,
                "complete_clusters": complete_clusters_count,
                "output_file": output_path if output_path else "N/A",
                "duration_s": run_duration
            })

            if complete_clusters_count > max_complete_clusters_found:
                max_complete_clusters_found = complete_clusters_count
                best_eps = eps_val
                best_min_samples = ms_val
                best_config_output_path = output_path
                best_total_clusters_for_best_config = total_clusters
            elif complete_clusters_count == max_complete_clusters_found and complete_clusters_count > 0:
                if eps_val < best_eps:
                    best_eps = eps_val
                    best_min_samples = ms_val
                    best_config_output_path = output_path
                    best_total_clusters_for_best_config = total_clusters
                elif eps_val == best_eps and ms_val < best_min_samples:
                    best_min_samples = ms_val
                    best_config_output_path = output_path
                    best_total_clusters_for_best_config = total_clusters

    print("\n--- DBSCAN Hyperparameter Exploration Summary ---")
    print(f"Target languages for 'complete' clusters: {base_languages} (count: {len(base_languages)})")
    for res in sorted(results_summary, key=lambda x: (-x['complete_clusters'], x['eps'], x['min_samples'])):
        print(f"eps={res['eps']:.3f}, min_samples={res['min_samples']}: {res['complete_clusters']} complete clusters (total: {res['total_clusters']}). File: {res['output_file']}. Took {res['duration_s']:.2f}s")

    if best_eps != -1:
        print(f"\n--- Best DBSCAN Setting Found ---")
        print(f"  eps = {best_eps:.3f}")
        print(f"  min_samples = {best_min_samples}")
        print(f"  Number of clusters with all {len(base_languages)} languages: {max_complete_clusters_found}")
        print(f"  Total non-noise clusters for this setting: {best_total_clusters_for_best_config}")
        if best_config_output_path:
            print(f"  Output file for best config: {best_config_output_path}")
    else:
        print("\nNo suitable DBSCAN configuration found that produced 'complete' clusters, or no clusters were generated.")

    return best_eps, best_min_samples, max_complete_clusters_found

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform DBSCAN-based cross-lingual clustering on word embeddings, with optional hyperparameter exploration.")

    parser.add_argument("--input_dir", "-i", type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing input files ('{{lang}}.json', '{{lang}}_{EMBEDDING_KEY}.pkl'). Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output_dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save the final clustered output file. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS,
                        help=f"DBSCAN eps (max distance for neighborhood, cosine distance). Default: {DEFAULT_EPS}")
    parser.add_argument("--min_samples", "--ms", type=int, default=DEFAULT_MIN_SAMPLES,
                        help=f"DBSCAN min_samples. Default: {DEFAULT_MIN_SAMPLES}")
    parser.add_argument("--embedding_key", type=str, default=EMBEDDING_KEY,
                        help=f"Key name for the embedding vector in the .pkl files. Default: {EMBEDDING_KEY}")
    parser.add_argument("--target_languages", nargs='+', default=TARGET_LANGUAGES,
                        help=f"List of target languages to process. Default: {' '.join(TARGET_LANGUAGES)}")

    parser.add_argument("--run_exploration", action='store_true',
                        help="Run hyperparameter exploration instead of a single run.")
    parser.add_argument("--explore_eps", nargs='+', type=float, default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help="List of eps values to try during exploration.")
    parser.add_argument("--explore_min_samples", nargs='+', type=int, default=[2, 3, 4],
                        help="List of min_samples values to try during exploration.")

    args = parser.parse_args()

    print(f"--- Script Configuration ---")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Embedding key: {args.embedding_key}")
    print(f"Target languages: {args.target_languages}")
    if not args.target_languages:
        print("Error: No target languages specified. Exiting.")
        exit(1)
    
    overall_start_time = time.time()

    if args.run_exploration:
        print(f"Running DBSCAN hyperparameter exploration with eps_values: {args.explore_eps} and min_samples_values: {args.explore_min_samples}")
        explore_hyperparameters(
            base_languages=args.target_languages,
            base_input_dir=args.input_dir,
            base_output_dir=args.output_dir,
            base_embedding_key=args.embedding_key,
            eps_values=args.explore_eps,
            min_samples_values=args.explore_min_samples
        )
    else:
        print(f"Running single DBSCAN clustering with eps={args.eps}, min_samples={args.min_samples}")
        output_file_path, total_clusters, complete_clusters = perform_dbscan_clustering(
            languages=args.target_languages,
            eps=args.eps,
            min_samples=args.min_samples,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            embedding_key=args.embedding_key
        )
        if output_file_path:
            print(f"\nPipeline completed. Final output saved to:")
            print(f"  - {output_file_path}")
            print(f"Found {complete_clusters} clusters containing all {len(args.target_languages)} target languages, out of {total_clusters} total clusters.")
        else:
            print("\nPipeline failed or produced no output (e.g., no embeddings found or no clusters formed).")

    overall_end_time = time.time()
    print(f"\nTotal execution time: {overall_end_time - overall_start_time:.2f} seconds.")
