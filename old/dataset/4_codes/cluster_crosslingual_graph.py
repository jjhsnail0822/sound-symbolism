import json
import argparse
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm
import time
from typing import Dict, List, Any, Optional, Tuple

# --- Constants (Adapted from cluster_crosslingual.py) ---
DEFAULT_INPUT_DIR = "dataset/1_preprocess/nat"
DEFAULT_OUTPUT_DIR = "dataset/1_preprocess/nat/crosslingual_graph_clusters"
DEFAULT_K_NEIGHBORS = 5 # Number of top-k neighbors from each other language
DEFAULT_SIMILARITY_THRESHOLD = 0.8 # Cosine similarity threshold to add an edge
TARGET_LANGUAGES = ["en", "fr", "ja", "ko"]
EMBEDDING_KEY = "en_embedding" # Default embedding key

# --- Helper Functions (Copied from cluster_crosslingual.py) ---
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

# --- Graph-based Clustering Logic ---

def perform_graph_clustering(
    languages: List[str],
    k_neighbors: int,
    similarity_threshold: float,
    input_dir: str,
    output_dir: str,
    embedding_key: str
) -> Tuple[Optional[str], int]: # MODIFIED RETURN TYPE
    """
    Performs graph-based clustering on word embeddings from multiple languages.
    1. Loads embeddings and original data for all specified languages.
    2. For each word, finds top-k most similar words in each of the *other* languages.
    3. Adds an edge if cosine similarity >= threshold.
    4. Forms clusters based on connected components in the graph.
    5. Saves only clusters containing all specified target languages.
    Returns the output file path and the count of saved clusters (those containing all target languages).
    """
    print("\n--- Graph-based Cross-lingual Clustering ---")
    print(f"Parameters: k_neighbors={k_neighbors}, similarity_threshold={similarity_threshold}")
    print(f"Target languages for this run: {languages}")
    start_time = time.time()

    all_word_items: List[Dict[str, Any]] = [] # Stores dicts: {word, lang, original_data, embedding, id}
    word_identifier_set = set() # To avoid duplicates if a word appears multiple times

    # 1. Load Data and Embeddings for all languages
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

        processed_lang_count = 0
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
                processed_lang_count += 1

    if not all_word_items:
        print("Error: No valid embeddings found across languages.")
        return None, 0 # MODIFIED RETURN
    print(f"\nLoaded a total of {len(all_word_items)} words from {len(languages)} languages.")

    # Normalize embeddings for cosine similarity
    for item in all_word_items:
        norm = np.linalg.norm(item['embedding'])
        if norm > 0:
            item['embedding'] = item['embedding'] / norm
        else:
            item['embedding'] = np.zeros_like(item['embedding'])

    # 2. Build Graph
    graph = nx.Graph()
    for item in all_word_items:
        graph.add_node(item['id'], data=item['original_data'])

    embeddings_matrix = np.array([item['embedding'] for item in all_word_items])
    item_details = [{'id': item['id'], 'language': item['language'], 'word': item['word']} for item in all_word_items] # Added 'word' for easier debugging if needed

    for i, source_item in enumerate(tqdm(all_word_items, desc=f"Building graph (k={k_neighbors},t={similarity_threshold})", unit="word")):
        source_id = source_item['id']
        source_lang = source_item['language']
        source_embedding = embeddings_matrix[i].reshape(1, -1)

        # Collect all potential neighbors (not from the source language)
        potential_neighbors_indices_and_langs = []
        for j, target_item_detail in enumerate(item_details):
            if target_item_detail['language'] != source_lang:
                potential_neighbors_indices_and_langs.append({'index': j, 'lang': target_item_detail['language']})
        
        if not potential_neighbors_indices_and_langs:
            continue

        potential_neighbor_global_indices = [p['index'] for p in potential_neighbors_indices_and_langs]
        neighbor_embeddings = embeddings_matrix[potential_neighbor_global_indices]
        
        if neighbor_embeddings.shape[0] == 0: # Should not happen if potential_neighbors_indices_and_langs is not empty
            continue

        similarities = cosine_similarity(source_embedding, neighbor_embeddings)[0]

        # Store all valid neighbors with their similarities and languages
        all_valid_neighbors = []
        for idx_in_potential, sim in enumerate(similarities):
            if sim >= similarity_threshold:
                global_idx = potential_neighbors_indices_and_langs[idx_in_potential]['index']
                target_lang = potential_neighbors_indices_and_langs[idx_in_potential]['lang']
                neighbor_id = item_details[global_idx]['id']
                if source_id != neighbor_id: # Ensure not connecting to itself (though lang check should prevent this)
                    all_valid_neighbors.append({'sim': sim, 'id': neighbor_id, 'lang': target_lang, 'global_idx': global_idx})
        
        # Sort all valid neighbors by similarity
        all_valid_neighbors.sort(key=lambda x: x['sim'], reverse=True)

        selected_neighbor_langs = set()
        selected_count = 0
        
        for neighbor_info in all_valid_neighbors:
            if selected_count >= k_neighbors:
                break # Reached k neighbors overall

            # Check if the language of this neighbor has already been selected for this source_item
            if neighbor_info['lang'] not in selected_neighbor_langs:
                if not graph.has_edge(source_id, neighbor_info['id']):
                    graph.add_edge(source_id, neighbor_info['id'], weight=neighbor_info['sim'])
                selected_neighbor_langs.add(neighbor_info['lang'])
                selected_count += 1

    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    # 3. Find Connected Components (Clusters)
    connected_components = list(nx.connected_components(graph))
    print(f"Found {len(connected_components)} connected components (clusters).")

    # 4. Build the final clustered data structure
    clustered_data_structure = []
    for i, component_nodes in enumerate(tqdm(connected_components, desc="Formatting clusters", unit="cluster", disable=len(connected_components) < 100)):
        cluster_id = i
        words_in_cluster = []
        for node_id in component_nodes:
            word_data = graph.nodes[node_id]['data'].copy()
            word_data['cluster_id'] = cluster_id
            words_in_cluster.append(word_data)
        
        words_in_cluster.sort(key=lambda x: (x.get('language', ''), x.get('word', '')))
        clustered_data_structure.append({'cluster_id': cluster_id, 'words': words_in_cluster})

    # Filter clusters to keep only those containing all target languages
    complete_clusters_data = []
    complete_clusters_count = 0
    if clustered_data_structure:
        num_of_languages_to_check = len(languages)
        for cluster in clustered_data_structure:
            langs_in_cluster = set()
            for word_item in cluster['words']:
                langs_in_cluster.add(word_item['language'])
            
            all_required_langs_present = True
            if not languages: # If no target languages specified, keep all clusters (should not happen with current main logic)
                pass
            else:
                for req_lang in languages:
                    if req_lang not in langs_in_cluster:
                        all_required_langs_present = False
                        break
            
            if all_required_langs_present:
                complete_clusters_data.append(cluster)
                complete_clusters_count += 1
    
    print(f"Found {complete_clusters_count} clusters containing all {len(languages)} specified target languages.")

    # 5. Save Clustered Data (only complete clusters)
    if not complete_clusters_data:
        print("No clusters found containing all target languages. No output file will be saved.")
        return None, 0

    # Re-assign cluster_id for the filtered list
    for new_id, cluster in enumerate(complete_clusters_data):
        cluster['cluster_id'] = new_id
        for word_item in cluster['words']:
            word_item['cluster_id'] = new_id

    output_filename = f'crosslingual_graph_k{k_neighbors}_t{str(similarity_threshold).replace(".", "p")}_clustered.json'
    output_path = os.path.join(output_dir, output_filename)
    save_json(complete_clusters_data, output_path)

    end_time = time.time()
    return output_path, complete_clusters_count

# NEW FUNCTION for hyperparameter exploration
def explore_hyperparameters(
    base_languages: List[str],
    base_input_dir: str,
    base_output_dir: str,
    base_embedding_key: str,
    k_values: List[int],
    t_values: List[float]
):
    print("\n--- Starting Hyperparameter Exploration ---")
    print(f"Target languages for 'complete' clusters: {base_languages} (count: {len(base_languages)})")
    if not base_languages:
        print("Error: No base languages specified for exploration.")
        return

    best_k = -1
    best_t = -1.0
    max_complete_clusters_found = -1
    best_config_output_path = None
    best_total_clusters = -1

    results_summary = []

    for k_val in k_values:
        for t_val in t_values:
            run_start_time = time.time()

            output_path, complete_clusters_count = perform_graph_clustering(
                languages=base_languages,
                k_neighbors=k_val,
                similarity_threshold=t_val,
                input_dir=base_input_dir,
                output_dir=base_output_dir, 
                embedding_key=base_embedding_key
            )
            
            run_duration = time.time() - run_start_time
            print(f"Run k={k_val}, t={t_val} completed in {run_duration:.2f}s. Found {complete_clusters_count} complete clusters.")
            
            num_total_clusters = 0
            if output_path:
                pass

            results_summary.append({
                "k": k_val, "t": t_val, 
                "complete_clusters": complete_clusters_count, 
                "output_file": output_path if output_path else "N/A",
                "duration_s": run_duration
            })

            if complete_clusters_count > max_complete_clusters_found:
                max_complete_clusters_found = complete_clusters_count
                best_k = k_val
                best_t = t_val
                best_config_output_path = output_path
            elif complete_clusters_count == max_complete_clusters_found and complete_clusters_count > 0:
                if t_val > best_t:
                    best_k = k_val
                    best_t = t_val
                    best_config_output_path = output_path
                elif t_val == best_t and k_val < best_k:
                    best_k = k_val
                    best_t = t_val
                    best_config_output_path = output_path

    print("\n--- Hyperparameter Exploration Summary ---")
    print(f"Target languages for 'complete' clusters: {base_languages} (count: {len(base_languages)})")
    for res in sorted(results_summary, key=lambda x: (-x['complete_clusters'], x['t'], x['k'])):
        print(f"k={res['k']}, t={res['t']:.2f}: {res['complete_clusters']} complete clusters. File: {res['output_file']}. Took {res['duration_s']:.2f}s")

    if best_k != -1:
        print(f"\n--- Best Setting Found ---")
        print(f"  k = {best_k}")
        print(f"  t = {best_t:.2f}")
        print(f"  Number of clusters with all {len(base_languages)} languages: {max_complete_clusters_found}")
        if best_config_output_path:
            print(f"  Output file for best config: {best_config_output_path}")
    else:
        print("\nNo suitable configuration found that produced 'complete' clusters, or no clusters were generated.")

    return best_k, best_t, max_complete_clusters_found

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform graph-based cross-lingual clustering on word embeddings, with optional hyperparameter exploration.")

    parser.add_argument("--input_dir", "-i", type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing input files ('{{lang}}.json', '{{lang}}_{EMBEDDING_KEY}.pkl'). Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output_dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save the final clustered output file. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--k_neighbors", "-k", type=int, default=DEFAULT_K_NEIGHBORS,
                        help=f"Number of top-k neighbors to consider from each other language (for single run). Default: {DEFAULT_K_NEIGHBORS}")
    parser.add_argument("--similarity_threshold", "-t", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                        help=f"Cosine similarity threshold for adding an edge (for single run). Default: {DEFAULT_SIMILARITY_THRESHOLD}")
    parser.add_argument("--embedding_key", type=str, default=EMBEDDING_KEY,
                        help=f"Key name for the embedding vector in the .pkl files. Default: {EMBEDDING_KEY}")
    parser.add_argument("--target_languages", nargs='+', default=TARGET_LANGUAGES,
                        help=f"List of target languages to process. Default: {' '.join(TARGET_LANGUAGES)}")

    # Arguments for hyperparameter exploration
    parser.add_argument("--run_exploration", action='store_true',
                        help="Run hyperparameter exploration instead of a single run.")
    parser.add_argument("--explore_k", nargs='+', type=int, default=[1, 2, 3],
                        help="List of k_neighbors values to try during exploration.")
    parser.add_argument("--explore_t", nargs='+', type=float, default=[0.3, 0.4, 0.5, 0.6],
                        help="List of similarity_threshold values to try during exploration.")

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
        print(f"Running hyperparameter exploration with k_values: {args.explore_k} and t_values: {args.explore_t}")
        explore_hyperparameters(
            base_languages=args.target_languages,
            base_input_dir=args.input_dir,
            base_output_dir=args.output_dir,
            base_embedding_key=args.embedding_key,
            k_values=args.explore_k,
            t_values=args.explore_t
        )
    else:
        print(f"Running single clustering with k={args.k_neighbors}, t={args.similarity_threshold}")
        output_file_path, complete_clusters = perform_graph_clustering(
            languages=args.target_languages,
            k_neighbors=args.k_neighbors,
            similarity_threshold=args.similarity_threshold,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            embedding_key=args.embedding_key
        )
        if output_file_path:
            print(f"\nPipeline completed. Final output saved to:")
            print(f"  - {output_file_path}")
            print(f"Found {complete_clusters} clusters containing all {len(args.target_languages)} target languages.")
        else:
            print("\nPipeline failed or produced no output (e.g., no embeddings found).")

    overall_end_time = time.time()
    print(f"\nTotal execution time: {overall_end_time - overall_start_time:.2f} seconds.")
