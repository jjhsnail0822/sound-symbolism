import json
import argparse
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time
from typing import Dict, List, Any, Optional, Tuple, Set

# --- Constants (Adapted from cluster_crosslingual.py) ---
DEFAULT_INPUT_DIR = "dataset/1_preprocess/nat"
DEFAULT_OUTPUT_DIR = "dataset/1_preprocess/nat/crosslingual_association_clusters"
DEFAULT_SIMILARITY_THRESHOLD = 0.5
TARGET_LANGUAGES = ["en", "fr", "ja", "ko"]
EMBEDDING_KEY = "en_embedding"
DEFAULT_BASE_LANGUAGE = "en"

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

# --- Association-based Clustering Logic ---

def _calculate_cluster_centroids_and_collect_words(
    languages: List[str],
    all_clustered_data_input: Dict[str, List[Dict]],
    all_embeddings_data_per_word_input: Dict[str, Dict[str, np.ndarray]],
    embedding_key: str
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[int, List[Dict]]]]:
    """
    For each language, calculates the centroid embedding for each of its clusters.
    Also collects all word objects for each cluster.
    """
    all_centroids: Dict[str, Dict[int, np.ndarray]] = {lang: {} for lang in languages}
    all_word_objs_by_cluster: Dict[str, Dict[int, List[Dict]]] = {lang: {} for lang in languages}

    for lang in languages:
        if lang not in all_clustered_data_input or not all_clustered_data_input[lang]:
            print(f"Warning: No raw clustered data found for language {lang} when calculating centroids.")
            continue
        if lang not in all_embeddings_data_per_word_input:
            print(f"Warning: No embeddings data found for language {lang} when calculating centroids.")
            continue

        lang_embeddings_map = all_embeddings_data_per_word_input[lang]

        for cluster_info in all_clustered_data_input[lang]:
            cluster_id = cluster_info.get('cluster_id')
            word_objects_in_cluster_raw = cluster_info.get('words', [])

            if cluster_id is None or not word_objects_in_cluster_raw:
                continue

            cluster_embeddings_vectors = []
            valid_word_objects_for_centroid = []

            for word_obj_raw in word_objects_in_cluster_raw:
                word_str = word_obj_raw.get('word')
                if word_str and word_str in lang_embeddings_map:
                    cluster_embeddings_vectors.append(lang_embeddings_map[word_str])
                    valid_word_objects_for_centroid.append(word_obj_raw)
            
            if cluster_embeddings_vectors:
                centroid = np.mean(np.array(cluster_embeddings_vectors), axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    all_centroids[lang][cluster_id] = centroid / norm
                else: # Handle zero-norm centroid (e.g. if all embeddings were zero)
                    all_centroids[lang][cluster_id] = np.zeros_like(centroid)
                
                # Store all original word objects for this cluster, regardless of whether they had embeddings
                all_word_objs_by_cluster[lang][cluster_id] = word_objects_in_cluster_raw
            else:
                print(f"Warning: No valid embeddings found for words in cluster {cluster_id} of language {lang}. Centroid not calculated.")

    return all_centroids, all_word_objs_by_cluster


def perform_association_clustering(
    base_language: str,
    target_languages: List[str], # Should include base_language
    similarity_threshold: float,
    input_dir: str,
    output_dir: str,
    embedding_key: str
) -> Tuple[Optional[str], int]:
    """
    Performs cross-lingual clustering by associating clusters from a base language
    with the most similar, unused clusters from other target languages,
    prioritizing higher similarity pairings globally.
    """
    print("\n--- Association-based Cross-lingual Clustering (Cluster-to-Cluster, Similarity Prioritized) ---")
    print(f"Base language for driving associations: {base_language}")
    print(f"All target languages for final clusters: {target_languages}")
    print(f"Similarity threshold for cluster association: {similarity_threshold}")
    start_time = time.time()

    # 1. Load all necessary data (same as before)
    all_embeddings_data_per_word: Dict[str, Dict[str, np.ndarray]] = {}
    all_raw_clustered_data: Dict[str, List[Dict]] = {}
    all_original_word_details: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for lang in tqdm(target_languages):
        embeddings_path = os.path.join(input_dir, f'{lang}_embeddings.pkl')
        lang_embeddings_list = load_pickle(embeddings_path)
        if lang_embeddings_list:
            current_lang_embeddings = {}
            for item in lang_embeddings_list:
                word = item.get('word')
                embedding_vector = item.get(embedding_key)
                if word and embedding_vector is not None:
                    emb_array = np.array(embedding_vector, dtype=np.float32)
                    norm = np.linalg.norm(emb_array)
                    current_lang_embeddings[word] = emb_array / norm if norm > 0 else np.zeros_like(emb_array)
            all_embeddings_data_per_word[lang] = current_lang_embeddings
        else:
            print(f"Warning: Could not load embeddings for language {lang}.")
            # If base language embeddings are missing, we can't proceed
            if lang == base_language:
                print(f"Error: Embeddings for base language {lang} are crucial and missing.")
                return None, 0


        lang_cluster_filepath = os.path.join(input_dir, f"{lang}_clustered.json")
        loaded_lang_clusters = load_json(lang_cluster_filepath)
        if loaded_lang_clusters:
            all_raw_clustered_data[lang] = loaded_lang_clusters
        else:
            print(f"Error: Could not load clustered data from {lang_cluster_filepath} for language {lang}.")
            return None, 0

        original_data_path = os.path.join(input_dir, f'{lang}.json')
        original_lang_data_list = load_json(original_data_path)
        if original_lang_data_list:
            all_original_word_details[lang] = {
                item.get('word'): item for item in original_lang_data_list if isinstance(item, dict) and item.get('word')
            }
        else:
            print(f"Warning: Original .json data for '{lang}' not loaded. Attempting to use data from '{lang}_clustered.json'.")
            if lang in all_raw_clustered_data:
                temp_original_data = {}
                for cluster_item in all_raw_clustered_data[lang]:
                    for word_item in cluster_item.get('words', []):
                        if 'word' in word_item and word_item['word'] not in temp_original_data :
                            temp_original_data[word_item['word']] = word_item
                if temp_original_data:
                    all_original_word_details[lang] = temp_original_data
                else:
                     print(f"Warning: Could not reconstruct original details for {lang} from its clustered file.")
            else:
                 print(f"Warning: No data to reconstruct original details for {lang}.")


    # 2. Calculate centroids for all clusters in all languages (same as before)
    all_cluster_centroids, all_cluster_word_objects = _calculate_cluster_centroids_and_collect_words(
        target_languages, all_raw_clustered_data, all_embeddings_data_per_word, embedding_key
    )

    if not all_cluster_centroids.get(base_language):
        print(f"Error: No cluster centroids could be calculated for the base language '{base_language}'. Cannot proceed.")
        return None, 0
    for lang in target_languages:
        if lang != base_language and not all_cluster_centroids.get(lang):
            print(f"Warning: No cluster centroids found for target language '{lang}'. It may not be possible to form complete clusters.")


    # 3. New Association Logic: Calculate all potential pairings and sort by similarity
    potential_pairings = []
    base_lang_centroids = all_cluster_centroids.get(base_language, {})

    for base_cluster_id, base_centroid_vec in tqdm(base_lang_centroids.items()):
        for assoc_lang in target_languages:
            if assoc_lang == base_language:
                continue
            
            assoc_lang_centroids = all_cluster_centroids.get(assoc_lang, {})
            for target_id, target_centroid_vec in assoc_lang_centroids.items():
                similarity = cosine_similarity(base_centroid_vec.reshape(1, -1), target_centroid_vec.reshape(1, -1))[0][0]
                if similarity >= similarity_threshold: # Pre-filter by threshold
                    potential_pairings.append({
                        "base_cluster_id": base_cluster_id,
                        "assoc_lang": assoc_lang,
                        "target_cluster_id": target_id,
                        "similarity": float(similarity) # Convert numpy.float32 to Python float
                    })
    
    # Sort by similarity in descending order
    potential_pairings.sort(key=lambda x: x["similarity"], reverse=True)

    final_crosslingual_clusters: List[Dict[str, Any]] = []
    new_cluster_id_counter = 0
    
    # Keep track of used base clusters and target language clusters
    # {base_cluster_id: {assoc_lang: target_cluster_id}}
    assigned_base_clusters: Dict[int, Dict[str, int]] = {}
    # {assoc_lang: {target_cluster_id_1, target_cluster_id_2}}
    used_target_lang_original_cluster_ids: Dict[str, Set[int]] = {
        lang: set() for lang in target_languages if lang != base_language
    }

    for pairing in tqdm(potential_pairings, desc="Assigning best pairings"):
        base_c_id = pairing["base_cluster_id"]
        assoc_l = pairing["assoc_lang"]
        target_c_id = pairing["target_cluster_id"]

        # Check if base cluster already has a partner for this assoc_lang
        if base_c_id in assigned_base_clusters and assoc_l in assigned_base_clusters[base_c_id]:
            continue 
        
        # Check if this target language cluster is already used
        if target_c_id in used_target_lang_original_cluster_ids[assoc_l]:
            continue

        # If checks pass, make the assignment
        if base_c_id not in assigned_base_clusters:
            assigned_base_clusters[base_c_id] = {}
        assigned_base_clusters[base_c_id][assoc_l] = target_c_id
        used_target_lang_original_cluster_ids[assoc_l].add(target_c_id)

    # 4. Assemble final clusters based on assignments
    for base_cluster_id, associations in tqdm(assigned_base_clusters.items(), desc="Assembling final clusters"):
        current_assembled_cluster_words: List[Dict[str, Any]] = []
        
        # Add words from the base language cluster
        base_cluster_original_words = all_cluster_word_objects.get(base_language, {}).get(base_cluster_id, [])
        for word_obj in base_cluster_original_words:
            full_word_detail = all_original_word_details.get(base_language, {}).get(word_obj['word'], word_obj).copy()
            full_word_detail['language'] = base_language
            full_word_detail.pop('cluster_id', None)
            current_assembled_cluster_words.append(full_word_detail)

        # Check if this base cluster successfully found partners for ALL other target languages
        all_targets_found = True
        for req_assoc_lang in target_languages:
            if req_assoc_lang == base_language:
                continue
            if req_assoc_lang not in associations:
                all_targets_found = False
                break
        
        if not all_targets_found:
            continue # Skip this base_cluster if it couldn't find partners for all required languages

        chosen_target_clusters_for_current_base_info = {} # For metadata

        for assoc_lang, target_cluster_id in associations.items():
            target_cluster_original_words = all_cluster_word_objects.get(assoc_lang, {}).get(target_cluster_id, [])
            for word_obj in target_cluster_original_words:
                full_word_detail = all_original_word_details.get(assoc_lang, {}).get(word_obj['word'], word_obj).copy()
                full_word_detail['language'] = assoc_lang
                full_word_detail.pop('cluster_id', None)
                current_assembled_cluster_words.append(full_word_detail)
            
            # For metadata: find the original similarity score for this chosen pair
            # This is a bit inefficient here, could be stored better if needed frequently
            original_sim = -1
            for pp in potential_pairings: # Search in original sorted list
                if pp['base_cluster_id'] == base_cluster_id and pp['assoc_lang'] == assoc_lang and pp['target_cluster_id'] == target_cluster_id:
                    original_sim = pp['similarity']
                    break
            chosen_target_clusters_for_current_base_info[f"{assoc_lang}_{target_cluster_id}"] = original_sim


        # Final check: ensure all target_languages are indeed in the assembled words
        # (This should be guaranteed by the 'all_targets_found' check earlier, but as a safeguard)
        languages_in_assembled_cluster = set(item['language'] for item in current_assembled_cluster_words)
        if all(tl in languages_in_assembled_cluster for tl in target_languages):
            current_assembled_cluster_words.sort(key=lambda x: (x.get('language', ''), x.get('word', '')))
            for item in current_assembled_cluster_words:
                item['crosslingual_cluster_id'] = new_cluster_id_counter
            
            final_crosslingual_clusters.append({
                'crosslingual_cluster_id': new_cluster_id_counter,
                'words': current_assembled_cluster_words,
                'source_base_cluster_id': f"{base_language}_{base_cluster_id}",
                'associated_target_clusters': chosen_target_clusters_for_current_base_info
            })
            new_cluster_id_counter += 1

    print(f"Found {len(final_crosslingual_clusters)} cross-lingual clusters containing all {len(target_languages)} target languages.")

    if not final_crosslingual_clusters:
        print("No complete cross-lingual clusters found. No output file will be saved.")
        return None, 0

    output_filename = f'crosslingual_clusterassoc_bl-{base_language}_t{str(similarity_threshold).replace(".", "p")}_sortedsim_clustered.json' # Added _sortedsim
    output_path = os.path.join(output_dir, output_filename)
    save_json(final_crosslingual_clusters, output_path)
    
    end_time = time.time()
    print(f"Association clustering (sorted similarity) took {end_time - start_time:.2f} seconds.")
    return output_path, len(final_crosslingual_clusters)

def explore_hyperparameters(
    base_language_for_exploration: str,
    target_languages_for_exploration: List[str],
    base_input_dir: str,
    base_output_dir: str,
    base_embedding_key: str,
    t_values: List[float]
):
    print("\n--- Starting Hyperparameter Exploration for Cluster-to-Cluster Association ---")
    print(f"Base language for clusters: {base_language_for_exploration}")
    print(f"Target languages for 'complete' clusters: {target_languages_for_exploration} (count: {len(target_languages_for_exploration)})")
    
    if not target_languages_for_exploration or base_language_for_exploration not in target_languages_for_exploration:
        print("Error: Invalid language configuration for exploration.")
        return [], -1.0, -1 # Return structure indicating failure or no results

    best_t = -1.0
    max_complete_clusters_found = -1
    best_config_output_path = None
    
    results_summary = []

    for t_val in t_values:
        run_start_time = time.time()
        print(f"\nExploring with t = {t_val:.2f}")
        # Call the updated perform_association_clustering function
        output_path, complete_clusters_count = perform_association_clustering(
            base_language=base_language_for_exploration,
            target_languages=target_languages_for_exploration,
            similarity_threshold=t_val,
            input_dir=base_input_dir,
            output_dir=base_output_dir,
            embedding_key=base_embedding_key
        )
        run_duration = time.time() - run_start_time
        print(f"Run t={t_val:.2f} completed in {run_duration:.2f}s. Found {complete_clusters_count} complete clusters.")
        
        results_summary.append({
            "t": t_val, 
            "complete_clusters": complete_clusters_count, 
            "output_file": output_path if output_path else "N/A",
            "duration_s": run_duration
        })

        if complete_clusters_count > max_complete_clusters_found:
            max_complete_clusters_found = complete_clusters_count
            best_t = t_val
            best_config_output_path = output_path
        elif complete_clusters_count == max_complete_clusters_found and complete_clusters_count >= 0: 
            if t_val > best_t: # Prefer higher threshold for tie-breaking if it yields same max clusters
                best_t = t_val
                best_config_output_path = output_path
    
    print("\n--- Hyperparameter Exploration Summary ---")
    print(f"Base language: {base_language_for_exploration}, Target languages for completion: {target_languages_for_exploration}")
    # Sort results for display: by most complete clusters, then by highest t_val
    for res in sorted(results_summary, key=lambda x: (-x['complete_clusters'], -x['t'])): 
        print(f"t={res['t']:.2f}: {res['complete_clusters']} complete clusters. File: {res['output_file']}. Took {res['duration_s']:.2f}s")

    if best_t != -1.0 and max_complete_clusters_found != -1:
        print(f"\n--- Best Setting Found ---")
        print(f"  Similarity Threshold (t) = {best_t:.2f}")
        print(f"  Number of clusters with all {len(target_languages_for_exploration)} languages: {max_complete_clusters_found}")
        if best_config_output_path:
            print(f"  Output file for best config: {best_config_output_path}")
    else:
        print("\nNo suitable configuration found that produced 'complete' clusters, or no clusters were generated.")

    return results_summary, best_t, max_complete_clusters_found

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform association-based cross-lingual clustering (cluster-to-cluster) using a base language's pre-clustered file and associating with other languages' pre-clustered files.")

    parser.add_argument("--input_dir", "-i", type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing input files ('{{lang}}.json', '{{lang}}_clustered.json', '{{lang}}_{EMBEDDING_KEY}.pkl' for all target languages). Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output_dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save the final clustered output file. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--similarity_threshold", "-t", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                        help=f"Cosine similarity threshold for associating cluster centroids. Default: {DEFAULT_SIMILARITY_THRESHOLD}")
    parser.add_argument("--embedding_key", type=str, default=EMBEDDING_KEY,
                        help=f"Key name for the embedding vector in the .pkl files. Default: {EMBEDDING_KEY}")
    
    parser.add_argument("--base_language", type=str, default=DEFAULT_BASE_LANGUAGE,
                        help=f"The language code to drive associations (e.g., 'en'). Default: {DEFAULT_BASE_LANGUAGE}")
    parser.add_argument("--target_languages", nargs='+', default=TARGET_LANGUAGES,
                        help=f"List of all target languages for final clusters (MUST include base_language, and all must have {{lang}}_clustered.json). Default: {' '.join(TARGET_LANGUAGES)}")

    parser.add_argument("--run_exploration", action='store_true',
                        help="Run hyperparameter exploration instead of a single run.")
    parser.add_argument("--explore_t", nargs='+', type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        help="List of similarity_threshold values to try during exploration.")

    args = parser.parse_args()

    print(f"--- Script Configuration ---")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Embedding key: {args.embedding_key}")
    print(f"Base language: {args.base_language}")
    print(f"Target languages: {args.target_languages}")

    if not args.target_languages:
        print("Error: No target languages specified. Exiting.")
        exit(1)
    if args.base_language not in args.target_languages:
        print(f"Error: Base language '{args.base_language}' must be included in the list of target languages '{args.target_languages}'. Exiting.")
        exit(1)
    
    # Check for required {lang}_clustered.json files for all target languages
    for lang_check in args.target_languages:
        expected_cluster_file = os.path.join(args.input_dir, f"{lang_check}_clustered.json")
        if not os.path.exists(expected_cluster_file):
            print(f"Error: Required cluster file not found for language '{lang_check}': {expected_cluster_file}. This file is needed for cluster-to-cluster association.")
            exit(1)
        # Also check for embeddings, though the function has warnings
        expected_embedding_file = os.path.join(args.input_dir, f"{lang_check}_embeddings.pkl")
        if not os.path.exists(expected_embedding_file):
             print(f"Warning: Embedding file not found for language '{lang_check}': {expected_embedding_file}. Centroid calculation might be affected.")

    overall_start_time = time.time()

    if args.run_exploration:
        print(f"Running hyperparameter exploration with t_values: {args.explore_t}")
        explore_hyperparameters(
            base_language_for_exploration=args.base_language,
            target_languages_for_exploration=args.target_languages,
            base_input_dir=args.input_dir,
            base_output_dir=args.output_dir,
            base_embedding_key=args.embedding_key,
            t_values=args.explore_t
        )
    else:
        print(f"Running single cluster-to-cluster association with base_language='{args.base_language}', t={args.similarity_threshold}")
        output_file_path, complete_clusters = perform_association_clustering(
            base_language=args.base_language,
            target_languages=args.target_languages,
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
            print("\nPipeline failed or produced no output (e.g., no complete clusters found or error during processing).")

    overall_end_time = time.time()
    print(f"\nTotal execution time: {overall_end_time - overall_start_time:.2f} seconds.")
