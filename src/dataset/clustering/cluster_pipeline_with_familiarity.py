import panphon.distance
import json
import argparse
import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import time
from typing import Dict, List, Any, Optional, Set, Tuple

# --- Constants ---
DEFAULT_INPUT_DIR = "data/processed/nat"
DEFAULT_EMBEDDING_INPUT_DIR = "data/processed/nat/clustering"
DEFAULT_OUTPUT_DIR = "data/processed/nat/clustering"
DEFAULT_NUM_CLUSTERS = 430
DEFAULT_NUM_DISTRACTORS = 3
RELATIVE_LENGTH_FACTOR = 0.5  # Allow length difference up to 50%

# --- IPA List (from cluster_pronunciations) ---
ipa_list = [
    "a", "aː", "ã", "b", "bʲ", "ç", "d", "d͡z", "d͡ʑ", "d͡ʒ", "e", "eː", "ẽ", "f", "h",
    "i", "iː", "ĩ", "j", "k", "kʰ", "kʲ", "k͈", "l", "m", "mʲ", "n", "o", "oː", "õ",
    "p", "pʰ", "pʲ", "p͈", "s", "s͈", "t", "tʰ", "t͈", "t͈͡ɕ", "t͡s", "t͡ɕ", "t͡ɕʰ", "t͡ʃ",
    "u", "v", "w", "y", "z", "æ", "ð", "ø", "ŋ", "œ", "œ̃", "ɑ", "ɑ̃", "ɔ", "ɔ̃",
    "ɕ", "ɖ", "ɖʲ", "ə", "ɛ", "ɛ̃", "ɡ", "ɡʲ", "ɥ", "ɪ", "ɯ", "ɯː", "ɯ̃", "ɰ", "ɲ",
    "ɴ", "ɸ", "ɹ", "ɹ̩", "ɾ", "ɾʲ", "ʀ", "ʃ", "ʊ", "ʌ", "ʑ", "ʒ", "ʔ", "θ",
]

# --- Global Cache and Distance Object (from cluster_pronunciations) ---
dst = panphon.distance.Distance()
distance_cache = {}  # Cache for storing computed distances

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

def get_ipa_length(ipa_string: Optional[str]) -> Optional[int]:
    """
    Calculates the number of segments in an IPA string using panphon.

    Args:
        ipa_string: The IPA string to analyze.

    Returns:
        The number of segments, or None if input is invalid or segmentation fails.
    """
    if not ipa_string or not ipa_string.strip():
        return None
    try:
        segments = ipa_string.split(" ") # Assuming space-separated segments
        return len(segments) if segments else 0
    except Exception: # Broad exception for safety, panphon might raise various errors
        return None

# --- Phase 1: Embedding Clustering ---

def perform_embedding_clustering(
    familiarity_type: str,
    num_clusters: int,
    input_dir: str,
    input_embedding_dir: str,
    output_dir: str
) -> Optional[str]:
    """
    Performs K-means clustering on word embeddings for a given familiarity type
    (e.g., 'common', 'rare') and saves the clustered data.
    Embeddings are loaded from 'all_embeddings.pkl' and filtered.
    Word data is loaded from '{familiarity_type}_words.json'.

    Args:
        familiarity_type: 'common' or 'rare'.
        num_clusters: Number of clusters for K-means.
        input_dir: Directory containing '{familiarity_type}_words.json' and 'all_embeddings.pkl'.
        input_embedding_dir: Directory containing the embeddings (if different from input_dir).
        output_dir: Directory to save the intermediate clustered JSON file.

    Returns:
        The path to the saved intermediate clustered JSON file, or None if an error occurred.
    """
    print(f"\n--- Phase 1: Embedding Clustering for '{familiarity_type}' words ---")
    start_time = time.time()

    # 1. Load Original Word Data (e.g., common_words.json)
    data_path = os.path.join(input_dir, f'{familiarity_type}_words.json')
    print(f"Loading original data from {data_path}")
    original_data_full = load_json(data_path)
    if original_data_full is None:
        return None
    if not isinstance(original_data_full, list):
        print(f"Error: Expected a list of words in {data_path}, got {type(original_data_full)}")
        return None

    target_words_set = {item['word'] for item in original_data_full if isinstance(item, dict) and 'word' in item}
    if not target_words_set:
        print(f"No words found in {data_path}. Skipping clustering for {familiarity_type}.")
        return None

    # 2. Load All Embeddings
    all_embeddings_path = os.path.join(input_embedding_dir, 'all_embeddings.pkl')
    print(f"Loading all embeddings from {all_embeddings_path}")
    all_embeddings_list = load_pickle(all_embeddings_path)
    if all_embeddings_list is None:
        return None
    if not isinstance(all_embeddings_list, list):
        print(f"Error: Expected a list of embeddings in {all_embeddings_path}, got {type(all_embeddings_list)}")
        return None

    # 3. Filter embeddings for target words and prepare for clustering
    embeddings_for_clustering = [] # Stores {'word': ..., 'embedding': ...} for words in original_data_full
    
    valid_embeddings_map = {} # word -> embedding
    for emb_item in all_embeddings_list:
        if isinstance(emb_item, dict) and 'word' in emb_item and 'en_embedding' in emb_item:
            if emb_item['word'] in target_words_set:
                 valid_embeddings_map[emb_item['word']] = emb_item['en_embedding']
        else:
            print(f"Warning: Skipping invalid embedding item: {emb_item}")


    # Build embeddings_for_clustering based on words present in original_data_full
    # This ensures the order of embeddings matches the words we intend to cluster
    # and that we only cluster words for which we have original data.
    for word_data_item in original_data_full:
        if not isinstance(word_data_item, dict):
            print(f"Warning: Skipping non-dictionary item in original_data_full: {word_data_item}")
            continue
        word = word_data_item.get('word')
        if word and word in valid_embeddings_map:
            embeddings_for_clustering.append({'word': word, 'embedding': valid_embeddings_map[word]})
        # elif word:
            # print(f"Warning: Word '{word}' from {familiarity_type}_words.json not found in all_embeddings.pkl or has no embedding.")

    if not embeddings_for_clustering:
        print(f"No words with embeddings found for '{familiarity_type}' after filtering. Cannot perform clustering.")
        return None
    
    print(f"Found {len(embeddings_for_clustering)} words with embeddings for '{familiarity_type}'.")

    try:
        embedding_vectors = np.array([item['embedding'] for item in embeddings_for_clustering])
    except Exception as e:
        print(f"Error extracting embedding vectors for '{familiarity_type}': {e}")
        return None

    # 4. Apply K-means Clustering
    print(f"Clustering '{familiarity_type}' words with K={num_clusters}")
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto' if hasattr(KMeans(), 'n_init') and KMeans().n_init == 'warn' else 10)
        cluster_labels = kmeans.fit_predict(embedding_vectors)
    except Exception as e:
        print(f"Error during K-means clustering for '{familiarity_type}': {e}")
        return None

    # 5. Merge Cluster Information
    print(f"Merging cluster information for '{familiarity_type}'")
    word_to_cluster_id = {}
    for i, item in enumerate(embeddings_for_clustering): # Iterate over the list used for clustering
        word_to_cluster_id[item['word']] = int(cluster_labels[i])

    clustered_data_structure = [{'cluster_id': i, 'words': []} for i in range(num_clusters)]
    words_added_count = 0

    # Iterate through the original_data_full to preserve all original fields
    # and add words to their respective clusters.
    for original_word_item in original_data_full:
        if not isinstance(original_word_item, dict):
            continue # Already warned above
        
        word = original_word_item.get('word')
        if word and word in word_to_cluster_id:
            cluster_id = word_to_cluster_id[word]
            if 0 <= cluster_id < num_clusters:
                # Add the original item (which includes 'word', 'ipa', 'language', etc.)
                clustered_data_structure[cluster_id]['words'].append(original_word_item)
                words_added_count += 1
            else:
                print(f"Warning: Invalid cluster_id {cluster_id} assigned to word '{word}' in '{familiarity_type}'. Skipping.")
        # elif word:
            # Word was in original_data_full but not in embeddings_for_clustering (no embedding or other issue)
            # These words will not be part of any cluster.
            # print(f"Info: Word '{word}' from {data_path} was not clustered (e.g. no embedding).")


    print(f"Added {words_added_count} words to clusters for '{familiarity_type}'.")

    # 6. Save Clustered Data (Intermediate)
    output_filename = f'crosslingual_intermediate_clustered_{familiarity_type}.json'
    output_path = os.path.join(output_dir, output_filename)
    save_json(clustered_data_structure, output_path)

    end_time = time.time()
    print(f"Phase 1 for '{familiarity_type}' finished in {end_time - start_time:.2f} seconds.")
    return output_path

# --- Phase 2: Pronunciation-Based Distractor Generation ---

def get_distance(ipa1: str, ipa2: str) -> float:
    """
    Calculates the distance between two IPA strings using a cache.
    Returns -1.0 if calculation fails or input is invalid.
    """
    if not ipa1 or not ipa2 or not ipa1.strip() or not ipa2.strip():
        return -1.0

    key = tuple(sorted((ipa1, ipa2)))

    if key in distance_cache:
        return distance_cache[key]

    try:
        # Ensure segments are space-separated if panphon expects that
        # Assuming ipa strings are already correctly formatted (e.g. "p a n")
        dist_val = dst.jt_feature_edit_distance_div_maxlen(ipa1, ipa2)
        distance_cache[key] = dist_val
        return dist_val
    except Exception: # Catching generic exception from panphon
        distance_cache[key] = -1.0 # Cache failure
        return -1.0

def find_distant_distractors(
    correct_word_info: Dict[str, Any],
    candidates_by_cluster: Dict[int, List[Dict[str, Any]]],
    all_cluster_ids: Set[int],
    num_distractors: int = 3,
    relative_length_factor: float = RELATIVE_LENGTH_FACTOR
) -> List[Dict[str, Any]]:
    """
    Finds phonetically distant distractors from other clusters, prioritizing those with
    relatively similar IPA length. This function is cross-lingual as candidates
    can be from any language within the same familiarity type.
    """
    correct_ipa = correct_word_info.get('ipa')
    # 'cluster_id_assigned_for_correct_word' is the key for the correct word's cluster ID
    correct_cluster_id = correct_word_info.get('cluster_id_assigned_for_correct_word') 

    if correct_cluster_id is None or correct_ipa is None:
        # print(f"Warning: Missing 'cluster_id_assigned_for_correct_word' or 'ipa' in correct_word_info: {correct_word_info.get('word')}")
        return []

    correct_ipa_len = get_ipa_length(correct_ipa)
    if correct_ipa_len is None or correct_ipa_len == 0:
        # print(f"Warning: Could not get valid IPA length for '{correct_ipa}' (word: {correct_word_info.get('word')})")
        return []

    selected_options_ipa = [correct_ipa]
    distractors_info = []

    allowed_absolute_diff = round(correct_ipa_len * relative_length_factor)
    min_allowed_len = max(1, correct_ipa_len - allowed_absolute_diff)
    max_allowed_len = correct_ipa_len + allowed_absolute_diff

    candidate_pool = []
    other_cluster_ids = all_cluster_ids - {correct_cluster_id}

    for cid in other_cluster_ids:
        for word_info in candidates_by_cluster.get(cid, []):
            ipa = word_info.get('ipa')
            if ipa and ipa.strip() and ipa != correct_ipa and word_info.get('word') != correct_word_info.get('word'):
                ipa_len = get_ipa_length(ipa)
                if ipa_len is not None and ipa_len > 0:
                    word_info_copy = word_info.copy()
                    word_info_copy['ipa_len'] = ipa_len
                    # word_info_copy should have 'original_cluster_id' from the preprocessing step
                    candidate_pool.append(word_info_copy)

    if not candidate_pool:
        return []

    available_candidate_indices = list(range(len(candidate_pool)))

    for _ in range(num_distractors):
        if not available_candidate_indices: break

        best_candidate_index_within_range = -1
        max_min_distance_within_range = -1.0
        best_length_diff_within_range = float('inf')

        best_candidate_index_overall = -1
        max_min_distance_overall = -1.0

        indices_to_remove_from_available = set() 

        for pool_idx_offset, actual_pool_idx in enumerate(available_candidate_indices):
            candidate_info = candidate_pool[actual_pool_idx]
            candidate_ipa = candidate_info['ipa']
            candidate_ipa_len = candidate_info['ipa_len'] 

            min_dist_to_selected = float('inf')
            valid_candidate_for_distances = True
            for selected_ipa_opt in selected_options_ipa:
                distance = get_distance(candidate_ipa, selected_ipa_opt)
                if distance < 0: 
                    valid_candidate_for_distances = False
                    indices_to_remove_from_available.add(actual_pool_idx)
                    break
                min_dist_to_selected = min(min_dist_to_selected, distance)

            if not valid_candidate_for_distances:
                continue

            is_within_range = (min_allowed_len <= candidate_ipa_len <= max_allowed_len)
            current_length_diff = abs(correct_ipa_len - candidate_ipa_len)

            if is_within_range:
                if min_dist_to_selected > max_min_distance_within_range:
                    max_min_distance_within_range = min_dist_to_selected
                    best_length_diff_within_range = current_length_diff
                    best_candidate_index_within_range = actual_pool_idx
                elif min_dist_to_selected == max_min_distance_within_range:
                    if current_length_diff < best_length_diff_within_range:
                        best_length_diff_within_range = current_length_diff
                        best_candidate_index_within_range = actual_pool_idx
            
            if min_dist_to_selected > max_min_distance_overall:
                max_min_distance_overall = min_dist_to_selected
                best_candidate_index_overall = actual_pool_idx

        if indices_to_remove_from_available:
            available_candidate_indices = [idx for idx in available_candidate_indices if idx not in indices_to_remove_from_available]
            if not available_candidate_indices: break

        selected_actual_index = -1
        if best_candidate_index_within_range != -1:
             selected_actual_index = best_candidate_index_within_range
        elif best_candidate_index_overall != -1: 
             selected_actual_index = best_candidate_index_overall
        
        if selected_actual_index != -1 and selected_actual_index in available_candidate_indices:
            best_candidate_info = candidate_pool[selected_actual_index]
            selected_options_ipa.append(best_candidate_info['ipa'])
            distractor_to_add = {
                'word': best_candidate_info.get('word'),
                'ipa': best_candidate_info.get('ipa'),
                'language': best_candidate_info.get('language'), 
                'cluster_id': best_candidate_info.get('original_cluster_id') # Use the correct key
            }
            distractors_info.append(distractor_to_add)
            available_candidate_indices.remove(selected_actual_index)
        else:
            break
            
    return distractors_info


def generate_pronunciation_options(
    familiarity_type: str,
    num_distractors: int,
    clustered_data_path: str, 
    output_dir: str
) -> Optional[str]:
    """
    Loads intermediate clustered data for a familiarity type, finds phonetically
    distant distractors (cross-lingually within the familiarity type),
    and saves the final results.
    """
    print(f"\n--- Phase 2: Pronunciation-Based Distractor Generation for '{familiarity_type}' ---")
    start_time = time.time()

    print(f"Loading intermediate clustered data from {clustered_data_path}")
    intermediate_clustered_data = load_json(clustered_data_path)
    if intermediate_clustered_data is None: return None

    load_time = time.time()
    print(f"Data loading for '{familiarity_type}' took {load_time - start_time:.2f} seconds.")

    print(f"Pre-processing candidates by cluster for '{familiarity_type}'...")
    candidates_by_cluster: Dict[int, List[Dict[str, Any]]] = {}
    all_cluster_ids: Set[int] = set()
    words_to_process_sequentially: List[Tuple[Dict[str, Any], int]] = []

    for cluster_data_item in tqdm(intermediate_clustered_data, desc=f"Indexing clusters for {familiarity_type}"):
        cluster_id = cluster_data_item.get('cluster_id')
        if cluster_id is None:
            print(f"Warning: Cluster item missing 'cluster_id': {cluster_data_item}")
            continue
        all_cluster_ids.add(cluster_id)
        
        current_cluster_words_list = []
        for word_info in cluster_data_item.get('words', []):
            if not isinstance(word_info, dict):
                print(f"Warning: Non-dict word_info skipped: {word_info}")
                continue
            
            ipa = word_info.get('ipa')
            ipa_len = get_ipa_length(ipa) 
            
            if ipa_len is not None and ipa_len > 0:
                word_info_copy = word_info.copy() 
                # Key for the correct word's cluster ID when it's passed to find_distant_distractors
                word_info_copy['cluster_id_assigned_for_correct_word'] = cluster_id 
                # Key for the candidate word's own original cluster ID
                word_info_copy['original_cluster_id'] = cluster_id 
                current_cluster_words_list.append(word_info_copy)
                # The first element of the tuple is the word_info dict, second is its cluster_id (for correct_word_info)
                words_to_process_sequentially.append((word_info_copy, cluster_id))
            # else:
                # print(f"Info: Word '{word_info.get('word')}' in cluster {cluster_id} has invalid/empty IPA '{ipa}'. Not considered for distractor generation or as a target.")

        candidates_by_cluster[cluster_id] = current_cluster_words_list

    preprocess_time = time.time()
    num_words_to_process = len(words_to_process_sequentially)
    num_clusters_found = len(all_cluster_ids)
    print(f"Found {num_words_to_process} words with valid IPA to process across {num_clusters_found} clusters for '{familiarity_type}'.")
    print(f"Pre-processing for '{familiarity_type}' took {preprocess_time - load_time:.2f} seconds.")

    if num_words_to_process == 0:
        print(f"No words to process for distractor generation in '{familiarity_type}'. Saving empty/original cluster structure.")
        output_filename = f"crosslingual_clustered_{familiarity_type}.json"
        output_file_path = os.path.join(output_dir, output_filename)
        save_json(intermediate_clustered_data, output_file_path)
        print(f"Phase 2 for '{familiarity_type}' finished early. Final output (same as intermediate): {output_file_path}")
        return output_file_path

    final_results_by_cluster: Dict[int, List[Dict[str, Any]]] = {cid: [] for cid in all_cluster_ids}
    
    print(f"Processing words for '{familiarity_type}' with distance caching and relative length constraint...")
    distance_cache.clear() 

    for correct_word_info_with_temp_keys, original_cluster_id_of_correct_word in tqdm(words_to_process_sequentially, desc=f"Generating distractors for {familiarity_type}"):
        # correct_word_info_with_temp_keys already has 'cluster_id_assigned_for_correct_word' and 'original_cluster_id'
        
        selected_distractors = find_distant_distractors(
            correct_word_info_with_temp_keys, 
            candidates_by_cluster, 
            all_cluster_ids,
            num_distractors=num_distractors,
            relative_length_factor=RELATIVE_LENGTH_FACTOR
        )

        result_word_info = correct_word_info_with_temp_keys.copy() 
        result_word_info['distractors'] = selected_distractors 
        
        final_results_by_cluster[original_cluster_id_of_correct_word].append(result_word_info)

    processing_time = time.time()
    print(f"Distractor generation for '{familiarity_type}' took {processing_time - preprocess_time:.2f} seconds.")
    print(f"Distance cache size for '{familiarity_type}': {len(distance_cache)}")

    final_output_data_list = []
    for cid in sorted(list(all_cluster_ids)): 
        processed_words_for_cluster = final_results_by_cluster.get(cid, [])
        
        original_words_in_cluster = []
        for cluster_item_from_input in intermediate_clustered_data:
            if cluster_item_from_input.get('cluster_id') == cid:
                original_words_in_cluster = cluster_item_from_input.get('words', [])
                break
        
        processed_word_ids = {pw.get('word') for pw in processed_words_for_cluster}
        
        final_words_for_this_cluster = list(processed_words_for_cluster) 

        for orig_word_info in original_words_in_cluster:
            if orig_word_info.get('word') not in processed_word_ids:
                orig_word_info_copy = orig_word_info.copy()
                orig_word_info_copy.pop('distractors', None)
                # Ensure original words (not processed) also don't have temporary keys
                orig_word_info_copy.pop('cluster_id_assigned_for_correct_word', None) 
                orig_word_info_copy.pop('original_cluster_id', None)
                final_words_for_this_cluster.append(orig_word_info_copy)
        
        # Clean up temporary keys from all words before saving
        # Decide if 'original_cluster_id' should be kept for the main word object
        for word_entry in final_words_for_this_cluster:
            word_entry.pop('cluster_id_assigned_for_correct_word', None)
            # If you don't want the main word object to have 'original_cluster_id':
            # word_entry.pop('original_cluster_id', None) 
            # Or, if you want to keep it, ensure it's consistently named.
            # For now, let's assume the main word object itself doesn't store its own cluster_id,
            # as it's implied by the parent structure. Distractors *do* store their original_cluster_id.

        if final_words_for_this_cluster or not original_words_in_cluster : 
             final_output_data_list.append({'cluster_id': cid, 'words': final_words_for_this_cluster})

    reconstruct_time = time.time()
    print(f"Result reconstruction for '{familiarity_type}' took {reconstruct_time - processing_time:.2f} seconds.")

    output_filename = f"crosslingual_clustered_{familiarity_type}.json"
    output_file_path = os.path.join(output_dir, output_filename)
    save_json(final_output_data_list, output_file_path)

    end_time = time.time()
    print(f"Phase 2 for '{familiarity_type}' finished in {end_time - start_time:.2f} seconds.")
    return output_file_path

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster word embeddings (common/rare) and generate cross-lingual pronunciation-based distractors."
    )

    # Removed --lang argument
    parser.add_argument("--input_dir", "-i", type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing 'common_words.json', 'rare_words.json'. Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--input_embedding_dir", "-e", type=str, default=DEFAULT_EMBEDDING_INPUT_DIR,
                        help=f"Directory containing 'all_embeddings.pkl'. Default: {DEFAULT_EMBEDDING_INPUT_DIR}")
    parser.add_argument("--output_dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save intermediate and final output files. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--num_clusters", "-k", type=int, default=DEFAULT_NUM_CLUSTERS,
                        help=f"Number of clusters for K-means (Phase 1). Default: {DEFAULT_NUM_CLUSTERS}")
    parser.add_argument("--num_distractors", "-n", type=int, default=DEFAULT_NUM_DISTRACTORS,
                        help=f"Number of distractors to generate for each word (Phase 2). Default: {DEFAULT_NUM_DISTRACTORS}")
    parser.add_argument("--process_types", "-p", type=str, nargs='+', default=["common", "rare"],
                        choices=["common", "rare"],
                        help="Specify which familiarity types to process ('common', 'rare', or both). Default: common rare")


    args = parser.parse_args()
    overall_start_time = time.time()

    # familiarity_types_to_process = ["common", "rare"]
    familiarity_types_to_process = args.process_types


    for f_type in familiarity_types_to_process:
        print(f"\n===== Processing familiarity type: {f_type} =====")
        
        # Phase 1
        intermediate_clustered_file_path = perform_embedding_clustering(
            familiarity_type=f_type,
            num_clusters=args.num_clusters,
            input_dir=args.input_dir,
            input_embedding_dir=args.input_embedding_dir,
            output_dir=args.output_dir
        )

        if intermediate_clustered_file_path:
            # Phase 2
            final_output_path = generate_pronunciation_options(
                familiarity_type=f_type,
                num_distractors=args.num_distractors,
                clustered_data_path=intermediate_clustered_file_path, # Output from Phase 1
                output_dir=args.output_dir
            )
            if final_output_path:
                 print(f"\nPipeline for '{f_type}' completed successfully. Final output: {final_output_path}")
            else:
                 print(f"\nPipeline for '{f_type}' finished, but Phase 2 encountered errors.")
        else:
            print(f"\nPipeline for '{f_type}' stopped because Phase 1 failed.")

    overall_end_time = time.time()
    print(f"\nTotal execution time for all specified types: {overall_end_time - overall_start_time:.2f} seconds.")