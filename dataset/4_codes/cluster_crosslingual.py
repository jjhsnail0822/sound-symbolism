import panphon.distance
import json
import argparse
import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import time
from typing import Dict, List, Any, Optional, Set

# --- Constants ---
DEFAULT_INPUT_DIR = "dataset/1_preprocess/nat"
DEFAULT_OUTPUT_DIR = "dataset/1_preprocess/nat"
DEFAULT_NUM_CLUSTERS = 430 # Adjust as needed for cross-lingual
DEFAULT_NUM_DISTRACTORS = 3
RELATIVE_LENGTH_FACTOR = 0.5
TARGET_LANGUAGES = ["fr", "ja", "ko"] # No en, because we test non-english languages with english monolingual LLMs
EMBEDDING_KEY = "en_embedding"

# --- IPA List ---
# (Keep the existing ipa_list)
ipa_list = [
    "a", "aː", "ã", "b", "bʲ", "ç", "d", "d͡z", "d͡ʑ", "d͡ʒ", "e", "eː", "ẽ", "f", "h",
    "i", "iː", "ĩ", "j", "k", "kʰ", "kʲ", "k͈", "l", "m", "mʲ", "n", "o", "oː", "õ",
    "p", "pʰ", "pʲ", "p͈", "s", "s͈", "t", "tʰ", "t͈", "t͈͡ɕ", "t͡s", "t͡ɕ", "t͡ɕʰ", "t͡ʃ",
    "u", "v", "w", "y", "z", "æ", "ð", "ø", "ŋ", "œ", "œ̃", "ɑ", "ɑ̃", "ɔ", "ɔ̃",
    "ɕ", "ɖ", "ɖʲ", "ə", "ɛ", "ɛ̃", "ɡ", "ɡʲ", "ɥ", "ɪ", "ɯ", "ɯː", "ɯ̃", "ɰ", "ɲ",
    "ɴ", "ɸ", "ɹ", "ɹ̩", "ɾ", "ɾʲ", "ʀ", "ʃ", "ʊ", "ʌ", "ʑ", "ʒ", "ʔ", "θ",
]


# --- Global Cache and Distance Object ---
dst = panphon.distance.Distance()
distance_cache = {}

# --- Helper Functions ---
# (load_json, save_json, load_pickle, get_ipa_length remain the same)
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
    Handles potential space-separated segments.
    """
    if not ipa_string or not ipa_string.strip():
        return None
    try:
        # Assume segments might be space-separated, otherwise count characters
        segments = ipa_string.split(" ")
        # Filter out empty strings resulting from multiple spaces
        segments = [seg for seg in segments if seg]
        return len(segments) if segments else 0
    except Exception as e:
        # Fallback or error logging if needed
        return None


# --- Phase 1: Cross-lingual Embedding Clustering (Individual Embeddings) ---

def perform_crosslingual_embedding_clustering_individual(
    languages: List[str],
    num_clusters: int,
    input_dir: str,
    output_dir: str,
    embedding_key: str = EMBEDDING_KEY # Use the specified key
) -> Optional[str]:
    """
    Performs K-means clustering directly on individual word embeddings from multiple
    languages and saves the clustered data.
    """
    # ... (Phase 1 function remains the same) ...
    print("\n--- Phase 1: Cross-lingual Embedding Clustering (Individual Embeddings) ---")
    start_time = time.time()

    all_items_to_cluster = [] # List of tuples: (embedding_vector, lang, word, original_item_data)
    word_identifier_set = set() # To avoid duplicates if a word appears multiple times

    # 1. Load Data and Embeddings for all languages
    for lang in languages:
        print(f"--- Processing language: {lang} ---")
        # Load Embeddings
        embeddings_path = os.path.join(input_dir, f'{lang}_embeddings.pkl')
        print(f"Loading embeddings from {embeddings_path}")
        embeddings_data = load_pickle(embeddings_path)
        if embeddings_data is None:
            print(f"Skipping language {lang} due to missing embeddings file.")
            continue

        # Load Original Data
        data_path = os.path.join(input_dir, f'{lang}.json')
        print(f"Loading original data from {data_path}")
        original_data = load_json(data_path)
        if original_data is None:
            print(f"Skipping language {lang} due to missing original data file.")
            continue

        # Create lookup for original data by word
        word_to_original_item = {item.get('word'): item for item in original_data if isinstance(item, dict)}

        # Process embeddings and link to original data
        processed_lang_count = 0
        for embed_item in embeddings_data:
            word = embed_item.get('word')
            # *** Use the specified embedding_key ***
            embedding_vector = embed_item.get(embedding_key)
            original_item = word_to_original_item.get(word)
            identifier = (lang, word)

            if word and embedding_vector is not None and original_item and identifier not in word_identifier_set:
                # Add language info to the original item data
                original_item_copy = original_item.copy()
                original_item_copy['language'] = lang
                # Optionally remove the large embedding from the item to save space in JSON
                original_item_copy.pop(embedding_key, None)
                original_item_copy.pop('embedding', None) # Remove old key too if present

                all_items_to_cluster.append((np.array(embedding_vector), lang, word, original_item_copy)) # Ensure vector is numpy array
                word_identifier_set.add(identifier)
                processed_lang_count += 1

        print(f"Processed {processed_lang_count} valid embedding entries for {lang}.")


    if not all_items_to_cluster:
        print("Error: No valid embeddings found across languages.")
        return None

    print(f"\nCollected {len(all_items_to_cluster)} total individual word embeddings for clustering.")

    # 2. Prepare data for K-means (using individual embeddings)
    embedding_vectors_list = [item[0] for item in all_items_to_cluster]

    # Check vector consistency before creating the large NumPy array
    first_shape = None
    if embedding_vectors_list:
        first_shape = embedding_vectors_list[0].shape
        if len(first_shape) != 1:
             print(f"Error: Expected 1D embedding vectors, but found shape {first_shape} for the first item.")
             return None
        for i, vec in enumerate(embedding_vectors_list):
            if vec.shape != first_shape:
                print(f"Error: Inconsistent embedding vector shape at index {i}. Expected {first_shape}, got {vec.shape}. Word: {all_items_to_cluster[i][2]}, Lang: {all_items_to_cluster[i][1]}")
                return None

    if not embedding_vectors_list:
        print("Error: No embedding vectors collected.")
        return None

    try:
        embedding_vectors_np = np.stack(embedding_vectors_list, axis=0)
    except ValueError as e:
        print(f"Error stacking embedding vectors: {e}. Check consistency of embedding dimensions.")
        return None


    if embedding_vectors_np.ndim != 2:
         print(f"Error: Problem with embedding dimensions after stacking. Expected 2D array, got shape {embedding_vectors_np.shape}")
         return None


    # 3. Apply K-means Clustering
    print(f"Clustering {embedding_vectors_np.shape[0]} embeddings with K={num_clusters}")
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        # Cluster assignments for each individual embedding vector
        cluster_assignments = kmeans.fit_predict(embedding_vectors_np)
    except Exception as e:
        print(f"Error during K-means clustering: {e}")
        return None

    # 4. Build the final clustered data structure
    print("Building final clustered data structure...")
    clustered_data_structure = [{'cluster_id': i, 'words': []} for i in range(num_clusters)]
    words_added_count = 0

    for i, item_tuple in enumerate(all_items_to_cluster):
        _, lang, word, original_item_data = item_tuple # Don't need embedding vector here
        cluster_id = int(cluster_assignments[i]) # Get cluster ID for this item

        if 0 <= cluster_id < num_clusters:
            # Add the cluster_id to the word's data before appending
            item_data_with_cluster = original_item_data.copy()
            item_data_with_cluster['cluster_id'] = cluster_id
            clustered_data_structure[cluster_id]['words'].append(item_data_with_cluster)
            words_added_count += 1
        else:
            print(f"Warning: Invalid cluster_id {cluster_id} assigned to word '{word}' ({lang}). Skipping.")


    print(f"Added {words_added_count} word entries to clusters.")

    # 5. Save Clustered Data
    output_filename = 'crosslingual_clustered.json' # Indicate individual embedding clustering
    output_path = os.path.join(output_dir, output_filename)
    save_json(clustered_data_structure, output_path)

    end_time = time.time()
    print(f"Phase 1 (Cross-lingual, Individual Embeddings) finished in {end_time - start_time:.2f} seconds.")
    return output_path


# --- Phase 2: Pronunciation-Based Distractor Generation (Configurable Scope) ---

def get_distance(ipa1: str, ipa2: str) -> float:
    # ... (get_distance remains the same) ...
    if not ipa1 or not ipa2 or not ipa1.strip() or not ipa2.strip():
        return -1.0
    key = tuple(sorted((ipa1, ipa2)))
    if key in distance_cache:
        return distance_cache[key]
    try:
        dist_val = dst.jt_feature_edit_distance_div_maxlen(ipa1, ipa2)
        distance_cache[key] = dist_val
        return dist_val
    except Exception as e:
        distance_cache[key] = -1.0
        return -1.0


def find_distant_distractors(
    correct_word_info: Dict[str, Any],
    candidates_by_cluster: Dict[int, List[Dict[str, Any]]], # All candidates across languages, grouped by cluster
    all_cluster_ids: Set[int],
    target_language: str, # Language of the correct word
    distractor_scope: str, # 'same_language' or 'cross_language'
    num_distractors: int = 3,
    relative_length_factor: float = RELATIVE_LENGTH_FACTOR
) -> List[Dict[str, Any]]:
    """
    Finds phonetically distant distractors, with configurable language scope.
    Prioritizes similar IPA length.
    """
    # ... (find_distant_distractors function remains the same) ...
    correct_ipa = correct_word_info.get('ipa')
    correct_cluster_id = correct_word_info.get('cluster_id') # Cluster ID from Phase 1

    # Basic validation
    if correct_cluster_id is None: return []
    correct_ipa_len = get_ipa_length(correct_ipa)
    if correct_ipa_len is None or correct_ipa_len == 0:
        return []

    selected_options_ipa = [correct_ipa]
    distractors_info = []

    # Calculate allowed IPA length range
    allowed_absolute_diff = round(correct_ipa_len * relative_length_factor)
    min_allowed_len = max(1, correct_ipa_len - allowed_absolute_diff)
    max_allowed_len = correct_ipa_len + allowed_absolute_diff

    # --- Build Candidate Pool based on scope ---
    candidate_pool = []
    other_cluster_ids = all_cluster_ids - {correct_cluster_id}

    for cluster_id in other_cluster_ids:
        for word_info in candidates_by_cluster.get(cluster_id, []):
            # Check language scope
            candidate_lang = word_info.get('language')
            if distractor_scope == 'same_language' and candidate_lang != target_language:
                continue # Skip if scope is same_language and candidate is from different lang

            # Basic checks for candidate validity
            ipa = word_info.get('ipa')
            # Ensure it's not the same word/IPA and has a valid length
            # (Comparing word AND language to be safe, though IPA check might suffice)
            is_same_word = (word_info.get('word') == correct_word_info.get('word') and candidate_lang == target_language)
            if ipa and ipa.strip() and ipa != correct_ipa and not is_same_word:
                ipa_len = get_ipa_length(ipa)
                if ipa_len is not None and ipa_len > 0:
                    word_info_copy = word_info.copy()
                    word_info_copy['ipa_len'] = ipa_len
                    candidate_pool.append(word_info_copy)

    if not candidate_pool: return []

    available_candidate_indices = list(range(len(candidate_pool)))

    # --- Select Distractors (logic remains similar) ---
    for _ in range(num_distractors):
        if not available_candidate_indices: break

        best_candidate_index_within_range = -1
        max_min_distance_within_range = -1.0
        best_length_diff_within_range = float('inf')

        best_candidate_index_overall = -1
        max_min_distance_overall = -1.0

        indices_to_remove = set()

        for pool_idx in available_candidate_indices:
            candidate_info = candidate_pool[pool_idx]
            candidate_ipa = candidate_info['ipa']
            candidate_ipa_len = candidate_info['ipa_len']

            min_dist_to_selected = float('inf')
            valid_candidate = True
            for selected_ipa in selected_options_ipa:
                distance = get_distance(candidate_ipa, selected_ipa)
                if distance < 0:
                    valid_candidate = False
                    indices_to_remove.add(pool_idx)
                    break
                min_dist_to_selected = min(min_dist_to_selected, distance)

            if not valid_candidate: continue

            is_within_range = (min_allowed_len <= candidate_ipa_len <= max_allowed_len)
            current_length_diff = abs(correct_ipa_len - candidate_ipa_len)

            if is_within_range:
                if min_dist_to_selected > max_min_distance_within_range:
                    max_min_distance_within_range = min_dist_to_selected
                    best_length_diff_within_range = current_length_diff
                    best_candidate_index_within_range = pool_idx
                elif min_dist_to_selected == max_min_distance_within_range:
                    if current_length_diff < best_length_diff_within_range:
                        best_length_diff_within_range = current_length_diff
                        best_candidate_index_within_range = pool_idx

            if min_dist_to_selected > max_min_distance_overall:
                max_min_distance_overall = min_dist_to_selected
                best_candidate_index_overall = pool_idx

        if indices_to_remove:
            available_candidate_indices = [idx for idx in available_candidate_indices if idx not in indices_to_remove]

        selected_index = -1
        if best_candidate_index_within_range != -1:
             selected_index = best_candidate_index_within_range
        elif best_candidate_index_overall != -1:
             selected_index = best_candidate_index_overall

        if selected_index != -1 and selected_index in available_candidate_indices:
            best_candidate_info = candidate_pool[selected_index]
            selected_options_ipa.append(best_candidate_info['ipa'])
            best_candidate_info.pop('ipa_len', None)
            distractors_info.append(best_candidate_info)
            available_candidate_indices.remove(selected_index)
        else:
            break

    return distractors_info


# --- Modified Phase 2 Function ---
def generate_pronunciation_options_for_language(
    language: str, # The specific language to process
    num_distractors: int,
    distractor_scope: str, # 'same_language' or 'cross_language'
    crosslingual_clustered_data: List[Dict[str, Any]], # Loaded Phase 1 data
    candidates_by_cluster_all: Dict[int, List[Dict[str, Any]]], # Pre-indexed candidates
    all_crosslingual_cluster_ids: Set[int] # Set of all cluster IDs
) -> List[Dict[str, Any]]: # Returns the processed data for this language
    """
    Processes a single language to find distractors based on the scope.
    Does NOT save to file, returns the processed data structure for the language.
    """
    print(f"\n--- Phase 2: Processing Lang: {language}, Scope: {distractor_scope} ---")
    start_time = time.time()

    # Find words belonging to the current target language
    words_to_process_lang: List[Dict[str, Any]] = []
    for cluster in crosslingual_clustered_data:
        for word_info in cluster.get('words', []):
             if word_info.get('language') == language:
                 # Basic validation (IPA check)
                 ipa = word_info.get('ipa')
                 ipa_len = get_ipa_length(ipa)
                 if ipa_len is not None and ipa_len > 0:
                     # Ensure cluster_id is present
                     word_info_copy = word_info.copy()
                     if 'cluster_id' not in word_info_copy:
                          word_info_copy['cluster_id'] = cluster.get('cluster_id')
                     words_to_process_lang.append(word_info_copy)


    print(f"Found {len(words_to_process_lang)} words for language '{language}' to process.")

    if not words_to_process_lang:
         print(f"No words found for language '{language}' to process. Skipping.")
         return [] # Return empty list if no words to process

    # Store results keyed by the cross-lingual cluster ID temporarily
    processed_results_lang_dict: Dict[int, List[Dict[str, Any]]] = {}
    print(f"Finding distractors for '{language}' words sequentially...")
    # distance_cache.clear() # Decide if cache should be cleared per language

    for word_info in tqdm(words_to_process_lang, desc=f"Processing {language} words"):
        correct_info = word_info
        cluster_id = correct_info.get('cluster_id')
        if cluster_id is None:
             print(f"Warning: Missing cluster_id for word '{correct_info.get('word')}' ({language}). Skipping.")
             continue

        # Find distractors using the full candidate pool and scope setting
        selected_distractors = find_distant_distractors(
            correct_info,
            candidates_by_cluster_all, # Pass ALL candidates
            all_crosslingual_cluster_ids,
            target_language=language, # Language of the correct word
            distractor_scope=distractor_scope, # Pass the scope setting
            num_distractors=num_distractors
        )

        # Prepare result structure
        result_info = correct_info.copy()
        result_info['distractors'] = [
            {'word': d.get('word'), 'ipa': d.get('ipa'), 'cluster_id': d.get('cluster_id'), 'language': d.get('language')}
            for d in selected_distractors
        ]

        if cluster_id not in processed_results_lang_dict:
            processed_results_lang_dict[cluster_id] = []
        processed_results_lang_dict[cluster_id].append(result_info)

    processing_time = time.time()
    print(f"Distractor finding for {language} took {processing_time - start_time:.2f} seconds.")
    # print(f"Distance cache size: {len(distance_cache)}") # Optional logging

    # --- Reconstruct the data structure for this language ---
    # This part ensures words that weren't processed (e.g., bad IPA) are still included.
    final_results_data_lang = []
    processed_word_keys_lang = set()

    for cluster_id, words in processed_results_lang_dict.items():
        for word_info in words:
            word = word_info.get('word')
            if word:
                processed_word_keys_lang.add((word, cluster_id))

    for original_cluster in crosslingual_clustered_data:
        cluster_id = original_cluster.get('cluster_id')
        if cluster_id is None: continue

        new_cluster_words_lang = []

        # Add words that were processed (have distractors added)
        if cluster_id in processed_results_lang_dict:
            new_cluster_words_lang.extend(processed_results_lang_dict[cluster_id])

        # Add back any words from this language that were NOT processed
        for original_word_info in original_cluster.get('words', []):
             if not isinstance(original_word_info, dict): continue
             if original_word_info.get('language') == language:
                 word = original_word_info.get('word')
                 # Check if it was processed; use cluster_id from original_cluster as fallback
                 current_cluster_id = original_word_info.get('cluster_id', cluster_id)
                 if word and (word, current_cluster_id) not in processed_word_keys_lang:
                     original_word_info_copy = original_word_info.copy()
                     original_word_info_copy.pop('distractors', None)
                     original_word_info_copy['cluster_id'] = current_cluster_id # Ensure cluster_id
                     new_cluster_words_lang.append(original_word_info_copy)

        # Only add the cluster if it contains words from this language
        if new_cluster_words_lang:
            final_results_data_lang.append({'cluster_id': cluster_id, 'words': new_cluster_words_lang})

    return final_results_data_lang


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform cross-lingual clustering on individual embeddings and generate pronunciation-based distractors with configurable scope.")

    parser.add_argument("--input_dir", "-i", type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing input files ({'{lang}'}.json, {'{lang}'}_{EMBEDDING_KEY}.pkl). Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output_dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save intermediate and final output files. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--num_clusters", "-k", type=int, default=DEFAULT_NUM_CLUSTERS,
                        help=f"Number of clusters for K-means (Phase 1). Default: {DEFAULT_NUM_CLUSTERS}")
    parser.add_argument("--num_distractors", "-n", type=int, default=DEFAULT_NUM_DISTRACTORS,
                        help=f"Number of distractors to generate (Phase 2). Default: {DEFAULT_NUM_DISTRACTORS}")
    # *** Argument for Distractor Scope ***
    parser.add_argument("--distractor_scope", "-d", type=str, default="same_language", choices=["same_language", "cross_language"],
                        help="Scope for selecting distractors: 'same_language' (default) or 'cross_language'.")
    # *** Argument for Embedding Key ***
    parser.add_argument("--embedding_key", type=str, default=EMBEDDING_KEY,
                        help=f"Key name for the embedding vector in the .pkl files. Default: {EMBEDDING_KEY}")


    args = parser.parse_args()

    # Update EMBEDDING_KEY based on argument
    EMBEDDING_KEY = args.embedding_key
    print(f"Using embedding key: {EMBEDDING_KEY}")


    overall_start_time = time.time()

    # --- Run Phase 1: Cross-lingual Clustering (Individual Embeddings) ---
    crosslingual_clustered_file_path = perform_crosslingual_embedding_clustering_individual(
        languages=TARGET_LANGUAGES,
        num_clusters=args.num_clusters,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        embedding_key=EMBEDDING_KEY # Pass the key
    )

    # --- Run Phase 2: Distractor Generation ---
    if crosslingual_clustered_file_path:
        print(f"\n--- Starting Phase 2: Distractor Generation (Scope: {args.distractor_scope}) ---")
        phase2_start_time = time.time()

        # Load the clustered data once
        print(f"Loading clustered data from {crosslingual_clustered_file_path}")
        crosslingual_clustered_data = load_json(crosslingual_clustered_file_path)

        if crosslingual_clustered_data is None:
            print("Error: Failed to load clustered data for Phase 2.")
        else:
            # Pre-index all candidates by cluster ID once
            print("Pre-indexing all candidates by cluster ID...")
            candidates_by_cluster_all: Dict[int, List[Dict[str, Any]]] = {}
            all_crosslingual_cluster_ids: Set[int] = set()
            for cluster in tqdm(crosslingual_clustered_data, desc="Indexing candidates"):
                cluster_id = cluster.get('cluster_id')
                if cluster_id is None: continue
                all_crosslingual_cluster_ids.add(cluster_id)
                cluster_words_list = []
                for word_info in cluster.get('words', []):
                    if isinstance(word_info, dict):
                         ipa = word_info.get('ipa')
                         ipa_len = get_ipa_length(ipa)
                         if ipa_len is not None and ipa_len > 0:
                             word_info_copy = word_info.copy()
                             # Ensure cluster_id is in word_info for find_distant_distractors
                             word_info_copy['cluster_id'] = cluster_id
                             cluster_words_list.append(word_info_copy)
                if cluster_words_list:
                    candidates_by_cluster_all[cluster_id] = cluster_words_list

            all_language_results = [] # To store results if combining
            final_output_paths = [] # To store paths if saving separately

            # Process each language
            for lang in TARGET_LANGUAGES:
                # Call the modified function which returns data instead of saving
                lang_processed_data = generate_pronunciation_options_for_language(
                    language=lang,
                    num_distractors=args.num_distractors,
                    distractor_scope=args.distractor_scope,
                    crosslingual_clustered_data=crosslingual_clustered_data, # Pass loaded data
                    candidates_by_cluster_all=candidates_by_cluster_all, # Pass indexed candidates
                    all_crosslingual_cluster_ids=all_crosslingual_cluster_ids
                )

                if lang_processed_data:
                    if args.distractor_scope == "cross_language":
                        # Append results for combining later
                        all_language_results.append(lang_processed_data)
                    else: # 'same_language' scope - save separately
                        output_filename = f"crosslingual_clustered_{lang}.json"
                        output_file_path = os.path.join(args.output_dir, output_filename)
                        save_json(lang_processed_data, output_file_path)
                        final_output_paths.append(output_file_path)
                else:
                     print(f"No processable data generated for language '{lang}'.")


            # --- Combine and Save if scope is cross_language ---
            if args.distractor_scope == "cross_language" and all_language_results:
                print("\nCombining results for all languages (cross_language scope)...")
                combined_results_by_cluster_id = {}
                # Iterate through results from each language
                for lang_result_list in all_language_results:
                    # Iterate through clusters within that language's results
                    for cluster_data in lang_result_list:
                        cluster_id = cluster_data['cluster_id']
                        if cluster_id not in combined_results_by_cluster_id:
                            combined_results_by_cluster_id[cluster_id] = {'cluster_id': cluster_id, 'words': []}
                        # Add words from this language's cluster to the combined cluster
                        combined_results_by_cluster_id[cluster_id]['words'].extend(cluster_data['words'])

                # Convert the dictionary back to a list of clusters
                final_combined_results = list(combined_results_by_cluster_id.values())
                # Sort by cluster_id for consistency (optional)
                final_combined_results.sort(key=lambda x: x['cluster_id'])

                # Save the combined file
                output_filename = f"crosslingual_clustered_{args.distractor_scope}.json"
                output_file_path = os.path.join(args.output_dir, output_filename)
                save_json(final_combined_results, output_file_path)
                final_output_paths.append(output_file_path) # Add the combined path to the list

            # --- Report Final Results ---
            phase2_end_time = time.time()
            print(f"\nPhase 2 finished in {phase2_end_time - phase2_start_time:.2f} seconds.")
            if final_output_paths:
                print(f"\nPipeline completed Phase 2. Final output(s):")
                for path in final_output_paths:
                    print(f"  - {path}")
            else:
                print("\nPipeline finished Phase 1, but Phase 2 produced no output.")

    else:
        print("\nPipeline stopped because Phase 1 (Cross-lingual Clustering) failed.")

    overall_end_time = time.time()
    print(f"\nTotal execution time: {overall_end_time - overall_start_time:.2f} seconds.")
