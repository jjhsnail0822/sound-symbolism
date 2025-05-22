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
DEFAULT_INPUT_DIR = "dataset/1_preprocess/nat"
DEFAULT_OUTPUT_DIR = "dataset/1_preprocess/nat"
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
        segments = ipa_string.split(" ")
        return len(segments) if segments else 0
    except Exception as e:
        return None

# --- Phase 1: Embedding Clustering ---

def perform_embedding_clustering(language: str, num_clusters: int, input_dir: str, output_dir: str) -> Optional[str]:
    """
    Performs K-means clustering on word embeddings and saves the clustered data.

    Args:
        language: Language code (e.g., 'ko').
        num_clusters: Number of clusters for K-means.
        input_dir: Directory containing embeddings and original data.
        output_dir: Directory to save the clustered JSON file.

    Returns:
        The path to the saved clustered JSON file, or None if an error occurred.
    """
    print("\n--- Phase 1: Embedding Clustering ---")
    start_time = time.time()

    # 1. Load Embeddings
    embeddings_path = os.path.join(input_dir, f'{language}_embeddings.pkl')
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = load_pickle(embeddings_path)
    if embeddings is None: return None

    # 2. Load Original Data
    data_path = os.path.join(input_dir, f'{language}.json')
    print(f"Loading original data from {data_path}")
    original_data = load_json(data_path)
    if original_data is None: return None

    # 3. Extract Embedding Vectors
    try:
        embedding_vectors = np.array([item['embedding'] for item in embeddings])
    except KeyError:
        print("Error: 'embedding' key not found in one or more items in the embeddings file.")
        return None
    except Exception as e:
        print(f"Error extracting embedding vectors: {e}")
        return None

    # 4. Apply K-means Clustering
    print(f"Clustering with K={num_clusters}")
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embedding_vectors)
    except Exception as e:
        print(f"Error during K-means clustering: {e}")
        return None

    # 5. Merge Cluster Information
    print("Merging cluster information with original data")
    word_to_cluster = {}
    for i, item in enumerate(embeddings):
        try:
            word_to_cluster[item['word']] = int(clusters[i])
        except KeyError:
            print(f"Warning: 'word' key not found in embeddings item at index {i}. Skipping.")
            continue

    clustered_data_structure = [{'cluster_id': i, 'words': []} for i in range(num_clusters)]
    words_added_count = 0
    for item in original_data:
        if not isinstance(item, dict):
            print(f"Warning: Skipping non-dictionary item in original data: {item}")
            continue
        if item.get('found', False) is False:
            continue
        word = item.get('word')
        if word and word in word_to_cluster:
            cluster_id = word_to_cluster[word]
            if 0 <= cluster_id < num_clusters:
                clustered_data_structure[cluster_id]['words'].append(item)
                words_added_count += 1
            else:
                print(f"Warning: Invalid cluster_id {cluster_id} assigned to word '{word}'. Skipping.")
        elif word:
            pass

    print(f"Added {words_added_count} words to clusters.")

    # 6. Save Clustered Data
    output_filename = f'{language}_clustered.json'
    output_path = os.path.join(output_dir, output_filename)
    save_json(clustered_data_structure, output_path)

    end_time = time.time()
    print(f"Phase 1 finished in {end_time - start_time:.2f} seconds.")
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
        dist_val = dst.jt_feature_edit_distance_div_maxlen(ipa1, ipa2)
        distance_cache[key] = dist_val
        return dist_val
    except Exception as e:
        distance_cache[key] = -1.0
        return -1.0

def find_distant_distractors(
    correct_word_info: Dict[str, Any],
    candidates_by_cluster: Dict[int, List[Dict[str, Any]]],
    all_cluster_ids: Set[int],
    num_distractors: int = 3,
    relative_length_factor: float = RELATIVE_LENGTH_FACTOR
) -> List[Dict[str, Any]]:
    """
    Finds phonetically distant distractors, prioritizing those with relatively similar IPA length.
    Falls back to distance-only selection if no candidates meet the length criteria.
    """
    correct_ipa = correct_word_info.get('ipa')
    correct_cluster_id = correct_word_info.get('cluster_id')

    if correct_cluster_id is None: return []
    correct_ipa_len = get_ipa_length(correct_ipa)
    if correct_ipa_len is None or correct_ipa_len == 0:
        return []

    selected_options_ipa = [correct_ipa]
    distractors_info = []

    allowed_absolute_diff = round(correct_ipa_len * relative_length_factor)
    min_allowed_len = max(1, correct_ipa_len - allowed_absolute_diff)
    max_allowed_len = correct_ipa_len + allowed_absolute_diff

    candidate_pool = []
    other_cluster_ids = all_cluster_ids - {correct_cluster_id}
    for cluster_id in other_cluster_ids:
        for word_info in candidates_by_cluster.get(cluster_id, []):
            ipa = word_info.get('ipa')
            if ipa and ipa.strip() and ipa != correct_ipa:
                ipa_len = get_ipa_length(ipa)
                if ipa_len is not None and ipa_len > 0:
                    word_info_copy = word_info.copy()
                    word_info_copy['ipa_len'] = ipa_len
                    candidate_pool.append(word_info_copy)

    if not candidate_pool: return []

    available_candidate_indices = list(range(len(candidate_pool)))

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

            if not valid_candidate:
                continue

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

def generate_pronunciation_options(
    language: str,
    num_distractors: int,
    clustered_data_path: str,
    output_dir: str
) -> Optional[str]:
    """
    Loads clustered data, finds phonetically distant distractors (considering relative length),
    and saves the final results.
    """
    print("\n--- Phase 2: Pronunciation-Based Distractor Generation ---")
    start_time = time.time()

    print(f"Loading intermediate clustered data from {clustered_data_path}")
    clustered_data = load_json(clustered_data_path)
    if clustered_data is None: return None

    load_time = time.time()
    print(f"Data loading took {load_time - start_time:.2f} seconds.")

    print("Pre-processing candidates by cluster...")
    candidates_by_cluster: Dict[int, List[Dict[str, Any]]] = {}
    all_cluster_ids: Set[int] = set()
    words_to_process_sequentially: List[Tuple[Dict[str, Any], int]] = []

    for cluster in tqdm(clustered_data, desc="Indexing clusters"):
        cluster_id = cluster.get('cluster_id')
        if cluster_id is None: continue
        all_cluster_ids.add(cluster_id)
        cluster_words_list = []
        for word_info in cluster.get('words', []):
            if not isinstance(word_info, dict): continue
            ipa = word_info.get('ipa')
            ipa_len = get_ipa_length(ipa)
            if ipa_len is not None and ipa_len > 0:
                word_info_with_id = word_info.copy()
                word_info_with_id['cluster_id'] = cluster_id
                cluster_words_list.append(word_info_with_id)
        candidates_by_cluster[cluster_id] = cluster_words_list

    for cluster_id, words_list in candidates_by_cluster.items():
         for word_info in words_list:
              words_to_process_sequentially.append((word_info, cluster_id))

    preprocess_time = time.time()
    print(f"Found {len(words_to_process_sequentially)} words with valid IPA to process across {len(all_cluster_ids)} clusters.")
    print(f"Pre-processing took {preprocess_time - load_time:.2f} seconds.")

    processed_results: Dict[int, List[Dict[str, Any]]] = {}
    print(f"Processing words sequentially with distance caching and relative length constraint...")
    distance_cache.clear()

    for word_info, cluster_id in tqdm(words_to_process_sequentially, desc="Processing words"):
        correct_info = word_info

        selected_distractors = find_distant_distractors(
            correct_info,
            candidates_by_cluster,
            all_cluster_ids,
            num_distractors=num_distractors
        )

        result_info = correct_info.copy()
        result_info['distractors'] = [
            {'word': d.get('word'), 'ipa': d.get('ipa'), 'cluster_id': d.get('cluster_id')}
            for d in selected_distractors
        ]

        if cluster_id not in processed_results:
            processed_results[cluster_id] = []
        processed_results[cluster_id].append(result_info)

    processing_time = time.time()
    print(f"Sequential processing took {processing_time - preprocess_time:.2f} seconds.")
    print(f"Distance cache size: {len(distance_cache)}")

    print("Reconstructing results...")
    final_results_data = []
    processed_word_keys = set()

    for cluster_id, words in processed_results.items():
        for word_info in words:
            word = word_info.get('word')
            if word:
                processed_word_keys.add((word, cluster_id))

    for original_cluster in clustered_data:
        cluster_id = original_cluster.get('cluster_id')
        if cluster_id is None: continue

        new_cluster_words = []
        if cluster_id in processed_results:
            new_cluster_words.extend(processed_results[cluster_id])

        for original_word_info in original_cluster.get('words', []):
             if not isinstance(original_word_info, dict): continue
             word = original_word_info.get('word')
             if word and (word, cluster_id) not in processed_word_keys:
                 original_word_info_copy = original_word_info.copy()
                 original_word_info_copy.pop('distractors', None)
                 new_cluster_words.append(original_word_info_copy)

        if new_cluster_words:
            final_results_data.append({'cluster_id': cluster_id, 'words': new_cluster_words})
        elif not original_cluster.get('words'):
             final_results_data.append({'cluster_id': cluster_id, 'words': []})

    reconstruct_time = time.time()
    print(f"Result reconstruction took {reconstruct_time - processing_time:.2f} seconds.")

    output_filename = f"{language}_clustered.json"
    output_file_path = os.path.join(output_dir, output_filename)
    save_json(final_results_data, output_file_path)

    end_time = time.time()
    print(f"Phase 2 finished in {end_time - start_time:.2f} seconds.")
    return output_file_path

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster word embeddings and generate pronunciation-based distractors.")

    parser.add_argument("--lang", "-l", type=str, required=True,
                        help="Language code (e.g., 'en', 'ko', 'ja', 'fr').")
    parser.add_argument("--input_dir", "-i", type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing input files ({'{lang}'}.json, {'{lang}'}_embeddings.pkl). Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output_dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save intermediate and final output files. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--num_clusters", "-k", type=int, default=DEFAULT_NUM_CLUSTERS,
                        help=f"Number of clusters for K-means (Phase 1). Default: {DEFAULT_NUM_CLUSTERS}")
    parser.add_argument("--num_distractors", "-n", type=int, default=DEFAULT_NUM_DISTRACTORS,
                        help=f"Number of distractors to generate for each word (Phase 2). Default: {DEFAULT_NUM_DISTRACTORS}")

    args = parser.parse_args()

    overall_start_time = time.time()

    intermediate_clustered_file_path = perform_embedding_clustering(
        language=args.lang,
        num_clusters=args.num_clusters,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )

    if intermediate_clustered_file_path:
        final_output_path = generate_pronunciation_options(
            language=args.lang,
            num_distractors=args.num_distractors,
            clustered_data_path=intermediate_clustered_file_path,
            output_dir=args.output_dir
        )
        if final_output_path:
             print(f"\nPipeline completed successfully. Final output: {final_output_path}")
        else:
             print("\nPipeline finished, but Phase 2 encountered errors.")
    else:
        print("\nPipeline stopped because Phase 1 failed.")

    overall_end_time = time.time()
    print(f"\nTotal execution time: {overall_end_time - overall_start_time:.2f} seconds.")