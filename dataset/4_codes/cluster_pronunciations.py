import panphon.distance
import json
import argparse
import os
from tqdm import tqdm
import time  # For timing

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
distance_cache = {}  # Cache for storing computed distances

# --- Function Definitions ---

def get_distance(ipa1, ipa2):
    """
    Calculates the distance between two IPA strings using a cache.
    Returns -1.0 if calculation fails or input is invalid.
    """
    # Basic validation
    if not ipa1 or not ipa2 or not ipa1.strip() or not ipa2.strip():
        return -1.0

    # Create a canonical key (sorted tuple) for the cache
    key = tuple(sorted((ipa1, ipa2)))

    # Check cache first
    if key in distance_cache:
        return distance_cache[key]

    # If not in cache, calculate, store, and return
    try:
        dist_val = dst.jt_feature_edit_distance_div_maxlen(ipa1, ipa2)
        distance_cache[key] = dist_val
        return dist_val
    except Exception as e:
        distance_cache[key] = -1.0  # Cache the error indicator
        return -1.0

def find_distant_distractors(correct_word_info, candidates_by_cluster, all_cluster_ids, num_distractors=3):
    """
    Finds phonetically most distant distractors using pre-processed candidates and distance caching.
    Args:
        correct_word_info (dict): Information about the correct word {'word': ..., 'ipa': ..., 'cluster_id': ...}
        candidates_by_cluster (dict): Dictionary mapping cluster_id to list of word_info in that cluster.
        all_cluster_ids (set): Set of all valid cluster IDs.
        num_distractors (int): The number of distractors to find

    Returns:
        list: List of selected distractor word information (dictionaries)
    """
    correct_ipa = correct_word_info.get('ipa')
    correct_cluster_id = correct_word_info.get('cluster_id')

    # --- Input Validation ---
    if correct_cluster_id is None: return []
    if not correct_ipa or not correct_ipa.strip(): return []

    selected_options_ipa = [correct_ipa]
    distractors_info = []

    # --- 1. Prepare Candidate Pool ---
    candidate_pool = []
    other_cluster_ids = all_cluster_ids - {correct_cluster_id}
    for cluster_id in other_cluster_ids:
        for word_info in candidates_by_cluster.get(cluster_id, []):
            ipa = word_info.get('ipa')
            # Ensure candidate IPA is valid and different from the correct one
            if ipa and ipa.strip() and ipa != correct_ipa:
                candidate_pool.append(word_info)  # Store the full info dict

    if not candidate_pool: return []  # No valid candidates found

    # --- 2. Iterative Distractor Selection ---
    # Use indices to manage the pool efficiently without costly list deletions
    candidate_indices = list(range(len(candidate_pool)))

    for _ in range(num_distractors):
        if not candidate_indices: break  # Stop if no more candidates

        best_candidate_index_in_pool = -1
        max_min_distance = -1.0

        valid_indices_in_current_iteration = []  # Keep track of valid candidates in this round

        for current_pool_index in candidate_indices:
            candidate_info = candidate_pool[current_pool_index]
            candidate_ipa = candidate_info['ipa']  # Already validated earlier

            min_dist_to_selected = float('inf')
            valid_candidate_for_comparison = True

            # Calculate distance to already selected options using the cache
            for selected_ipa in selected_options_ipa:
                distance = get_distance(candidate_ipa, selected_ipa)  # Use cached function
                if distance < 0:  # Handle calculation errors or invalid IPAs from cache
                    valid_candidate_for_comparison = False
                    break  # No need to compare further if one distance fails
                min_dist_to_selected = min(min_dist_to_selected, distance)

            if not valid_candidate_for_comparison:
                continue  # Skip this candidate for this selection round

            valid_indices_in_current_iteration.append(current_pool_index)

            # Check if this candidate maximizes the minimum distance
            if min_dist_to_selected > max_min_distance:
                max_min_distance = min_dist_to_selected
                best_candidate_index_in_pool = current_pool_index

        # Update candidate_indices for the next iteration (only keep valid ones)
        candidate_indices = [idx for idx in candidate_indices if idx in valid_indices_in_current_iteration]

        # Add the best candidate found in this iteration
        if best_candidate_index_in_pool != -1:
            best_candidate_info = candidate_pool[best_candidate_index_in_pool]
            selected_options_ipa.append(best_candidate_info['ipa'])
            distractors_info.append(best_candidate_info)
            # Remove the selected index from the list for the next iteration
            try:
                candidate_indices.remove(best_candidate_index_in_pool)
            except ValueError:
                pass
        else:
            # No suitable candidate found in this iteration
            break

    return distractors_info[:num_distractors]  # Return up to num_distractors


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find phonetically distant distractors for words in a clustered JSON file.")
    parser.add_argument("--lang", "-l", type=str, required=True, help="Language code (e.g., 'en', 'ko', 'ja') to determine the input JSON file.")
    parser.add_argument("--input_dir", "-i", type=str, default="dataset/1_preprocess/nat", help="Directory containing the input clustered JSON files.")
    parser.add_argument("--output_dir", "-o", type=str, default="dataset/1_preprocess/nat", help="Directory to save the output JSON file with distractors.")
    parser.add_argument("--num_distractors", "-n", type=int, default=3, help="Number of distractors to generate for each word.")

    args = parser.parse_args()

    start_time = time.time()

    # --- 1. Load Data ---
    json_file_path = os.path.join(args.input_dir, f"{args.lang}_clustered.json")
    all_words_data = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            all_words_data = json.load(f)
        print(f"Successfully loaded data from {json_file_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}. Make sure the path and language code are correct.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {json_file_path}. Check file format. Details: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the JSON file: {e}")
        exit(1)

    load_time = time.time()
    print(f"Data loading took {load_time - start_time:.2f} seconds.")

    # --- 1.5 Pre-process candidates ---
    print("Pre-processing candidates by cluster...")
    candidates_by_cluster = {}
    all_cluster_ids = set()
    words_to_process_sequentially = []  # List of tuples: (word_info, cluster_id)

    for cluster in tqdm(all_words_data, desc="Indexing clusters"):
        cluster_id = cluster.get('cluster_id')
        if cluster_id is None: continue

        all_cluster_ids.add(cluster_id)
        cluster_words_list = []  # Store word_info dicts for this cluster
        for word_info in cluster.get('words', []):
            # Add cluster_id to word_info for easier access later
            word_info_with_id = word_info.copy()
            word_info_with_id['cluster_id'] = cluster_id
            cluster_words_list.append(word_info_with_id)
            # Add to the list of words needing processing
            words_to_process_sequentially.append((word_info, cluster_id))  # Pass original word_info

        # Store the list of word_info dicts for this cluster
        candidates_by_cluster[cluster_id] = cluster_words_list

    preprocess_time = time.time()
    print(f"Found {len(words_to_process_sequentially)} words to process across {len(all_cluster_ids)} clusters.")
    print(f"Pre-processing took {preprocess_time - load_time:.2f} seconds.")

    # --- 2. Process all words sequentially ---
    processed_results = {}  # Aggregate results by cluster_id
    print(f"Processing words sequentially with distance caching...")

    for word_info, cluster_id in tqdm(words_to_process_sequentially, desc="Processing words"):
        correct_info = word_info.copy()
        correct_info['cluster_id'] = cluster_id  # Ensure cluster_id is present

        selected_distractors = find_distant_distractors(
            correct_info,
            candidates_by_cluster,
            all_cluster_ids,
            num_distractors=args.num_distractors
        )

        # debug
        print(f"Selected distractors for {correct_info['word']}: {[d['word'] for d in selected_distractors]}")

        # Add distractors to the word info
        correct_info['distractors'] = [
            {'word': d.get('word'), 'ipa': d.get('ipa'), 'cluster_id': d.get('cluster_id')}
            for d in selected_distractors
        ]

        # Aggregate results
        if cluster_id not in processed_results:
            processed_results[cluster_id] = []
        processed_results[cluster_id].append(correct_info)

    processing_time = time.time()
    print(f"Sequential processing took {processing_time - preprocess_time:.2f} seconds.")
    print(f"Distance cache size: {len(distance_cache)}")  # Show cache effectiveness

    # --- 2.5 Reconstruct results data preserving original structure ---
    print("Reconstructing results...")
    final_results_data = []
    for original_cluster in all_words_data:
        cluster_id = original_cluster.get('cluster_id')
        # Check if the cluster was processed and has results
        if cluster_id in processed_results:
            new_cluster = original_cluster.copy()  # Keep original cluster metadata
            # Replace 'words' with the processed words for this cluster
            new_cluster['words'] = processed_results[cluster_id]
            final_results_data.append(new_cluster)

    reconstruct_time = time.time()
    print(f"Result reconstruction took {reconstruct_time - processing_time:.2f} seconds.")

    # --- 3. Save Results ---
    output_filename = f"{args.lang}_clustered_with_options.json"
    output_file_path = os.path.join(args.output_dir, output_filename)
    os.makedirs(args.output_dir, exist_ok=True)  # Ensure output directory exists
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_results_data, f, ensure_ascii=False, indent=4)
        print(f"\nSuccessfully saved results with distractors to {output_file_path}")
    except Exception as e:
        print(f"\nError saving results to {output_file_path}: {e}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
