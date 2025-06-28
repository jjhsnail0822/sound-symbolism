import json
import copy

# Policy
# 1. Unanimous agreement across all three models (GPT-4.1, Gemma-3, Qwen-3)
# 2. GPT-4.1 answer with logit probability >= 0.95
# 3. Highest probability feature among the three models' results

# Binary Ground Truth Generation for Semantic Dimensions
# Removes 'neither' option and focuses on binary features.

def load_json(filepath):
    """Loads JSON data from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """Saves data to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def find_word_in_list(word_list, word_key):
    """Helper function to find a word dictionary by its 'word' key in a list of dictionaries."""
    if not isinstance(word_list, list):
        return None
    for item in word_list:
        if isinstance(item, dict) and item.get("word") == word_key:
            return item
    return None

def determine_final_gt(combined_words_list, gpt4_data, gemma_data, qwen_data):
    # Initialize the final ground truth structure based on combined_words_list.
    # The structure will be a dictionary keyed by language,
    # with values being lists of word objects.
    final_gt = {}
    for item in combined_words_list:
        lang = item.get("language")
        word_val = item.get("word")

        if not lang or not word_val:
            # print(f"Info: Skipping item in combined_words due to missing 'language' or 'word': {str(item)[:100]}")
            continue

        if lang not in final_gt:
            final_gt[lang] = []
        
        # Create a base structure for the word in final_gt by copying the item
        # from combined_words_list to preserve all its original information.
        current_word_obj_for_final_gt = copy.deepcopy(item)
        # Initialize or ensure 'dimensions' key exists for policy logic.
        # This will be filled by policy logic later.
        current_word_obj_for_final_gt["dimensions"] = {} 
        
        # 'word_group' if present in 'item' is already carried over by deepcopy.
        # Other fields like "meaning", "ipa", "ref", "url", "found", 
        # "ipa_source", "romanization", "en_meaning", "infinigram_count" 
        # from 'item' are also preserved.

        final_gt[lang].append(current_word_obj_for_final_gt)

    # Iterate through language keys present in the final_gt (derived from combined_words_list)
    for lang_key, lang_list_in_final_gt in final_gt.items():
        # Get the corresponding list of words for this language from each model's data.
        # If a language key is not present in a model's data, default to an empty list.
        gpt4_lang_list_original = gpt4_data.get(lang_key, [])
        gemma_lang_list = gemma_data.get(lang_key, [])
        qwen_lang_list = qwen_data.get(lang_key, [])
        
        # Iterate through word objects in the final_gt's language list.
        # Each word_obj_in_final_gt originated from an item in combined_words_list.
        for word_obj_in_final_gt in lang_list_in_final_gt:
            actual_word_key = word_obj_in_final_gt["word"] # The word string

            # Find the corresponding word object in the gpt4, Gemma, and Qwen lists
            # for the current language (lang_key).
            # find_word_in_list operates on a list of words assumed to be for a single language.
            gpt4_word_obj_original = find_word_in_list(gpt4_lang_list_original, actual_word_key)
            gemma_word_obj = find_word_in_list(gemma_lang_list, actual_word_key)
            qwen_word_obj = find_word_in_list(qwen_lang_list, actual_word_key)

            # If the word is not found consistently in all three model outputs,
            # its 'dimensions' will remain empty as initialized.
            # The word itself (from combined_words_list) will still be present in the final output.
            if not (gpt4_word_obj_original and gemma_word_obj and qwen_word_obj):
                # print(f"Info: Word '{actual_word_key}' (lang: {lang_key}) not found consistently across all model outputs. Dimensions will be empty.")
                continue # Move to the next word in final_gt for this language

            # Ensure each found word object from the models has a 'dimensions' dictionary
            # and that 'dimensions' itself is a dictionary.
            if not all(isinstance(word_obj, dict) and 
                       "dimensions" in word_obj and 
                       isinstance(word_obj["dimensions"], dict)
                       for word_obj in [gpt4_word_obj_original, gemma_word_obj, qwen_word_obj]):
                # print(f"Warning: 'dimensions' key missing, not a dict, or word data malformed for '{actual_word_key}' (lang: {lang_key}) in model outputs. Dimensions will be empty.")
                continue # Dimensions for this word remain empty; move to the next word.

            new_dimensions_for_current_word = {} # To store dimension results for the current word
            
            dimensions_gpt4 = gpt4_word_obj_original["dimensions"]
            dimensions_gemma = gemma_word_obj["dimensions"]
            dimensions_qwen = qwen_word_obj["dimensions"]

            # Iterate through dimension keys from GPT-4's perspective (as per original logic).
            # This determines which dimensions are considered for policy application.
            for dim_key in dimensions_gpt4: 
                if not isinstance(dim_key, str) or '-' not in dim_key:
                    # print(f"Info: Skipping dimension key '{dim_key}' for word '{actual_word_key}' (lang: {lang_key}) as it does not seem to be a 'feature1-feature2' pair.")
                    continue

                # Check if this dimension key is present in all three models' data for the current word.
                if dim_key not in dimensions_gemma or dim_key not in dimensions_qwen:
                    # print(f"Info: Dimension '{dim_key}' for word '{actual_word_key}' (lang: {lang_key}) not present consistently in all models. Skipping this dimension.")
                    continue
                
                try:
                    feature1_name, feature2_name = dim_key.split('-', 1) 
                except ValueError:
                    # print(f"Warning: Could not parse dimension key '{dim_key}' for word '{actual_word_key}' (lang: {lang_key}). Skipping this dimension.")
                    continue

                gpt4_res = dimensions_gpt4[dim_key]
                gemma_res = dimensions_gemma[dim_key]
                qwen_res = dimensions_qwen[dim_key]

                # Validate the structure of results (logits, answer) from each model for the current dimension.
                if not all(
                    isinstance(res, dict) and
                    "logits" in res and isinstance(res.get("logits"), dict) and
                    # Ensure specific logit keys "1", "2", "3" exist.
                    all(k in res["logits"] for k in ["1", "2", "3"]) and 
                    "answer" in res
                    for res in [gpt4_res, gemma_res, qwen_res]
                ):
                    # print(f"Warning: Logits/answer structure malformed for word '{actual_word_key}', dimension '{dim_key}' (lang: {lang_key}). Skipping this dimension.")
                    continue

                chosen_feature = None
                policy_applied = None

                # Policy 1: Unanimous agreement
                ans_gpt4 = gpt4_res.get("answer")
                ans_gemma = gemma_res.get("answer")
                ans_qwen = qwen_res.get("answer")

                # Ensure all answers are strings before comparing them (case-insensitively).
                if isinstance(ans_gpt4, str) and isinstance(ans_gemma, str) and isinstance(ans_qwen, str) and \
                   ans_gpt4.lower() == ans_gemma.lower() == ans_qwen.lower():
                    chosen_feature = ans_gpt4.lower()
                    policy_applied = "1"
                
                # Policy 2: GPT-4.1 logit probability >= 0.95
                if chosen_feature is None:
                    # gpt4_ans_str is ans_gpt4, already fetched.
                    gpt4_logits = gpt4_res.get("logits", {}) # Default to empty dict if logits somehow missing (though checked above)
                    confidence_met = False
                    if isinstance(ans_gpt4, str): # Check if gpt4's answer is a string
                        gpt4_ans_lower = ans_gpt4.lower()
                        # Check feature names against the answer, ensuring logit values are numbers (float or int).
                        logit1 = gpt4_logits.get("1")
                        logit2 = gpt4_logits.get("2")
                        logit3 = gpt4_logits.get("3")

                        if gpt4_ans_lower == feature1_name.lower() and isinstance(logit1, (int, float)) and logit1 >= 0.95:
                            confidence_met = True
                        elif gpt4_ans_lower == feature2_name.lower() and isinstance(logit2, (int, float)) and logit2 >= 0.95:
                            confidence_met = True
                        elif gpt4_ans_lower == "neither" and isinstance(logit3, (int, float)) and logit3 >= 0.95:
                            confidence_met = True

                    if confidence_met:
                        chosen_feature = gpt4_ans_lower # gpt4_ans_lower is already defined and lowercased
                        policy_applied = "2"

                # Policy 3: Highest probability feature among the three models' results
                if chosen_feature is None:
                    candidates = []
                    
                    for model_name, res_data in [("gpt-4.1", gpt4_res), ("gemma-3", gemma_res), ("qwen-3", qwen_res)]:
                        # Ensure logits exist and are dicts (already checked, but good for safety here too).
                        if isinstance(res_data, dict) and isinstance(res_data.get("logits"), dict):
                            logits = res_data["logits"]
                            # Ensure logit values are numbers (float or int) before using them, defaulting to 0.0 if not.
                            prob1 = logits.get("1") if isinstance(logits.get("1"), (int, float)) else 0.0
                            prob2 = logits.get("2") if isinstance(logits.get("2"), (int, float)) else 0.0
                            prob3 = logits.get("3") if isinstance(logits.get("3"), (int, float)) else 0.0
                            
                            candidates.append({"feature": feature1_name, "prob": prob1, "model": model_name})
                            candidates.append({"feature": feature2_name, "prob": prob2, "model": model_name})
                            candidates.append({"feature": "neither", "prob": prob3, "model": model_name})
                    
                    if not candidates: 
                        # This case should ideally not be reached if models have data and logits.
                        # print(f"Info: No candidates for policy 3 for word '{actual_word_key}', dim '{dim_key}' (lang: {lang_key}). Skipping this dimension.")
                        continue 
                        
                    best_candidate = max(candidates, key=lambda x: x["prob"])
                    chosen_feature = best_candidate["feature"].lower() # Standardize to lowercase
                    policy_applied = f"3: Highest probability ({best_candidate['prob']:.4f} from {best_candidate['model']})"
                
                # Add the dimension to the result only if a feature was chosen and it is not 'neither'.
                if chosen_feature is not None and policy_applied is not None and chosen_feature != 'neither':
                    new_dimensions_for_current_word[dim_key] = {
                        "answer": chosen_feature, # Already lowercased
                        "policy": policy_applied
                    }
                else:
                    # print(f"Info: No policy applied or answer was 'neither' for word '{actual_word_key}', dimension '{dim_key}' (lang: {lang_key}). Dimension will not be included.")
                    pass # Dimension not added if no feature chosen, policy not applied, or answer is 'neither'.

            # Update the dimensions of the word object in final_gt (which originated from combined_words_list)
            word_obj_in_final_gt["dimensions"] = new_dimensions_for_current_word
            
    return final_gt

if __name__ == '__main__':
    common_words_path = 'data/processed/nat/common_words.json' # Renamed for clarity
    rare_words_path = 'data/processed/nat/rare_words.json'   # Renamed for clarity

    gpt4_file = "data/processed/nat/semantic_dimension/semantic_dimension_gt_gpt-4.1_logits.json" 
    gemma_file = "data/processed/nat/semantic_dimension/semantic_dimension_gt_gemma-3-27b-it_logits.json"
    qwen_file = "data/processed/nat/semantic_dimension/semantic_dimension_gt_Qwen3-32B_logits.json"
    output_file = "data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json"

    try:
        print(f"Loading common words from {common_words_path}...")
        common_words_data = load_json(common_words_path)
        print(f"Loading rare words from {rare_words_path}...")
        rare_words_data = load_json(rare_words_path)
        
        print("Combining common and rare words into a single list...")
        # Add 'word_group' information before combining
        for common_word in common_words_data:
            common_word['word_group'] = 'common'
        for rare_word in rare_words_data:
            rare_word['word_group'] = 'rare'
        
        # Combine common and rare words into a single list
        combined_words = common_words_data + rare_words_data # This list is passed to determine_final_gt
        print(f"Total words loaded and combined: {len(combined_words)}")

        print(f"Loading GPT-4.1 results from {gpt4_file}...")
        gpt4_data = load_json(gpt4_file)
        print(f"Loading Gemma-3 results from {gemma_file}...")
        gemma_data = load_json(gemma_file)
        print(f"Loading Qwen-3 results from {qwen_file}...")
        qwen_data = load_json(qwen_file)
    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from an input file. {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        exit(1)

    print("Determining final ground truth based on combined_words list...")
    # Pass the combined_words list as the first argument
    final_results = determine_final_gt(combined_words, gpt4_data, gemma_data, qwen_data)

    try:
        save_json(final_results, output_file)
        print(f"Final ground truth successfully saved to {output_file}")
    except Exception as e:
        print(f"An error occurred while saving the output file: {e}")
        exit(1)
