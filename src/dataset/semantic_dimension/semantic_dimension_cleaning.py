import json
import copy

# Policy
# 1. Unanimous agreement across all three models (GPT-4.1, Gemma-3, Qwen-3)
# 2. GPT-4.1 answer with logit probability >= 0.95
# 3. Highest probability feature among the three models' results

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

def determine_final_gt(gpt4_data, gemma_data, qwen_data):
    final_gt = copy.deepcopy(gpt4_data)

    # Iterate through language keys present in the copied data (final_gt)
    for lang_key, lang_list_in_final_gt in final_gt.items():
        # Ensure gpt4_data (original) also has this lang_key for fetching original dimensions
        gpt4_lang_list_original = gpt4_data.get(lang_key)
        gemma_lang_list = gemma_data.get(lang_key)
        qwen_lang_list = qwen_data.get(lang_key)

        # Verify that the data for the current language key is a list in all models
        if not (isinstance(lang_list_in_final_gt, list) and \
                isinstance(gpt4_lang_list_original, list) and \
                isinstance(gemma_lang_list, list) and \
                isinstance(qwen_lang_list, list)):
            print(f"Warning: Data for language key '{lang_key}' is not consistently a list across all models or is missing. Skipping '{lang_key}'.")
            continue
        
        # Iterate through word objects in the final_gt's language list
        for word_obj_in_final_gt in lang_list_in_final_gt:
            if not isinstance(word_obj_in_final_gt, dict) or "word" not in word_obj_in_final_gt:
                print(f"Warning: Invalid word object format or missing 'word' key in final_gt for lang '{lang_key}'. Skipping item: {str(word_obj_in_final_gt)[:100]}")
                continue
            
            actual_word_key = word_obj_in_final_gt["word"]

            # Find the corresponding word object in the original gpt4, Gemma, and Qwen lists
            gpt4_word_obj_original = find_word_in_list(gpt4_lang_list_original, actual_word_key)
            gemma_word_obj = find_word_in_list(gemma_lang_list, actual_word_key)
            qwen_word_obj = find_word_in_list(qwen_lang_list, actual_word_key)

            if not (gpt4_word_obj_original and gemma_word_obj and qwen_word_obj):
                print(f"Warning: Word '{actual_word_key}' (lang: {lang_key}) not found in all original models' lists. Skipping word, dimensions will be empty if they existed.")
                if "dimensions" in word_obj_in_final_gt: # Ensure dimensions key exists before clearing
                    word_obj_in_final_gt["dimensions"] = {} # Clear dimensions in final_gt for this word
                continue

            # Ensure each word object has a 'dimensions' dictionary in original data
            if not all(isinstance(word_obj, dict) and 
                       "dimensions" in word_obj and 
                       isinstance(word_obj["dimensions"], dict)
                       for word_obj in [gpt4_word_obj_original, gemma_word_obj, qwen_word_obj]):
                print(f"Warning: 'dimensions' key missing, not a dict, or word data malformed for '{actual_word_key}' (lang: {lang_key}) in original models. Skipping word, dimensions will be empty.")
                if "dimensions" in word_obj_in_final_gt:
                    word_obj_in_final_gt["dimensions"] = {}
                continue

            new_dimensions_for_current_word = {} # Store new dimension results for this word
            
            dimensions_gpt4 = gpt4_word_obj_original["dimensions"]
            dimensions_gemma = gemma_word_obj["dimensions"]
            dimensions_qwen = qwen_word_obj["dimensions"]

            # Iterate through dimension keys from the original GPT-4's perspective
            for dim_key in dimensions_gpt4: 
                if not isinstance(dim_key, str) or '-' not in dim_key:
                    # print(f"Info: Skipping dimension key '{dim_key}' for word '{actual_word_key}' (lang: {lang_key}) as it does not seem to be a 'feature1-feature2' pair.")
                    continue

                if dim_key not in dimensions_gemma or dim_key not in dimensions_qwen:
                    # print(f"Warning: Dimension '{dim_key}' for word '{actual_word_key}' (lang: {lang_key}) not present consistently in all models. Skipping dimension.")
                    continue
                
                try:
                    feature1_name, feature2_name = dim_key.split('-', 1) 
                except ValueError:
                    # print(f"Warning: Could not parse dimension key '{dim_key}' for word '{actual_word_key}' (lang: {lang_key}). Skipping dimension.")
                    continue

                gpt4_res = dimensions_gpt4[dim_key]
                gemma_res = dimensions_gemma[dim_key]
                qwen_res = dimensions_qwen[dim_key]

                if not all(
                    isinstance(res, dict) and
                    "logits" in res and isinstance(res.get("logits"), dict) and
                    all(k in res["logits"] for k in ["1", "2", "3"]) and 
                    "answer" in res
                    for res in [gpt4_res, gemma_res, qwen_res]
                ):
                    # print(f"Warning: Logits/answer structure malformed for word '{actual_word_key}', dimension '{dim_key}' (lang: {lang_key}). Skipping.")
                    continue

                chosen_feature = None
                policy_applied = None

                # # Policy 1: Majority vote (at least 2 out of 3 models agree)
                # answers = [
                #     gpt4_res["answer"].lower() if isinstance(gpt4_res.get("answer"), str) else None,
                #     gemma_res["answer"].lower() if isinstance(gemma_res.get("answer"), str) else None,
                #     qwen_res["answer"].lower() if isinstance(qwen_res.get("answer"), str) else None
                # ]
                # answers = [ans for ans in answers if ans is not None] # Filter out None if answer was not a string

                # if len(answers) == 3: # Proceed only if all models provided a string answer
                #     answer_counts = {}
                #     for ans in answers:
                #         answer_counts[ans] = answer_counts.get(ans, 0) + 1
                    
                #     for ans, count in answer_counts.items():
                #         if count >= 2:
                #             chosen_feature = ans
                #             policy_applied = 1
                #             break
                
                # Policy 1: Unanimous agreement
                if gpt4_res["answer"] == gemma_res["answer"] == qwen_res["answer"]:
                    chosen_feature = gpt4_res["answer"].lower()
                    policy_applied = "1"

                # Policy 2: GPT-4.1 logit probability >= 0.95
                if chosen_feature is None:
                    gpt4_ans_str = gpt4_res["answer"]
                    gpt4_logits = gpt4_res["logits"] 
                    confidence_met = False
                    if isinstance(gpt4_ans_str, str):
                        gpt4_ans_lower = gpt4_ans_str.lower()
                        if gpt4_ans_lower == feature1_name.lower() and gpt4_logits.get("1", 0.0) >= 0.95:
                            confidence_met = True
                        elif gpt4_ans_lower == feature2_name.lower() and gpt4_logits.get("2", 0.0) >= 0.95:
                            confidence_met = True
                        elif gpt4_ans_lower == "neither" and gpt4_logits.get("3", 0.0) >= 0.95:
                            confidence_met = True

                    if confidence_met:
                        chosen_feature = gpt4_ans_lower
                        policy_applied = "2"

                # Policy 3: Highest probability feature among the three models' results
                if chosen_feature is None:
                    candidates = []
                    
                    for model_name, res_data in [("gpt-4.1", gpt4_res), ("gemma-3", gemma_res), ("qwen-3", qwen_res)]:
                        # Ensure logits exist and are dicts before trying to access them
                        if isinstance(res_data, dict) and isinstance(res_data.get("logits"), dict):
                            logits = res_data["logits"]
                            candidates.append({"feature": feature1_name, "prob": logits.get("1",0.0), "model": model_name})
                            candidates.append({"feature": feature2_name, "prob": logits.get("2",0.0), "model": model_name})
                            candidates.append({"feature": "neither", "prob": logits.get("3",0.0), "model": model_name})
                    
                    if not candidates: 
                        # print(f"Error: No candidates for policy 3 for word '{actual_word_key}', dim '{dim_key}' (lang: {lang_key}). Skipping.")
                        continue 
                        
                    best_candidate = max(candidates, key=lambda x: x["prob"])
                    chosen_feature = best_candidate["feature"].lower() # Standardize to lowercase
                    policy_applied = f"3: Highest probability ({best_candidate['prob']:.4f} from {best_candidate['model']})"
                
                if chosen_feature is not None and policy_applied is not None:
                    new_dimensions_for_current_word[dim_key] = {
                        "answer": chosen_feature, # Already lowercased
                        "policy": policy_applied
                    }
                else:
                    # print(f"Info: No policy applied for word '{actual_word_key}', dimension '{dim_key}' (lang: {lang_key}). Dimension will not be included.")
                    pass # Dimension not added if no feature chosen

            # Update the dimensions of the word object in final_gt
            word_obj_in_final_gt["dimensions"] = new_dimensions_for_current_word
            
    return final_gt

if __name__ == '__main__':
    gpt4_file = "data/processed/nat/semantic_dimension/semantic_dimension_gt_gpt-4.1_logits.json" 
    gemma_file = "data/processed/nat/semantic_dimension/semantic_dimension_gt_gemma-3-27b-it_logits.json"
    qwen_file = "data/processed/nat/semantic_dimension/semantic_dimension_gt_Qwen3-32B_logits.json"
    output_file = "data/processed/nat/semantic_dimension/semantic_dimension_gt.json"

    try:
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

    print("Determining final ground truth...")
    final_results = determine_final_gt(gpt4_data, gemma_data, qwen_data)

    try:
        save_json(final_results, output_file)
        print(f"Final ground truth successfully saved to {output_file}")
    except Exception as e:
        print(f"An error occurred while saving the output file: {e}")
        exit(1)
