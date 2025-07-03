import json

def load_json(filepath):
    """Loads JSON data from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_word_in_list(word_list, word_key):
    """Helper function to find a word dictionary by its 'word' key in a list of dictionaries."""
    if not isinstance(word_list, list):
        return None
    for item in word_list:
        if isinstance(item, dict) and item.get("word") == word_key:
            return item
    return None

def analyze_policy_coverage(gpt4_data, gemma_data, qwen_data):
    """
    Analyzes the coverage of each policy when applied independently.
    A success is counted if it produces a valid, non-'neither' answer.
    """
    total_dimensions = 0
    policy1_successes = 0  # Unanimous agreement
    policy2_successes = 0  # GPT-4.1 confidence >= 0.95
    policy3_successes = 0  # Highest probability

    # Iterate through languages based on GPT-4 data
    for lang_key, gpt4_lang_list in gpt4_data.items():
        gemma_lang_list = gemma_data.get(lang_key, [])
        qwen_lang_list = qwen_data.get(lang_key, [])

        for gpt4_word_obj in gpt4_lang_list:
            word_key = gpt4_word_obj.get("word")
            if not word_key:
                continue

            gemma_word_obj = find_word_in_list(gemma_lang_list, word_key)
            qwen_word_obj = find_word_in_list(qwen_lang_list, word_key)

            # Check if the word exists in all three models and has valid 'dimensions'
            if not (gemma_word_obj and qwen_word_obj and
                    isinstance(gpt4_word_obj.get("dimensions"), dict) and
                    isinstance(gemma_word_obj.get("dimensions"), dict) and
                    isinstance(qwen_word_obj.get("dimensions"), dict)):
                continue

            dimensions_gpt4 = gpt4_word_obj["dimensions"]
            dimensions_gemma = gemma_word_obj["dimensions"]
            dimensions_qwen = qwen_word_obj["dimensions"]

            # Iterate through dimensions
            for dim_key in dimensions_gpt4:
                if dim_key not in dimensions_gemma or dim_key not in dimensions_qwen:
                    continue

                try:
                    feature1_name, feature2_name = dim_key.split('-', 1)
                except ValueError:
                    continue

                gpt4_res = dimensions_gpt4[dim_key]
                gemma_res = dimensions_gemma[dim_key]
                qwen_res = dimensions_qwen[dim_key]

                # Validate data structure
                if not all(isinstance(res, dict) and "logits" in res and "answer" in res for res in [gpt4_res, gemma_res, qwen_res]):
                    continue
                
                total_dimensions += 1

                # --- Policy 1: Unanimous agreement ---
                ans_gpt4 = gpt4_res.get("answer")
                ans_gemma = gemma_res.get("answer")
                ans_qwen = qwen_res.get("answer")
                if isinstance(ans_gpt4, str) and ans_gpt4.lower() == ans_gemma.lower() == ans_qwen.lower():
                    if ans_gpt4.lower() != 'neither':
                        policy1_successes += 1

                # --- Policy 2: GPT-4.1 confidence >= 0.95 ---
                if isinstance(ans_gpt4, str):
                    ans_lower = ans_gpt4.lower()
                    logits = gpt4_res.get("logits", {})
                    logit1 = logits.get("1")
                    logit2 = logits.get("2")
                    
                    confidence_met = False
                    if ans_lower == feature1_name.lower() and isinstance(logit1, (int, float)) and logit1 >= 0.95:
                        confidence_met = True
                    elif ans_lower == feature2_name.lower() and isinstance(logit2, (int, float)) and logit2 >= 0.95:
                        confidence_met = True
                    
                    if confidence_met: # 'neither' is already excluded by the condition above
                        policy2_successes += 1

                # --- Policy 3: Highest probability ---
                candidates = []
                for model_res, f1, f2 in [(gpt4_res, feature1_name, feature2_name), (gemma_res, feature1_name, feature2_name), (qwen_res, feature1_name, feature2_name)]:
                    logits = model_res.get("logits", {})
                    prob1 = logits.get("1") if isinstance(logits.get("1"), (int, float)) else 0.0
                    prob2 = logits.get("2") if isinstance(logits.get("2"), (int, float)) else 0.0
                    prob3 = logits.get("3") if isinstance(logits.get("3"), (int, float)) else 0.0
                    candidates.append({"feature": f1, "prob": prob1})
                    candidates.append({"feature": f2, "prob": prob2})
                    candidates.append({"feature": "neither", "prob": prob3})
                
                if candidates:
                    best_candidate = max(candidates, key=lambda x: x["prob"])
                    if best_candidate["feature"] != "neither":
                        policy3_successes += 1

    return total_dimensions, policy1_successes, policy2_successes, policy3_successes

if __name__ == '__main__':
    gpt4_file = "data/processed/nat/semantic_dimension/semantic_dimension_gt_gpt-4.1_logits.json"
    gemma_file = "data/processed/nat/semantic_dimension/semantic_dimension_gt_gemma-3-27b-it_logits.json"
    qwen_file = "data/processed/nat/semantic_dimension/semantic_dimension_gt_Qwen3-32B_logits.json"

    try:
        print("Loading model results...")
        gpt4_data = load_json(gpt4_file)
        gemma_data = load_json(gemma_file)
        qwen_data = load_json(qwen_file)
    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading files: {e}")
        exit(1)

    print("Analyzing policy coverage...")
    total, p1, p2, p3 = analyze_policy_coverage(gpt4_data, gemma_data, qwen_data)

    print("\n--- Policy Coverage Analysis Results ---")
    if total == 0:
        print("No valid dimension data to analyze.")
    else:
        p1_perc = (p1 / total) * 100
        p2_perc = (p2 / total) * 100
        p3_perc = (p3 / total) * 100

        print(f"Total dimensions analyzed: {total}")
        print("-" * 30)
        print(f"Policy 1 (Unanimous): {p1} pairs retained ({p1_perc:.2f}%)")
        print(f"Policy 2 (GPT-4.1 ≥ 0.95): {p2} pairs retained ({p2_perc:.2f}%)")
        print(f"Policy 3 (Highest Probability): {p3} pairs retained ({p3_perc:.2f}%)")
        print("-" * 30)