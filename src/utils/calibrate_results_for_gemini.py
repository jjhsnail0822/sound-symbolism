import os
import json
from glob import glob

RESULT_ROOT = "results/experiments/semantic_dimension/binary"
PROMPT_ROOT = "data/prompts/semantic_dimension"


def load_prompt_answers(prompt_path):
    """Load prompt file and return mapping from (lang, word, dim) to answer."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mapping = {}
    for item in data:
        meta = item.get("meta_data", {})
        key = (meta.get("language"), meta.get("word"), meta.get("dimension"))
        mapping[key] = meta.get("answer")
    return mapping


def process_result_file(result_path, prompt_map):
    with open(result_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    new_results = []
    correct_count = 0

    for entry in result_data.get("results", []):
        meta = entry.get("meta_data", {})
        key = (meta.get("language"), meta.get("word"), meta.get("dimension"))
        if key not in prompt_map:
            # remove entries not present in prompt
            continue
        meta["answer"] = prompt_map[key]
        entry["meta_data"] = meta
        entry["is_correct"] = str(entry.get("model_answer")).strip() == str(prompt_map[key])
        if entry["is_correct"]:
            correct_count += 1
        new_results.append(entry)

    result_data["results"] = new_results
    result_data["total_count"] = len(new_results)
    result_data["correct_count"] = correct_count
    result_data["accuracy"] = correct_count / len(new_results) if new_results else 0.0

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"Updated {result_path}")


def main():
    prompt_files = glob(os.path.join(PROMPT_ROOT, '*.json'))
    prompt_bases = {os.path.basename(p).replace('.json', ''): p for p in prompt_files}

    # Process individual result files first
    for sub in os.listdir(RESULT_ROOT):
        subdir = os.path.join(RESULT_ROOT, sub)
        if not os.path.isdir(subdir):
            continue
        
        result_files = glob(os.path.join(subdir, 'semantic_dimension*.json'))
        for result_file in result_files:
            if os.path.basename(result_file).startswith('all_results'):
                continue

            result_base = os.path.basename(result_file).replace('.json', '')
            
            found_prompt = False
            for p_base, p_path in prompt_bases.items():
                if result_base.startswith(p_base):
                    prompt_path = p_path
                    if 'old_no_gemini' in prompt_path:
                        continue
                    
                    prompt_map = load_prompt_answers(prompt_path)
                    process_result_file(result_file, prompt_map)
                    found_prompt = True
                    break
            
            if not found_prompt:
                print(f"Could not find a matching prompt for {result_file}, skipping.")

    # Update all_results summary files
    for sub in os.listdir(RESULT_ROOT):
        subdir = os.path.join(RESULT_ROOT, sub)
        if not os.path.isdir(subdir):
            continue

        # Find and update all summary files
        for all_results_file in glob(os.path.join(subdir, 'all_results*.json')):
            try:
                with open(all_results_file, 'r', encoding='utf-8') as f:
                    all_results_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Could not read or decode {all_results_file}, skipping.")
                continue

            # Extract model name from the all_results filename itself
            all_results_filename = os.path.basename(all_results_file)
            try:
                # Assumes filename is "all_results_{model_name}.json"
                model_filename_part = all_results_filename.replace('all_results_', '').replace('.json', '')
            except IndexError:
                print(f"Could not extract model name from {all_results_filename}, skipping.")
                continue

            for item in all_results_data:
                data_path = item.get("data_path")
                model = item.get("model") # Get model name from inside the json
                if not data_path or not model:
                    continue

                prompt_filename = os.path.basename(data_path)
                prompt_base = prompt_filename.rsplit('.json', 1)[0]
                
                # Construct filename from model field, removing any slashes
                model_filename_part = model.split('/')[-1]
                individual_result_filename = f"{prompt_base}_{model_filename_part}.json"
                individual_result_path = os.path.join(subdir, individual_result_filename)

                if os.path.exists(individual_result_path):
                    try:
                        with open(individual_result_path, 'r', encoding='utf-8') as f_ind:
                            individual_data = json.load(f_ind)
                        
                        correct_count = individual_data.get("correct_count")
                        total_count = individual_data.get("total_count")

                        if correct_count is not None and total_count is not None:
                            item["correct_count"] = correct_count
                            item["total_count"] = total_count
                            item["accuracy"] = correct_count / total_count if total_count > 0 else 0.0
                    except (json.JSONDecodeError, FileNotFoundError):
                         print(f"Could not update from {individual_result_path}, skipping.")
                         continue
            
            with open(all_results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results_data, f, ensure_ascii=False, indent=4)
            
            print(f"Updated summary file: {all_results_file}")


if __name__ == "__main__":
    main()