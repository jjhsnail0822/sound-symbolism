import json
import os
import re
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_statistics_for_file(input_file_path: str):
    """
    Calculates classification metrics for a single experiment result file.

    Args:
        input_file_path (str): The path to the input JSON file.

    Returns:
        dict: A dictionary containing the calculated statistics for the file.
              Returns None if the file is empty or cannot be processed.
    """
    with open(input_file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {input_file_path}. Skipping.")
            return None

    results_list = data.get("results") if isinstance(data, dict) else data
    if not results_list:
        print(f"Warning: No results found in {input_file_path}. Skipping.")
        return None

    results_by_dimension = defaultdict(lambda: {"y_true": [], "y_pred": []})

    for result in results_list:
        if not result or "meta_data" not in result:
            continue

        try:
            dimension = result["meta_data"]["dimension"]
            y_true = int(result["meta_data"]["answer"])
            y_pred = int(result["model_answer"])
            
            if y_pred == 0:
                y_pred = 2 if y_true == 1 else 1
            
            if y_pred in [1, 2]:
                results_by_dimension[dimension]["y_true"].append(y_true)
                results_by_dimension[dimension]["y_pred"].append(y_pred)
        except (ValueError, TypeError, KeyError):
            continue

    statistics = {}
    all_y_true = []
    all_y_pred = []
    for dimension, values in results_by_dimension.items():
        y_true = values["y_true"]
        y_pred = values["y_pred"]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1, zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)

        statistics[dimension] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "count": len(y_true)
        }

    overall_accuracy = accuracy_score(all_y_true, all_y_pred) if all_y_true else 0

    return {
        "overall_accuracy": overall_accuracy,
        "dimensions": statistics
    }

def aggregate_statistics(root_dir: str, output_file: str):
    """
    Aggregates statistics from all relevant JSON files into a single file.
    """
    aggregated_results = defaultdict(lambda: defaultdict(dict))
    
    # Regex to parse the filename
    # semantic_dimension_binary_{input_type}-{word_type}_{model_name}.json
    pattern = re.compile(r"semantic_dimension_binary_(?P<input_type>.+?)-(?P<word_type>.+?)_(?P<model_name>.+)\.json")

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("semantic_dimension") and filename.endswith(".json"):
                match = pattern.match(filename)
                if not match:
                    print(f"Warning: Filename format mismatch, skipping: {filename}")
                    continue
                
                parts = match.groupdict()
                model_name = parts["model_name"]
                input_type = parts["input_type"]
                word_type = parts["word_type"]
                
                file_path = os.path.join(dirpath, filename)
                print(f"Processing {file_path}...")
                
                stats = calculate_statistics_for_file(file_path)
                
                if stats:
                    aggregated_results[model_name][word_type][input_type] = stats

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the aggregated results to the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated_results, f, indent=4)

    print(f"\nAggregation complete. Statistics saved to {output_file}")

if __name__ == '__main__':
    ROOT_DIRECTORY = "results/experiments/semantic_dimension/binary"
    OUTPUT_FILE = "results/statistics/semdim_stat.json"
    
    if os.path.exists(ROOT_DIRECTORY):
        aggregate_statistics(ROOT_DIRECTORY, OUTPUT_FILE)
    else:
        print(f"Error: Root directory not found at {ROOT_DIRECTORY}")
        print("Please update the ROOT_DIRECTORY path in the script.")