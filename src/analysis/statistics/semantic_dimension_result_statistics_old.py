import json
import os
import re
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_statistics_for_file(input_file_path: str):
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

    results_by_lang_dim = defaultdict(lambda: defaultdict(lambda: {"y_true": [], "y_pred": []}))

    for result in results_list:
        if not result or "meta_data" not in result:
            continue

        # Use .get() for safe access to language key
        language = result["meta_data"].get("language")
        dimension = result["meta_data"]["dimension"]
        y_true = int(result["meta_data"]["answer"])
        y_pred = int(result["model_answer"])
        
        if y_pred == 0:
            y_pred = 2 if y_true == 1 else 1
        
        if y_pred in [1, 2]:
            results_by_lang_dim[language][dimension]["y_true"].append(y_true)
            results_by_lang_dim[language][dimension]["y_pred"].append(y_pred)
        else:
            raise ValueError(f"Warning: Invalid prediction {y_pred} for language {language}, dimension {dimension}.")

    lang_statistics = {}
    for lang, dimensions_data in results_by_lang_dim.items():
        statistics = {}
        all_y_true = []
        all_y_pred = []
        for dimension, values in dimensions_data.items():
            y_true = values["y_true"]
            y_pred = values["y_pred"]
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', labels=[1, 2], zero_division=0
            )
            accuracy = accuracy_score(y_true, y_pred)

            statistics[dimension] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "macro_f1_score": f1,
                "count": len(y_true),
                "y_true": y_true, # Temporarily store raw values
                "y_pred": y_pred  # Temporarily store raw values
            }

        overall_accuracy = accuracy_score(all_y_true, all_y_pred) if all_y_true else 0

        lang_statistics[lang] = {
            "overall_accuracy": overall_accuracy,
            "dimensions": statistics
        }

    return lang_statistics

def aggregate_statistics(root_dir: str, output_file: str):
    # Structure: [language][model_name][word_type][input_type]
    aggregated_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

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
                
                lang_stats = calculate_statistics_for_file(file_path)
                
                if lang_stats:
                    for lang, stats in lang_stats.items():
                        aggregated_results[lang][model_name][word_type][input_type] = stats

    # Combine 'common' and 'rare' to create 'natural' word group for each language
    for lang in aggregated_results:
        for model_name, word_types in aggregated_results[lang].items():
            # Check if both 'common' and 'rare' exist for this model
            if "common" in word_types and "rare" in word_types:
                for input_type, common_stats in word_types.get("common", {}).items():
                    if input_type in word_types.get("rare", {}):
                        rare_stats = word_types["rare"][input_type]
                        
                        natural_stats = {"dimensions": {}}
                        all_y_true_natural = []
                        all_y_pred_natural = []

                        all_dims = set(common_stats["dimensions"].keys()) | set(rare_stats["dimensions"].keys())

                        for dim in all_dims:
                            y_true = common_stats["dimensions"].get(dim, {}).get("y_true", []) + rare_stats["dimensions"].get(dim, {}).get("y_true", [])
                            y_pred = common_stats["dimensions"].get(dim, {}).get("y_pred", []) + rare_stats["dimensions"].get(dim, {}).get("y_pred", [])
                            
                            if not y_true:
                                continue

                            all_y_true_natural.extend(y_true)
                            all_y_pred_natural.extend(y_pred)

                            precision, recall, f1, _ = precision_recall_fscore_support(
                                y_true, y_pred, average='macro', labels=[1, 2], zero_division=0
                            )
                            accuracy = accuracy_score(y_true, y_pred)

                            natural_stats["dimensions"][dim] = {
                                "accuracy": accuracy,
                                "precision": precision,
                                "recall": recall,
                                "macro_f1_score": f1,
                                "count": len(y_true)
                            }
                        
                        if all_y_true_natural:
                            natural_stats["overall_accuracy"] = accuracy_score(all_y_true_natural, all_y_pred_natural)
                            if "natural" not in aggregated_results[lang][model_name]:
                                aggregated_results[lang][model_name]["natural"] = {}
                            aggregated_results[lang][model_name]["natural"][input_type] = natural_stats

    # Clean up y_true/y_pred from the final output
    for lang in aggregated_results:
        for model_name in aggregated_results[lang]:
            for word_type in aggregated_results[lang][model_name]:
                for input_type in aggregated_results[lang][model_name][word_type]:
                    if "dimensions" in aggregated_results[lang][model_name][word_type][input_type]:
                        for dim in aggregated_results[lang][model_name][word_type][input_type]["dimensions"]:
                            aggregated_results[lang][model_name][word_type][input_type]["dimensions"][dim].pop("y_true", None)
                            aggregated_results[lang][model_name][word_type][input_type]["dimensions"][dim].pop("y_pred", None)

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