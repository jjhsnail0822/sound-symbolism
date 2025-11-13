import json
import os
import re
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

ROOT_DIRECTORY = "results/experiments/semantic_dimension/binary"
OUTPUT_FILE = "results/statistics/semdim_stat.json"

# Regex to parse the filename
# semantic_dimension_binary_{input_type}-{word_group}_{model_name}.json
pattern = re.compile(r"semantic_dimension_binary_(?P<input_type>.+?)-(?P<word_group>.+?)_(?P<model_name>.+)\.json")

# raw_data = {"model_name": {"word_group": {"input_type": {"dimension": {"language": {"y_true": [], "y_pred": []}}}}}}
raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"y_true": [], "y_pred": []})))))
for dirpath, _, filenames in os.walk(ROOT_DIRECTORY):
    for filename in filenames:
        if filename.startswith("semantic_dimension") and filename.endswith(".json"):
            match = pattern.match(filename)
            if match:
                model_name = match.group('model_name')
                word_group = match.group('word_group')
                input_type = match.group('input_type')

                file_path = os.path.join(dirpath, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for data_point in data['results']:
                    if word_group == "common" or word_group == "rare":
                        word_group = "natural"
                    current_dimension = data_point['meta_data']['dimension']
                    current_language = data_point['meta_data']['language']
                    current_y_true = int(data_point['meta_data']['answer'])
                    current_y_pred = int(data_point['model_answer'])
                    if current_y_pred not in [1, 2]:
                        current_y_pred = 1 if current_y_true == 2 else 2

                    raw_data[model_name][word_group][input_type][current_dimension][current_language]["y_true"].append(current_y_true)
                    raw_data[model_name][word_group][input_type][current_dimension][current_language]["y_pred"].append(current_y_pred)

# Calculate macro-F1 scores and other statistics
# statistics = {"input_type": {"word_group": {"model_name": {"dimension": {"language": {"accuracy": float, "precision": float, "recall": float, "macro_f1_score": float, "count": int}}}}}}
statistics = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))

for model_name, word_groups in raw_data.items():
    for word_group, input_types in word_groups.items():
        for input_type, dimensions in input_types.items():
            for dimension, languages in dimensions.items():
                if model_name.startswith("gpt-4o"):
                    model_name = "gpt-4o"

                for language, values in languages.items():
                    y_true = values["y_true"]
                    y_pred = values["y_pred"]

                    if not y_true or not y_pred:
                        continue

                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='macro', labels=[1, 2], zero_division=0
                    )
                    accuracy = accuracy_score(y_true, y_pred)

                    statistics[model_name][word_group][input_type][dimension][language] = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "macro_f1_score": f1,
                        "count": len(y_true)
                    }

                statistics[model_name][word_group][input_type][dimension]["all"] = {
                    "accuracy": np.mean([statistics[model_name][word_group][input_type][dimension][lang]["accuracy"] for lang in languages]),
                    "precision": np.mean([statistics[model_name][word_group][input_type][dimension][lang]["precision"] for lang in languages]),
                    "recall": np.mean([statistics[model_name][word_group][input_type][dimension][lang]["recall"] for lang in languages]),
                    "macro_f1_score": np.mean([statistics[model_name][word_group][input_type][dimension][lang]["macro_f1_score"] for lang in languages]),
                    "count": sum([statistics[model_name][word_group][input_type][dimension][lang]["count"] for lang in languages])
                }

                # also calculate scores from mere sum of y_true and y_pred

                all_y_true = []
                all_y_pred = []
                for lang in languages:
                    for y in languages[lang]["y_true"]:
                        all_y_true.append(y)
                    for y in languages[lang]["y_pred"]:
                        all_y_pred.append(y)
                    if len(all_y_true) != len(all_y_pred):
                        raise ValueError(f"Mismatch in counts for {model_name}, {word_group}, {input_type}, {dimension}, {lang}")
                if all_y_true and all_y_pred:
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        all_y_true, all_y_pred, average='macro', labels=[1, 2], zero_division=0
                    )
                    accuracy = accuracy_score(all_y_true, all_y_pred)

                    statistics[model_name][word_group][input_type][dimension]["all_sum"] = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "macro_f1_score": f1,
                        "count": len(all_y_true)
                    }

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(statistics, f, indent=4, ensure_ascii=False)

print(f"Statistics saved to {OUTPUT_FILE}")