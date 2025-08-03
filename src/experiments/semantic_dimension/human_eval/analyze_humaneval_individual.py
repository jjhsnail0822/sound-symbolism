import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def analyze_results(file_path):
    """
    Analyzes human evaluation results to calculate accuracy and macro-F1 score.

    Args:
        file_path (str): Path to the JSON file exported from Label Studio.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format - {file_path}")
        return

    results = []
    for item in data:
        try:
            # Ground Truth
            ground_truth = int(item['data']['answer'])
            
            # Skip if there are no annotations.
            if not item.get('annotations'):
                continue
            
            # Human-submitted answer
            human_answer = int(item['annotations'][0]['result'][0]['value']['choices'][0])
            
            # Metadata
            meta_data = item['data']['meta_data']
            dimension = meta_data['dimension']
            language = meta_data['language']
            
            # Classify word group
            word_group = 'constructed' if language == 'art' else 'natural'
        
            results.append({
                'dimension': dimension,
                'word_group': word_group,
                'ground_truth': ground_truth,
                'prediction': human_answer
            })
        except (KeyError, IndexError, TypeError, ValueError) as e:
            # Skip data with unexpected structure.
            # print(f"Error processing data (ID: {item.get('id')}): {e} - Skipping.")
            continue

    if not results:
        print("No valid data to analyze.")
        return

    df = pd.DataFrame(results)

    def calculate_scores(dataframe, group_name):
        if dataframe.empty:
            return
        
        y_true = dataframe['ground_truth']
        y_pred = dataframe['prediction']
        
        # Check if there is enough data
        if len(y_true) == 0:
            print(f"--- {group_name} Results ---")
            print("  - No data to calculate.\n")
            return

        accuracy = accuracy_score(y_true, y_pred)
        # Macro F1 score calculates F1 for each label and then finds their unweighted mean.
        # zero_division=0 handles cases where there are no predictions for a class by returning 0 instead of a warning.
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        print(f"--- {group_name} Results (Total: {len(y_true)}) ---")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Macro-F1 Score: {macro_f1:.4f}\n")

    # 1. Overall Results
    calculate_scores(df, "Overall")

    # 2. Results by Dimension
    print("--- Results by Dimension ---")
    for dimension_name, group_df in df.groupby('dimension'):
        calculate_scores(group_df, f"Dimension: {dimension_name}")

    # 3. Results by Word Group (Natural vs. Constructed)
    print("--- Results by Word Group ---")
    for word_group_name, group_df in df.groupby('word_group'):
        calculate_scores(group_df, f"Word Group: {word_group_name}")


if __name__ == '__main__':
    json_file_path = 'results/experiments/semantic_dimension/human_eval/annotator_9.json'

    analyze_results(json_file_path)