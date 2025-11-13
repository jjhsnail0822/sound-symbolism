import json
import pandas as pd
from sklearn.metrics import f1_score
import os
import glob
from collections import defaultdict

def parse_file(file_path):
    """
    Parses a single Label Studio JSON file and extracts relevant data.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: A list of dictionaries, each containing parsed result data.
              Returns an empty list if the file cannot be processed.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading file {os.path.basename(file_path)}: {e}")
        return []

    results = []
    for item in data:
        try:
            # Skip if there are no annotations.
            if not item.get('annotations'):
                continue
            
            # Ground Truth
            ground_truth = int(item['data']['answer'])
            
            # Human-submitted answer
            human_answer = int(item['annotations'][0]['result'][0]['value']['choices'][0])
            
            # Metadata
            dimension = item['data']['meta_data']['dimension']
            
            results.append({
                'dimension': dimension,
                'ground_truth': ground_truth,
                'prediction': human_answer
            })
        except (KeyError, IndexError, TypeError, ValueError):
            # Skip data with unexpected structure.
            continue
            
    return results

def analyze_all_results(directory_path, output_json_path):
    """
    Analyzes all human evaluation JSON files in a directory, calculates the average
    F1 score for each dimension across all annotators, ranks them, and saves the result.

    Args:
        directory_path (str): Path to the directory containing the JSON files.
        output_json_path (str): Path to save the final ranked results JSON file.
    """
    all_json_files = glob.glob(os.path.join(directory_path, '*.json'))
    # Exclude the output file itself from the list of files to be analyzed.
    output_filename = os.path.basename(output_json_path)
    json_files = [f for f in all_json_files if os.path.basename(f) != output_filename]
    
    if not json_files:
        print(f"No JSON files found in '{directory_path}' to analyze.")
        return

    # Store F1 scores for each dimension from all files
    dimension_f1_scores = defaultdict(list)

    print(f"Found {len(json_files)} files to analyze...")

    for file_path in json_files:
        parsed_data = parse_file(file_path)
        if not parsed_data:
            print(f"Skipping empty or invalid file: {os.path.basename(file_path)}")
            continue

        df = pd.DataFrame(parsed_data)
        
        # Calculate Macro-F1 score for each dimension in the current file
        for dimension_name, group_df in df.groupby('dimension'):
            if group_df.empty:
                continue
            
            y_true = group_df['ground_truth']
            y_pred = group_df['prediction']
            
            if len(y_true) > 0:
                macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                dimension_f1_scores[dimension_name].append(macro_f1)

    if not dimension_f1_scores:
        print("No valid data found across all files to analyze.")
        return

    # Calculate the average F1 score for each dimension
    avg_dimension_scores = {
        dim: sum(scores) / len(scores) for dim, scores in dimension_f1_scores.items()
    }

    # Calculate standard deviation for each dimension
    std_dimension_scores = {
        dim: pd.Series(scores).std() if len(scores) > 1 else 0 for dim, scores in dimension_f1_scores.items()
    }

    # Sort dimensions by average F1 score in descending order
    ranked_dimensions = sorted(avg_dimension_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Prepare final results for printing and saving
    final_results = {
        'ranked_dimensions': [
            {'dimension': dim, 'average_f1_score': score, 'std_dev': std_dimension_scores[dim]} for dim, score in ranked_dimensions
        ]
    }

    # Print the ranked results to the console
    print("\n--- Average F1 Score by Dimension (Ranked) ---")
    for result in final_results['ranked_dimensions']:
        print(f"  - Dimension: {result['dimension']:<15} | Average F1: {result['average_f1_score']:.4f} | Std Dev: {result['std_dev']:.4f}")

    # Save the results to a JSON file
    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nResults successfully saved to {output_json_path}")
    except IOError as e:
        print(f"\nError saving results to file: {e}")


if __name__ == '__main__':
    # Directory containing the JSON results from different annotators
    json_dir_path = 'results/experiments/semantic_dimension/human_eval/'
    
    # Path for the output file with the final analysis
    output_file_path = 'results/experiments/semantic_dimension/human_eval/ranked_dimension_f1_scores.json'

    analyze_all_results(json_dir_path, output_file_path)