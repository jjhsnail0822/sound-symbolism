import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_average_cosine_distances(json_file_path):
    """
    Loads cosine distance data from a JSON file, calculates the average distance
    for each input type pair across all layers, and plots the results.

    Args:
        json_file_path (str): The path to the input JSON file.
    """
    # Load the JSON data from the file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # A nested dictionary to store distances: {input_pair: {layer_num: [distances]}}
    distances_by_pair_layer = defaultdict(lambda: defaultdict(list))

    # Iterate over each item (e.g., "art_lee-lay_abrupt-continuous") in the data
    for item_key, item_value in data.items():
        # Check if 'distances_between_input_types' exists for the item
        if 'distances_between_input_types' in item_value:
            distances_data = item_value['distances_between_input_types']
            # Iterate over each layer (e.g., "layer_0", "layer_1")
            for layer_key, layer_value in distances_data.items():
                try:
                    # Extract the integer part of the layer key (e.g., 0 from "layer_0")
                    layer_num = int(layer_key.split('_')[1])
                    # Iterate over each input type pair (e.g., "original_vs_ipa")
                    for pair_key, pair_value in layer_value.items():
                        if 'cosine_distance' in pair_value:
                            distance = pair_value['cosine_distance']
                            distances_by_pair_layer[pair_key][layer_num].append(distance)
                except (IndexError, ValueError) as e:
                    print(f"Could not parse layer key: {layer_key}. Error: {e}")
                    continue

    # Calculate the average distance for each pair at each layer
    avg_distances = defaultdict(dict)
    for pair, layers in distances_by_pair_layer.items():
        for layer, dist_list in layers.items():
            if dist_list:
                avg_distances[pair][layer] = np.mean(dist_list)

    # Convert the averaged data to a pandas DataFrame for easy plotting
    df_avg = pd.DataFrame(avg_distances).sort_index()

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot a line for each input type pair
    for column in df_avg.columns:
        ax.plot(df_avg.index, df_avg[column], marker='o', linestyle='-', label=column)

    # Set plot titles and labels
    word_group = json_file_path.split('/')[-1].split('_')[1]
    model_name = json_file_path.split('/')[-1].split('_')[2].replace('.json', '')
    ax.set_title(f'Average Cosine Distance between Input Types per Layer ({word_group}, {model_name})', fontsize=16)
    ax.set_xlabel('Layer Number', fontsize=12)
    ax.set_ylabel('Average Cosine Distance', fontsize=12)

    # Customize ticks and legend
    ax.set_xticks(df_avg.index)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Input Type Pairs', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'results/plots/layer_divergence/distances_{word_group}_{model_name}.png', dpi=300)

if __name__ == '__main__':
    # The path to your JSON file
    json_path = 'results/layer_divergence/distances_natural_Qwen2.5-Omni-3B.json'
    plot_average_cosine_distances(json_path)