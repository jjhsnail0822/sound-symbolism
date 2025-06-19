import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from collections import defaultdict
import re

# --- Data Loading and Processing Function ---
def get_dimension_accuracies(result_file_path):
    """
    Calculates the accuracy for each semantic dimension from the result file.
    """
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
    except FileNotFoundError:
        # print(f"Warning: Result file not found: {result_file_path}")
        return None

    results_list = result_data.get('results', [])
    if not results_list:
        print(f"Warning: No 'results' found in {result_file_path}")
        return None

    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for item in results_list:
        # Extract 'dimension' information from 'meta_data'
        meta_data = item.get('meta_data')
        if not isinstance(meta_data, dict):
            continue
            
        dimension = meta_data.get('dimension')
        if dimension is None:
            continue
        
        is_correct = item.get('is_correct')
        if is_correct is None:
            continue

        total_counts[dimension] += 1
        if is_correct:
            correct_counts[dimension] += 1

    if not total_counts:
        return None

    dim_accuracies = {dim: correct_counts[dim] / total_counts[dim] for dim in total_counts}
    return dim_accuracies

# --- Plotting Function ---
def plot_sorted_bars(ax, accuracies, title, color_map):
    """
    Creates a vertical bar chart of dimensions sorted by accuracy, using a consistent color map.
    """
    if not accuracies:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    # Sort in descending order based on accuracy
    sorted_items = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)
    dimensions = [item[0].replace('_', ' ').title() for item in sorted_items]
    acc_values = [item[1] for item in sorted_items]
    
    # Assign colors based on the consistent dimension-to-color map
    bar_colors = [color_map.get(item[0], '#CCCCCC') for item in sorted_items] # Default to gray if not found

    bars = ax.bar(dimensions, acc_values, color=bar_colors)

    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    
    # Add a 50% baseline
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1.2, label='50% Baseline')
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

# --- Main Execution ---
def main():
    """
    Main function to run the analysis and plotting for all specified experiments.
    Averages results across all found models for each experiment.
    Ensures consistent dimension coloring across all plots.
    """
    # --- Configuration ---
    EXPERIMENTS = [
        {
            "name": "semantic_dimension_binary_original",
            "title": "Semantic Dimension Breakdown Averages (Original Text)",
            "dir": "results/experiments/semantic_dimension/binary/original"
        },
        {
            "name": "semantic_dimension_binary_romanized",
            "title": "Semantic Dimension Breakdown Averages (Romanized)",
            "dir": "results/experiments/semantic_dimension/binary/romanized"
        },
        {
            "name": "semantic_dimension_binary_ipa",
            "title": "Semantic Dimension Breakdown Averages (IPA)",
            "dir": "results/experiments/semantic_dimension/binary/ipa"
        },
        {
            "name": "semantic_dimension_binary_audio",
            "title": "Semantic Dimension Breakdown Averages (Audio)",
            "dir": "results/experiments/semantic_dimension/binary/audio"
        }
    ]

    output_base_dir = os.path.join('results', 'plots', 'experiments', 'semantic_dimension_binary_breakdown')
    os.makedirs(output_base_dir, exist_ok=True)

    # --- Phase 1: Collect all data and dimension names ---
    all_experiments_data = []
    all_dimension_names = set()

    for config in EXPERIMENTS:
        EXP_NAME = config["name"]
        exp_dir = config["dir"]
        
        print(f"\n{'='*40}\nProcessing Experiment: {EXP_NAME}\n{'='*40}")

        all_files = glob.glob(os.path.join(exp_dir, f'{EXP_NAME}-*.json'))
        
        model_names = set()
        for f in all_files:
            basename = os.path.basename(f)
            match = re.search(f'^{re.escape(EXP_NAME)}-(?:common|rare)_(.*)\\.json$', basename)
            if match:
                model_names.add(match.group(1))

        if not model_names:
            print(f"No result files found in '{exp_dir}' for experiment '{EXP_NAME}'. Skipping.")
            continue

        model_names = sorted(list(model_names))
        print(f"Found {len(model_names)} models for averaging: {', '.join(model_names)}")

        all_common_accuracies = defaultdict(list)
        all_rare_accuracies = defaultdict(list)

        for model_name in model_names:
            common_file_path = os.path.join(exp_dir, f"{EXP_NAME}-common_{model_name}.json")
            rare_file_path = os.path.join(exp_dir, f"{EXP_NAME}-rare_{model_name}.json")

            common_accuracies = get_dimension_accuracies(common_file_path)
            if common_accuracies:
                for dim, acc in common_accuracies.items():
                    all_common_accuracies[dim].append(acc)

            rare_accuracies = get_dimension_accuracies(rare_file_path)
            if rare_accuracies:
                for dim, acc in rare_accuracies.items():
                    all_rare_accuracies[dim].append(acc)

        avg_common_accuracies = {dim: np.mean(accs) for dim, accs in all_common_accuracies.items()}
        avg_rare_accuracies = {dim: np.mean(accs) for dim, accs in all_rare_accuracies.items()}

        if avg_common_accuracies or avg_rare_accuracies:
            all_experiments_data.append({
                "config": config,
                "common": avg_common_accuracies,
                "rare": avg_rare_accuracies
            })
            all_dimension_names.update(avg_common_accuracies.keys())
            all_dimension_names.update(avg_rare_accuracies.keys())
        else:
            print(f"  Skipping plot for {EXP_NAME} due to missing data across all models.")

    # --- Phase 2: Create a consistent color map for all dimensions ---
    sorted_dims = sorted(list(all_dimension_names))
    num_dims = len(sorted_dims)

    # Select a color map that provides good visual distinction for the number of dimensions.
    if num_dims <= 20:
        # 'tab20' is excellent for up to 20 categories.
        colors = list(plt.cm.get_cmap('tab20').colors)
    elif num_dims <= 40:
        # For up to 40, we can combine 'tab20' and 'tab20b' for more distinct colors.
        colors = list(plt.cm.get_cmap('tab20').colors) + list(plt.cm.get_cmap('tab20b').colors)
    else:
        # For a large number of categories, 'hsv' provides a wide spectrum of distinct hues.
        # This is better for categorical data than a sequential map like 'viridis'.
        colors = plt.cm.hsv(np.linspace(0, 1, num_dims))

    dimension_color_map = {dim: colors[i] for i, dim in enumerate(sorted_dims)}

    # --- Phase 3: Plot all experiments with the consistent color map ---
    for exp_data in all_experiments_data:
        config = exp_data["config"]
        avg_common_accuracies = exp_data["common"]
        avg_rare_accuracies = exp_data["rare"]
        
        TITLE_PREFIX = config["title"]
        EXP_NAME = config["name"]

        print(f"\n--- Plotting for {EXP_NAME} ---")

        fig, axes = plt.subplots(1, 2, figsize=(28, 12), sharey=True)
        fig.suptitle(f'{TITLE_PREFIX}', fontsize=20, weight='bold')

        plot_sorted_bars(axes[0], avg_common_accuracies, 'Common Words', dimension_color_map)
        plot_sorted_bars(axes[1], avg_rare_accuracies, 'Rare Words', dimension_color_map)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        out_png = os.path.join(output_base_dir, f'{EXP_NAME}_average_breakdown.png')
        
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to {out_png}")
        plt.close(fig)

if __name__ == '__main__':
    main()