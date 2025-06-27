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
def plot_sorted_bars(ax, accuracies, title):
    """
    Creates a vertical bar chart of dimensions sorted by accuracy.
    Bar color is determined by accuracy value (blue for <0.5, red for >0.5).
    """
    if not accuracies:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    # Sort in descending order based on accuracy
    sorted_items = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)
    dimensions = [item[0].replace('_', ' ').title() for item in sorted_items]
    acc_values = np.array([item[1] for item in sorted_items])
    
    # Assign colors based on accuracy value relative to 0.5 baseline
    # Using a diverging colormap (coolwarm: blue -> white -> red)
    cmap = plt.get_cmap('coolwarm')
    
    # Create a continuous, non-linear mapping to enhance color saturation.
    # This avoids the pale center of the 'coolwarm' map while maintaining a smooth gradient.
    power = 0.25  # Exponent < 1 pushes values away from the center (0.5), increasing saturation.
    centered_values = acc_values - 0.5  # Center values around 0. Range: [-0.5, 0.5]
    
    # Apply power transformation to absolute values and then restore sign
    transformed_abs = (np.abs(centered_values) / 0.5)**power * 0.5
    color_norm_values = 0.5 + np.sign(centered_values) * transformed_abs

    bar_colors = cmap(color_norm_values)

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
    Bar colors are based on accuracy relative to the 50% baseline.
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

    # --- Phase 1: Collect all data ---
    all_experiments_data = []

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
        else:
            print(f"  Skipping plot for {EXP_NAME} due to missing data across all models.")

    # --- Phase 2: Plot all experiments ---
    for exp_data in all_experiments_data:
        config = exp_data["config"]
        avg_common_accuracies = exp_data["common"]
        avg_rare_accuracies = exp_data["rare"]
        
        TITLE_PREFIX = config["title"]
        EXP_NAME = config["name"]

        print(f"\n--- Plotting for {EXP_NAME} ---")

        fig, axes = plt.subplots(1, 2, figsize=(28, 12), sharey=True)
        fig.suptitle(f'{TITLE_PREFIX}', fontsize=20, weight='bold')

        plot_sorted_bars(axes[0], avg_common_accuracies, 'Common Words')
        plot_sorted_bars(axes[1], avg_rare_accuracies, 'Rare Words')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        out_png = os.path.join(output_base_dir, f'{EXP_NAME}_average_breakdown.png')
        
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to {out_png}")
        plt.close(fig)

if __name__ == '__main__':
    main()