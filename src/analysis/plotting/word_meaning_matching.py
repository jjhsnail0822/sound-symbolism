import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

def create_summary_plot(plot_data, all_models, title, output_filename, y_lims):
    """
    Creates and saves a summary plot showing individual model accuracies with jitter.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    # Adjust figsize for a denser plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x_labels = list(plot_data.keys())
    x_pos = np.arange(len(x_labels))

    # --- Jitter and Color setup ---
    model_list = sorted(list(all_models))
    num_models = len(model_list)
    # Further reduced jitter_width to make lines closer
    jitter_width = 0.20
    if num_models > 1:
        jitters = np.linspace(-jitter_width / 2, jitter_width / 2, num_models)
    else:
        jitters = [0]
    model_jitter = {model: jitters[i] for i, model in enumerate(model_list)}

    # Assign a unique color to each model
    colors = plt.get_cmap('tab10', num_models)
    model_colors = {model: colors(i) for i, model in enumerate(model_list)}

    # For creating a clean legend for points
    common_plotted = False
    rare_plotted = False

    for i, label in enumerate(x_labels):
        exp_models = plot_data.get(label, {})
        if not exp_models:
            continue

        for model_name, accs in exp_models.items():
            common_acc = accs.get('COMMON', np.nan)
            rare_acc = accs.get('RARE', np.nan)

            jitter = model_jitter.get(model_name, 0)
            x = x_pos[i] + jitter
            model_color = model_colors.get(model_name, 'grey')

            # Draw vertical line for each model using its assigned color (thicker line)
            if not np.isnan(common_acc) and not np.isnan(rare_acc):
                ax.vlines(x, rare_acc, common_acc, color=model_color, linestyle='-', linewidth=3, alpha=0.8, zorder=1)

            # Plot points for each model (larger points)
            if not np.isnan(common_acc):
                ax.scatter(x, common_acc, color='black', s=70, zorder=2,
                           label='Common' if not common_plotted else "")
                common_plotted = True
            if not np.isnan(rare_acc):
                ax.scatter(x, rare_acc, color='purple', s=70, zorder=2,
                           label='Rare' if not rare_plotted else "")
                rare_plotted = True

    # --- Plot Styling (larger fonts) ---
    ax.set_ylabel('Accuracy', fontsize=15)
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=13)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax.set_ylim(y_lims)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x')

    # --- Create Combined Legend (with frame, no title, larger font) ---
    # Get handles for 'Common' and 'Rare' points
    handles, labels = ax.get_legend_handles_labels()
    
    # Create proxy artists for model colors (thicker line in legend)
    model_legend_elements = [Line2D([0], [0], color=model_colors[model], lw=3, label=model)
                             for model in model_list if model in model_colors]
    
    # Combine handles and display legend
    handles.extend(model_legend_elements)
    ax.legend(handles=handles, frameon=True, fontsize=11, loc='best')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    plt.show()

# --- Main script ---

# 1) Define experiment configurations
exp_configs = {
    'Original': 'original',
    'Original & Audio': 'original_and_audio',
    'Romanized': 'romanized',
    'Romanized & Audio': 'romanized_and_audio',
    'IPA': 'ipa',
    'IPA & Audio': 'ipa_and_audio',
    'Audio': 'audio',
}
exp_base_dir = 'results/experiments/word_meaning_matching'

# 2) Load data from all experiments
all_data = {}
for label, folder_name in exp_configs.items():
    exp_dir = os.path.join(exp_base_dir, folder_name)
    files = glob.glob(os.path.join(exp_dir, 'all_results_*.json'))
    if not files:
        print(f"Warning: No result files found for experiment '{label}' in {exp_dir}")
        continue

    data_per_exp = {}
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            arr = json.load(f)
        for rec in arr:
            m = rec['model'].split('/')[-1]
            dp = os.path.basename(rec['data_path'])
            word_group = dp.split('-')[-1].split('.')[0].upper()
            task_type = 'Word→Meaning' if 'word_to_meaning' in dp else 'Meaning→Word'
            cat = f'{word_group} {task_type}'
            if m not in data_per_exp:
                data_per_exp[m] = {}
            data_per_exp[m][cat] = rec['accuracy']
    all_data[label] = data_per_exp

# 3) Re-structure data for plotting (without averaging)
plot_data_w2m = {}
plot_data_m2w = {}
all_models = set()

for exp_label, exp_data in all_data.items():
    plot_data_w2m[exp_label] = {}
    plot_data_m2w[exp_label] = {}
    for model, model_data in exp_data.items():
        all_models.add(model)
        
        common_w2m = model_data.get('COMMON Word→Meaning', np.nan)
        rare_w2m = model_data.get('RARE Word→Meaning', np.nan)
        if not (np.isnan(common_w2m) and np.isnan(rare_w2m)):
            plot_data_w2m[exp_label][model] = {'COMMON': common_w2m, 'RARE': rare_w2m}

        common_m2w = model_data.get('COMMON Meaning→Word', np.nan)
        rare_m2w = model_data.get('RARE Meaning→Word', np.nan)
        if not (np.isnan(common_m2w) and np.isnan(rare_m2w)):
            plot_data_m2w[exp_label][model] = {'COMMON': common_m2w, 'RARE': rare_m2w}

# 4) Calculate global y-axis limits for both plots
all_accuracies = []
for data_dict in [plot_data_w2m, plot_data_m2w]:
    for exp_models in data_dict.values():
        for accs in exp_models.values():
            if not np.isnan(accs.get('COMMON')):
                all_accuracies.append(accs['COMMON'])
            if not np.isnan(accs.get('RARE')):
                all_accuracies.append(accs['RARE'])

if all_accuracies:
    min_val, max_val = np.nanmin(all_accuracies), np.nanmax(all_accuracies)
    padding = (max_val - min_val) * 0.1
    y_lims = (max(0, min_val - padding), min(1.0, max_val + padding))
else:
    y_lims = (0, 1) # Default if no data

# 5) Generate and save the two plots
if any(plot_data_w2m.values()):
    create_summary_plot(
        plot_data_w2m,
        all_models,
        'Word→Meaning Matching Accuracy (All Models)',
        'results/plots/experiments/word_meaning_matching/summary_w2m_indiv.png',
        y_lims
    )

if any(plot_data_m2w.values()):
    create_summary_plot(
        plot_data_m2w,
        all_models,
        'Meaning→Word Matching Accuracy (All Models)',
        'results/plots/experiments/word_meaning_matching/summary_m2w_indiv.png',
        y_lims
    )