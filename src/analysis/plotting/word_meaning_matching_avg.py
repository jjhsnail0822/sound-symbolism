import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

def create_summary_plot(plot_data, title, output_filename, y_lims):
    """
    Creates and saves a summary plot showing average model accuracies.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 7))

    x_labels = list(plot_data.keys())
    x_pos = np.arange(len(x_labels))

    common_plotted = False
    rare_plotted = False

    for i, label in enumerate(x_labels):
        accs = plot_data.get(label, {})
        if not accs:
            continue

        common_acc = accs.get('COMMON', np.nan)
        rare_acc = accs.get('RARE', np.nan)
        x = x_pos[i]

        # Draw vertical bar connecting common and rare accuracies
        if not np.isnan(common_acc) and not np.isnan(rare_acc):
            bar_bottom = min(rare_acc, common_acc)
            bar_height = abs(common_acc - rare_acc)
            ax.bar(x, bar_height, width=0.4, bottom=bar_bottom, color='lightgrey', zorder=1)

        # Plot points for average accuracies as horizontal lines
        line_width = 0.4
        if not np.isnan(common_acc):
            ax.hlines(y=common_acc, xmin=x - line_width / 2, xmax=x + line_width / 2,
                      color='black', lw=3, zorder=2,
                      label='Common (Avg)' if not common_plotted else "")
            common_plotted = True
        if not np.isnan(rare_acc):
            ax.hlines(y=rare_acc, xmin=x - line_width / 2, xmax=x + line_width / 2,
                      color='purple', lw=3, zorder=2,
                      label='Rare (Avg)' if not rare_plotted else "")
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

    # --- Create Legend ---
    ax.legend(frameon=True, fontsize=11, loc='best')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    plt.show()

# --- Main script ---

# 1) Define experiment configurations
exp_configs = {
    'Original Text': 'original',
    'Romanized': 'romanized',
    'IPA': 'ipa',
    'Audio': 'audio'
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

# 3) Re-structure data for plotting (calculate averages)
plot_data_w2m = {}
plot_data_m2w = {}
all_models = set()

for exp_label, exp_data in all_data.items():
    # Collect all models found
    for model in exp_data.keys():
        all_models.add(model)

    # Collect accuracies for averaging
    w2m_common_accs = [d.get('COMMON Word→Meaning', np.nan) for d in exp_data.values()]
    w2m_rare_accs = [d.get('RARE Word→Meaning', np.nan) for d in exp_data.values()]
    m2w_common_accs = [d.get('COMMON Meaning→Word', np.nan) for d in exp_data.values()]
    m2w_rare_accs = [d.get('RARE Meaning→Word', np.nan) for d in exp_data.values()]

    # Filter out NaNs before averaging
    w2m_common_accs = [acc for acc in w2m_common_accs if not np.isnan(acc)]
    w2m_rare_accs = [acc for acc in w2m_rare_accs if not np.isnan(acc)]
    m2w_common_accs = [acc for acc in m2w_common_accs if not np.isnan(acc)]
    m2w_rare_accs = [acc for acc in m2w_rare_accs if not np.isnan(acc)]

    # Calculate and store averages for Word->Meaning
    if w2m_common_accs or w2m_rare_accs:
        plot_data_w2m[exp_label] = {
            'COMMON': np.mean(w2m_common_accs) if w2m_common_accs else np.nan,
            'RARE': np.mean(w2m_rare_accs) if w2m_rare_accs else np.nan
        }

    # Calculate and store averages for Meaning->Word
    if m2w_common_accs or m2w_rare_accs:
        plot_data_m2w[exp_label] = {
            'COMMON': np.mean(m2w_common_accs) if m2w_common_accs else np.nan,
            'RARE': np.mean(m2w_rare_accs) if m2w_rare_accs else np.nan
        }

if all_models:
    print(f"Averaging results over {len(all_models)} models: {sorted(list(all_models))}")

# 4) Calculate global y-axis limits for both plots
all_accuracies = []
for data_dict in [plot_data_w2m, plot_data_m2w]:
    for accs in data_dict.values():
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
        'Word→Meaning Matching Accuracy (Model Average)',
        'results/plots/experiments/word_meaning_matching/summary_w2m_avg.png',
        y_lims
    )

if any(plot_data_m2w.values()):
    create_summary_plot(
        plot_data_m2w,
        'Meaning→Word Matching Accuracy (Model Average)',
        'results/plots/experiments/word_meaning_matching/summary_m2w_avg.png',
        y_lims
    )