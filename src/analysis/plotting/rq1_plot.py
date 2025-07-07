import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.font_manager import FontProperties

# python src/analysis/plotting/rq1_plot.py --json_path ./results/statistics/semdim_stat.json --metric f1 --sem_dims "sem_dim_01,sem_dim_05,sem_dim_12"

def load_stat_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def weighted_avg(values, weights):
    values = np.array(values)
    weights = np.array(weights)
    if weights.sum() == 0:
        return 0.0
    return np.sum(values * weights) / np.sum(weights)

def extract_metric(entry, metric='f1'):
    # entry: [F1, num_of_words] or [F1, num_of_words, acc]
    if metric.lower() == 'f1':
        return entry[0]
    elif metric.lower() == 'accuracy' or metric.lower() == 'acc':
        return entry[2] if len(entry) > 2 else entry[0]
    else:
        raise ValueError(f"Unknown metric: {metric}")

def extract_num_words(entry):
    return entry[1]

def compute_avg_by_condition(data, group_keys, metric='f1'):
    # group_keys: list of keys to group by (e.g., ['model', 'word_type'])
    # Returns: {semantic_dim: (weighted_avg, total_words)}
    from collections import defaultdict
    semdim_scores = defaultdict(list)
    semdim_words = defaultdict(list)
    # Traverse all combinations
    for model_name, model_data in data.items():
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                for sem_dim, entry in input_data.items():
                    key_dict = {
                        'model': model_name,
                        'word_type': word_type,
                        'input_type': input_type
                    }
                    group = tuple(key_dict[k] for k in group_keys)
                    semdim_scores[(group, sem_dim)].append(extract_metric(entry, metric))
                    semdim_words[(group, sem_dim)].append(extract_num_words(entry))
    avg_dict = {}
    for (group, sem_dim), scores in semdim_scores.items():
        weights = semdim_words[(group, sem_dim)]
        avg = weighted_avg(scores, weights)
        total_words = np.sum(weights)
        avg_dict[(group, sem_dim)] = (avg, total_words)
    return avg_dict

def plot_horizontal_bar(avg_dict, group, metric='f1', title=None, save_path=None, sem_dims=None):
    # group: tuple (e.g., (model_name,), (word_type,), (input_type,), (model_name, input_type))
    # avg_dict: {(group, sem_dim): (avg, total_words)}
    # sem_dims: list of semantic dimensions to plot (or None for all)
    semdim_avgs = {}
    for (g, sem_dim), (avg, total_words) in avg_dict.items():
        if g == group:
            if sem_dims is None or sem_dim in sem_dims:
                semdim_avgs[sem_dim] = avg
    if not semdim_avgs:
        return
    sorted_items = sorted(semdim_avgs.items(), key=lambda x: x[1], reverse=True)
    sem_dims_plot, avgs = zip(*sorted_items)
    top3 = set([sem_dims_plot[i] for i in range(min(3, len(sem_dims_plot)))])
    bottom3 = set([sem_dims_plot[i] for i in range(-3, 0)])
    colors = []
    for v in avgs:
        if v >= 0.5:
            colors.append((1, 1 - (v-0.5)*2, 1 - (v-0.5)*2))
        else:
            colors.append((1 - (0.5-v)*2, 1 - (0.5-v)*2, 1))
    font_bold = FontProperties(weight='bold')
    y_labels = []
    for sd in sem_dims_plot:
        if sd in top3 or sd in bottom3:
            y_labels.append(f"$\\bf{{{sd}}}$")
        else:
            y_labels.append(sd)
    fig, ax = plt.subplots(figsize=(8, max(6, len(sem_dims_plot)*0.4)))
    bars = ax.barh(range(len(avgs)), avgs, color=colors, edgecolor='black')
    ax.set_yticks(range(len(avgs)))
    ax.set_yticklabels(y_labels, fontsize=12, va='center')
    ax.set_xlabel(f'{metric} score', fontsize=13)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    for i, label in enumerate(ax.get_yticklabels()):
        if sem_dims_plot[i] in top3 or sem_dims_plot[i] in bottom3:
            label.set_fontweight('bold')
    if title:
        ax.set_title(title, fontsize=15, pad=15)
    plt.tight_layout()
    if save_path:
        file_name = f"{metric}_{group[0]}_{group[1]}_{len(sem_dims)}_dims.png"
        plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    plt.close()

def main(json_path, metric='f1', sem_dims=None):
    data = load_stat_json(json_path)
    avg_by_model = compute_avg_by_condition(data, ['model'], metric)
    for model_name in set(g[0] for (g, sd) in avg_by_model.keys()):
        plot_horizontal_bar(avg_by_model, (model_name,), metric, title=f"Model: {model_name}", sem_dims=sem_dims)
    avg_by_wordtype = compute_avg_by_condition(data, ['word_type'], metric)
    for word_type in set(g[0] for (g, sd) in avg_by_wordtype.keys()):
        plot_horizontal_bar(avg_by_wordtype, (word_type,), metric, title=f"Word Type: {word_type}", sem_dims=sem_dims)
    avg_by_inputtype = compute_avg_by_condition(data, ['input_type'], metric)
    for input_type in set(g[0] for (g, sd) in avg_by_inputtype.keys()):
        plot_horizontal_bar(avg_by_inputtype, (input_type,), metric, title=f"Input Type: {input_type}", sem_dims=sem_dims)
    avg_by_model_input = compute_avg_by_condition(data, ['model', 'input_type'], metric)
    model_input_groups = set(g for (g, sd) in avg_by_model_input.keys())
    for group in model_input_groups:
        plot_horizontal_bar(avg_by_model_input, group, metric, title=f"Model: {group[0]}, Input Type: {group[1]}", sem_dims=sem_dims)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='./results/statistics/semdim_stat.json')
    parser.add_argument('--metric', type=str, default='f1', help='f1 or accuracy')
    parser.add_argument('--sem_dims', type=str, default=None, help='Comma-separated list of semantic dimensions to plot (default: all)')
    parser.add_argument('--save_path', type=str, default="./results/plots/rq1", help='Path to save the plots')
    args = parser.parse_args()
    sem_dims = None
    if args.sem_dims:
        sem_dims = [s.strip() for s in args.sem_dims.split(',') if s.strip()]
    main(args.json_path, metric=args.metric, sem_dims=sem_dims)
