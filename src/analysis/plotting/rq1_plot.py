# python src/analysis/plotting/rq1_plot.py --json_path ./results/statistics/semdim_stat.json --metric macro_f1_score --filter_constructed

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.font_manager import FontProperties
import pandas as pd
from matplotlib.colors import to_rgb


constructed_dims = {
    "beautiful-ugly", "delicate-rugged", "tense-relaxed", "simple-complex", 
    "happy-sad", "sharp-round", "fast-slow", "masculine-feminine", 
    "strong-weak", "exciting-calming", "harsh-mellow", "ordinary-unique", 
    "dangerous-safe", "realistic-fantastical", "abrupt-continuous", 
    "passive-active", "big-small", "heavy-light", "hard-soft", 
    "solid-nonsolid", "inhibited-free"
}

def get_metric_label(metric='macro_f1_score'):
    """Helper function to get proper metric labels for plots"""
    if metric.lower() == 'macro_f1_score':
        return 'Macro F1 Score'
    elif metric.lower() == 'accuracy' or metric.lower() == 'acc':
        return 'Accuracy'
    elif metric.lower() == 'precision':
        return 'Precision'
    elif metric.lower() == 'recall':
        return 'Recall'
    else:
        return metric.upper()

def find_category(sem_dim):
    categories = {
        "valence": ["good-bad", "beautiful-ugly", "pleasant-unpleasant"],
        "potency": ["strong-weak", "big-small", "rugged_delicate"],
        "activity": ["active-passive", "sharp-round", "fast-slow"],
        "other factors": ["realistic-fantastical", "structured-disorganized", "ordinary-unique", "interesting-uninteresting", "simple-complex"],
        "stimulus dimensions": ["abrupt-continuous", "exciting-calming", "hard-soft", "happy-sad", "harsh-mellow", "heavy-light", "inhibited-free", "masculine-feminine", "solid-nonsolid", "tense-relaxed"],
        "other dimensions": ["dangerous-safe"]
    }
    for category, sem_dims in categories.items():
        if sem_dim in sem_dims:
            return category
    return None

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

def extract_metric(entry, metric='macro_f1_score'):
    # entry: {"accuracy": 0.7, "precision": 0.7, "recall": 0.7, "f1_score": 0.7, "count": 2800}
    return entry[metric]

def extract_num_words(entry):
    return entry["count"]

def extract_data_by_condition(data, condition_keys, metric='macro_f1_score', filter_constructed=False, constructed_dims=constructed_dims):
    from collections import defaultdict
    condition_scores = defaultdict(lambda: defaultdict(list))
    condition_words = defaultdict(lambda: defaultdict(list))
    
    for model_name, model_data in data.items():
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                            continue
                    
                    condition_dict = {
                        'model': model_name,
                        'word_type': word_type,
                        'input_type': input_type
                    }
                    condition = tuple(condition_dict[k] for k in condition_keys)
                    condition_scores[condition][sem_dim].append(extract_metric(entry, metric))
                    condition_words[condition][sem_dim].append(extract_num_words(entry))
    
    return condition_scores, condition_words

def compute_averages_from_scores(condition_scores, condition_words):
    avg_dict = {}
    for condition, semdim_scores in condition_scores.items():
        for sem_dim, scores in semdim_scores.items():
            weights = condition_words[condition][sem_dim]
            avg = weighted_avg(scores, weights)
            total_words = np.sum(weights)
            avg_dict[(condition, sem_dim)] = (avg, total_words)
    return avg_dict

def setup_plot_style(ax, xlabel='', ylabel='', title='', xlim=None, ylim=(0, 1)):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if title:
        ax.set_title(title, fontsize=15, pad=15)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

    ax.grid(True, alpha=0.3)

def get_category_colors():
    return {
        "valence": "#FF6B6B",
        "potency": "#4ECDC4",
        "activity": "#45B7D1",
        "other factors": "#96CEB4",
        "stimulus dimensions": "#FF8C42",
        "other dimensions": "#DDA0DD"
    }

def save_plot(fig, save_path, file_name, dpi=300):
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, file_name)
    fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to {full_path}")
    plt.close()

def extract_wordtype_data(data, metric='macro_f1_score', filter_constructed=False, constructed_dims=constructed_dims):
    from collections import defaultdict
    wordtype_scores = defaultdict(lambda: defaultdict(list))
    wordtype_words = defaultdict(lambda: defaultdict(list))
    
    for model_name, model_data in data.items():
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                        continue
                    wordtype_scores[word_type][sem_dim].append(extract_metric(entry, metric))
                    wordtype_words[word_type][sem_dim].append(extract_num_words(entry))
    
    return wordtype_scores, wordtype_words

def extract_inputtype_data(data, metric='macro_f1_score', filter_constructed=False, constructed_dims=constructed_dims):
    from collections import defaultdict
    inputtype_scores = defaultdict(lambda: defaultdict(list))
    inputtype_words = defaultdict(lambda: defaultdict(list))
    
    for model_name, model_data in data.items():
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                        continue
                    inputtype_scores[input_type][sem_dim].append(extract_metric(entry, metric))
                    inputtype_words[input_type][sem_dim].append(extract_num_words(entry))
    
    return inputtype_scores, inputtype_words

def compute_avg_by_condition(data, group_keys, metric='macro_f1_score', filter_constructed=False, constructed_dims=constructed_dims):
    condition_scores, condition_words = extract_data_by_condition(data, group_keys, metric, filter_constructed, constructed_dims)
    return compute_averages_from_scores(condition_scores, condition_words)

def plot_horizontal_bar(avg_dict, group, metric='macro_f1_score', title=None, save_path=None, sem_dims=None, filter_constructed=False):
    semdim_avgs = {}
    for (g, sem_dim), (avg, total_words) in avg_dict.items():
        if g == group:
            if sem_dims is None or sem_dim in sem_dims:
                semdim_avgs[sem_dim] = avg
    
    if not semdim_avgs:
        return
    sorted_items = sorted(semdim_avgs.items(), key=lambda x: x[1], reverse=False)
    sem_dims_plot, avgs = zip(*sorted_items)
    top3 = set([sem_dims_plot[i] for i in range(min(3, len(sem_dims_plot)))])
    bottom3 = set([sem_dims_plot[i] for i in range(-3, 0)])
    
    category_colors = get_category_colors()

    if not semdim_avgs:
        return
    sorted_items = sorted(semdim_avgs.items(), key=lambda x: x[1], reverse=False)
    sem_dims_plot, avgs = zip(*sorted_items)
    
    colors = []
    for v in avgs:
        if v >= 0.5:
            green_intensity = min(1.0, (v - 0.5) * 2)
            colors.append((1 - green_intensity * 0.7, 1 - green_intensity * 0.3, 1 - green_intensity * 0.7))
        else:
            red_intensity = min(1.0, (0.5 - v) * 2)
            colors.append((1 - red_intensity * 0.3, 1 - red_intensity * 0.7, 1 - red_intensity * 0.7))
    
    y_labels = []
    text_colors = []
    for sd in sem_dims_plot:
        category = find_category(sd)
        if category and category in category_colors:
            text_color = category_colors[category]
        else:
            text_color = 'black'
        
        if sd in top3 or sd in bottom3:
            y_labels.append(f"$\\bf{{{sd}}}$")
        else:
            y_labels.append(sd)
        text_colors.append(text_color)
    fig, ax = plt.subplots(figsize=(8, max(6, len(sem_dims_plot)*0.4)))
    bars = ax.barh(range(len(avgs)), avgs, color=colors, edgecolor='black')
    ax.set_yticks(range(len(avgs)))
    ax.set_yticklabels(y_labels, fontsize=12, va='center')
    setup_plot_style(ax, xlabel=get_metric_label(metric), xlim=(0, 1))
    
    for i, label in enumerate(ax.get_yticklabels()):
        if sem_dims_plot[i] in top3 or sem_dims_plot[i] in bottom3:
            label.set_fontweight('bold')
        label.set_color(text_colors[i])
    
    if title:
        ax.set_title(title, fontsize=15, pad=15)
    plt.tight_layout()
    group_str = '_'.join(str(g) for g in group)

    if group_str == '':
        full_save_path = os.path.join(save_path, metric)
    else:
        full_save_path = os.path.join(save_path, metric, group_str)
    
    suffix = "_filter" if filter_constructed else ""
    file_name = f"{len(sem_dims) if sem_dims is not None else 'all'}_dims{suffix}.png"
    save_plot(fig, full_save_path, file_name)

def compute_avg_by_category(data, metric='macro_f1_score', filter_constructed=False, constructed_dims=constructed_dims):
    from collections import defaultdict
    category_scores = defaultdict(list)
    category_words = defaultdict(list)
    
    for model_name, model_data in data.items():
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed:
                        if sem_dim not in constructed_dims:
                            continue
                    
                    category = find_category(sem_dim)
                    if category:
                        category_scores[category].append(extract_metric(entry, metric))
                        category_words[category].append(extract_num_words(entry))
    
    category_avgs = {}
    for category, scores in category_scores.items():
        weights = category_words[category]
        avg = weighted_avg(scores, weights)
        total_words = np.sum(weights)
        category_avgs[category] = (avg, total_words)
    
    return category_avgs

def plot_wordtype_performance(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims):
    """Plot model performance by word type (constructed, common, rare) using line plot"""
    from collections import defaultdict
    
    # Extract data by model and word type
    model_wordtype_scores = defaultdict(lambda: defaultdict(list))
    model_wordtype_words = defaultdict(lambda: defaultdict(list))
    
    for model_name, model_data in data.items():
        for word_type, word_data in model_data.items():
            if word_type not in ['constructed', 'common', 'rare']:
                continue
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                        continue
                    model_wordtype_scores[model_name][word_type].append(extract_metric(entry, metric))
                    model_wordtype_words[model_name][word_type].append(extract_num_words(entry))
    
    # Calculate averages for each model and word type
    model_wordtype_avgs = {}
    for model_name in model_wordtype_scores:
        model_wordtype_avgs[model_name] = {}
        for word_type in ['constructed', 'common', 'rare']:
            if word_type in model_wordtype_scores[model_name]:
                scores = model_wordtype_scores[model_name][word_type]
                weights = model_wordtype_words[model_name][word_type]
                if scores:
                    avg = weighted_avg(scores, weights)
                    model_wordtype_avgs[model_name][word_type] = avg
    
    if not model_wordtype_avgs:
        return
    
    word_types = ['constructed', 'common', 'rare']
    models = list(model_wordtype_avgs.keys())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use different colors for each model
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    markers = ['o', 's', '^', 'D', '*', 'v', '<', '>', 'p', 'h']
    
    for i, model_name in enumerate(models):
        values = []
        for word_type in word_types:
            value = model_wordtype_avgs[model_name].get(word_type, 0.0)
            values.append(value)
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax.plot(word_types, values, marker=marker, linewidth=2, markersize=8, 
                color=color, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Word Types', fontsize=13)
    ax.set_ylabel(get_metric_label(metric), fontsize=13)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    # Add horizontal line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    suffix = "_filter" if filter_constructed else ""
    file_name = f"wordtype_performance_{metric}{suffix}.png"
    save_plot(fig, save_path, file_name)

def plot_inputtype_performance(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims):
    from collections import defaultdict
    
    semdim_scores = defaultdict(lambda: defaultdict(list))
    semdim_words = defaultdict(lambda: defaultdict(list))
    
    for model_name, model_data in data.items():
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed:
                        if sem_dim not in constructed_dims:
                            continue
                    
                    semdim_scores[sem_dim][input_type].append(extract_metric(entry, metric))
                    semdim_words[sem_dim][input_type].append(extract_num_words(entry))
    
    semdim_avgs = {}
    for sem_dim in semdim_scores:
        semdim_avgs[sem_dim] = {}
        # for input_type in ['original', 'original_and_audio', 'audio', 'ipa_and_audio', 'ipa']:
        for input_type in ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']:
            if input_type in semdim_scores[sem_dim]:
                scores = semdim_scores[sem_dim][input_type]
                weights = semdim_words[sem_dim][input_type]
                avg = weighted_avg(scores, weights)
                semdim_avgs[sem_dim][input_type] = avg
            else:
                semdim_avgs[sem_dim][input_type] = 0.0
    
    if not semdim_avgs:
        return
    
    input_types = ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']
    sem_dims = list(semdim_avgs.keys())
    
    csv_data = []
    for sem_dim in sem_dims:
        row = {'semantic_dimension': sem_dim}
        for input_type in input_types:
            row[input_type] = semdim_avgs[sem_dim].get(input_type, 0.0)
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    csv_file_name = f"inputtype_performance_{metric}{suffix}.csv"
    csv_path = os.path.join(save_path, csv_file_name)
    df.to_csv(csv_path, index=False)
    print(f"Input type performance CSV saved to {csv_path}")
    
    print(f"\n=== Input Type Performance ({metric.upper()}) ===")
    print(df.to_string(index=False, float_format='%.3f'))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {
        "valence": "#FF6B6B",
        "potency": "#4ECDC4",
        "activity": "#45B7D1",
        "other factors": "#96CEB4",
        "stimulus dimensions": "#FF8C42",
        "other dimensions": "#DDA0DD"
    }
    
    for sem_dim in sem_dims:
        values = [semdim_avgs[sem_dim].get(it, 0.0) for it in input_types]
        category = find_category(sem_dim)
        color = colors.get(category, 'gray')
        line = ax.plot(input_types, values, marker='o', linewidth=2, markersize=6, color=color, label=sem_dim, alpha=0.8)
        
        original_value = semdim_avgs[sem_dim].get('original', 0.0)
        ax.text(-0.1, original_value, sem_dim, ha='right', va='center', fontsize=9, 
                color=color, alpha=0.8, transform=ax.get_yaxis_transform())
    
    ax.set_xlabel('Input Types', fontsize=13)
    ax.set_ylabel(get_metric_label(metric), fontsize=13)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

    
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    file_name = f"inputtype_performance_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Input type performance plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def plot_inputtype_wordtype_scatter(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims):
    from collections import defaultdict
    
    input_type_scores = defaultdict(lambda: defaultdict(list))
    input_type_words = defaultdict(lambda: defaultdict(list))
    
    for model_name, model_data in data.items():
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed:
                        if sem_dim not in constructed_dims:
                            continue
                    
                    input_type_scores[input_type][word_type].append(extract_metric(entry, metric))
                    input_type_words[input_type][word_type].append(extract_num_words(entry))
    
    input_type_avgs = {}
    for input_type in ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']:
        input_type_avgs[input_type] = {}
        for word_type in ['constructed', 'rare', 'common']:
            if word_type in input_type_scores[input_type]:
                scores = input_type_scores[input_type][word_type]
                weights = input_type_words[input_type][word_type]
                avg = weighted_avg(scores, weights)
                input_type_avgs[input_type][word_type] = avg
            else:
                input_type_avgs[input_type][word_type] = 0.0
    
    if not input_type_avgs:
        return
    
    input_types = ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']
    word_types = ['constructed', 'rare', 'common']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#ffd43b', '#20c997', '#f06595']
    markers = ['s', '^', 'o']
    
    for i, word_type in enumerate(word_types):
        x_coords = []
        y_coords = []
        for j, input_type in enumerate(input_types):
            if input_type in input_type_avgs and word_type in input_type_avgs[input_type]:
                x_coords.append(j)
                y_coords.append(input_type_avgs[input_type][word_type])
        
        ax.scatter(x_coords, y_coords, c=colors[i], marker=markers[i], s=100, 
                  label=word_type, alpha=0.8, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Input Types', fontsize=13)
    ax.set_ylabel(get_metric_label(metric), fontsize=13)
    ax.set_xticks(range(len(input_types)))
    ax.set_xticklabels(input_types, ha='right')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    file_name = f"inputtype_wordtype_scatter_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Input type vs word type scatter plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def plot_inputtype_wordtype_modeltype_scatter(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims):
    from collections import defaultdict
    
    def normalize_model_name(model_name):
        if 'gpt' in model_name.lower():
            return 'GPT'
        return model_name
    
    model_input_word_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    model_input_word_words = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for model_name, model_data in data.items():
        normalized_model = normalize_model_name(model_name)
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                            continue
                    
                    model_input_word_scores[normalized_model][input_type][word_type].append(extract_metric(entry, metric))
                    model_input_word_words[normalized_model][input_type][word_type].append(extract_num_words(entry))
    
    model_input_word_avgs = {}
    for model_name in model_input_word_scores:
        model_input_word_avgs[model_name] = {}
        for input_type in ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']:
            model_input_word_avgs[model_name][input_type] = {}
            for word_type in ['constructed', 'rare', 'common']:
                if input_type in model_input_word_scores[model_name] and word_type in model_input_word_scores[model_name][input_type]:
                    scores = model_input_word_scores[model_name][input_type][word_type]
                    weights = model_input_word_words[model_name][input_type][word_type]
                    if scores:
                        avg = weighted_avg(scores, weights)
                        model_input_word_avgs[model_name][input_type][word_type] = avg
    
    if not model_input_word_avgs:
        return
    
    input_types = ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']
    word_types = ['common', 'rare', 'constructed']
    models = list(model_input_word_avgs.keys())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    word_type_colors = ['#ffd43b', '#20c997', '#f06595']
    model_markers = ['o', 's', '^', 'D', '*', 'v', '<', '>', 'p', 'h']
    
    # Calculate x positions for each input type with word type offsets
    x_positions = {}
    word_type_offset = 0.2
    
    for i, input_type in enumerate(input_types):
        x_positions[input_type] = {}
        for j, word_type in enumerate(word_types):
            # Base position for input type + offset for word type
            x_positions[input_type][word_type] = i + (j - 1) * word_type_offset
    
    for j, model_name in enumerate(models):
        for i, word_type in enumerate(word_types):
            x_coords = []
            y_coords = []
            for input_type in input_types:
                if (model_name in model_input_word_avgs and 
                    input_type in model_input_word_avgs[model_name] and 
                    word_type in model_input_word_avgs[model_name][input_type]):
                    x_coords.append(x_positions[input_type][word_type])
                    y_coords.append(model_input_word_avgs[model_name][input_type][word_type])
            
            if x_coords:
                label = f"{model_name}-{word_type}" if i == 0 else None
                ax.scatter(x_coords, y_coords, c=word_type_colors[i], marker=model_markers[j % len(model_markers)], s=50, 
                          label=label, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Input Types', fontsize=13)
    ax.set_ylabel(get_metric_label(metric), fontsize=13)
    ax.set_xticks(range(len(input_types)))
    ax.set_xticklabels(input_types, ha='center')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.grid(True, alpha=0.3)
    
    legend_elements = []
    for i, word_type in enumerate(word_types):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=word_type_colors[i], 
                                        markersize=8, label=word_type))
    for j, model_name in enumerate(models):
        legend_elements.append(plt.Line2D([0], [0], marker=model_markers[j % len(model_markers)], color='w', markerfacecolor='gray', 
                                        markersize=8, label=model_name))
    
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
    
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    file_name = f"inputtype_wordtype_modeltype_scatter_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Input type vs word type vs model type scatter plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def plot_inputtype_wordtype_modeltype_semdim_scatter(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims):
    """Plot scatter plot for each individual semantic dimension"""
    from collections import defaultdict
    
    def normalize_model_name(model_name):
        if 'gpt' in model_name.lower():
            return 'GPT'
        return model_name
    
    # Extract data by semantic dimension, model, input type, and word type
    semdim_model_input_word_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    semdim_model_input_word_words = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for model_name, model_data in data.items():
        normalized_model = normalize_model_name(model_name)
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                            continue
                    
                    semdim_model_input_word_scores[sem_dim][normalized_model][input_type][word_type].append(extract_metric(entry, metric))
                    semdim_model_input_word_words[sem_dim][normalized_model][input_type][word_type].append(extract_num_words(entry))
    
    # Calculate averages for each semantic dimension
    semdim_model_input_word_avgs = {}
    for sem_dim in semdim_model_input_word_scores:
        semdim_model_input_word_avgs[sem_dim] = {}
        for model_name in semdim_model_input_word_scores[sem_dim]:
            semdim_model_input_word_avgs[sem_dim][model_name] = {}
            for input_type in ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']:
                semdim_model_input_word_avgs[sem_dim][model_name][input_type] = {}
                for word_type in ['constructed', 'rare', 'common']:
                    if (input_type in semdim_model_input_word_scores[sem_dim][model_name] and 
                        word_type in semdim_model_input_word_scores[sem_dim][model_name][input_type]):
                        scores = semdim_model_input_word_scores[sem_dim][model_name][input_type][word_type]
                        weights = semdim_model_input_word_words[sem_dim][model_name][input_type][word_type]
                        if scores:
                            avg = weighted_avg(scores, weights)
                            semdim_model_input_word_avgs[sem_dim][model_name][input_type][word_type] = avg
    
    # Create plot for each semantic dimension
    for sem_dim in semdim_model_input_word_avgs:
        if not semdim_model_input_word_avgs[sem_dim]:
            continue
            
        input_types = ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']
        word_types = ['common', 'rare', 'constructed']
        models = list(semdim_model_input_word_avgs[sem_dim].keys())
        
        if not models:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        word_type_colors = ['#ffd43b', '#20c997', '#f06595']
        model_markers = ['o', 's', '^', 'D', '*', 'v', '<', '>', 'p', 'h']
        
        # Calculate x positions for each input type with word type offsets
        x_positions = {}
        word_type_offset = 0.2
        
        for i, input_type in enumerate(input_types):
            x_positions[input_type] = {}
            for j, word_type in enumerate(word_types):
                x_positions[input_type][word_type] = i + (j - 1) * word_type_offset
        
        for j, model_name in enumerate(models):
            for i, word_type in enumerate(word_types):
                x_coords = []
                y_coords = []
                for input_type in input_types:
                    if (model_name in semdim_model_input_word_avgs[sem_dim] and 
                        input_type in semdim_model_input_word_avgs[sem_dim][model_name] and 
                        word_type in semdim_model_input_word_avgs[sem_dim][model_name][input_type]):
                        x_coords.append(x_positions[input_type][word_type])
                        y_coords.append(semdim_model_input_word_avgs[sem_dim][model_name][input_type][word_type])
                
                if x_coords:
                    label = f"{model_name}-{word_type}" if i == 0 else None
                    ax.scatter(x_coords, y_coords, c=word_type_colors[i], marker=model_markers[j % len(model_markers)], s=50, 
                              label=label, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Input Types', fontsize=13)
        ax.set_ylabel(get_metric_label(metric), fontsize=13)
        ax.set_xticks(range(len(input_types)))
        ax.set_xticklabels(input_types, ha='center')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        
        ax.grid(True, alpha=0.3)
        
        legend_elements = []
        for i, word_type in enumerate(word_types):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=word_type_colors[i], 
                                            markersize=8, label=word_type))
        for j, model_name in enumerate(models):
            legend_elements.append(plt.Line2D([0], [0], marker=model_markers[j % len(model_markers)], color='w', markerfacecolor='gray', 
                                            markersize=8, label=model_name))
        
        ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
        ax.set_title(f'Semantic Dimension: {sem_dim}', fontsize=14, pad=15)
        
        plt.tight_layout()
        
        os.makedirs(save_path, exist_ok=True)
        suffix = "_filter" if filter_constructed else ""
        # Clean sem_dim name for filename
        clean_sem_dim = sem_dim.replace('-', '_').replace(' ', '_')
        file_name = f"inputtype_wordtype_modeltype_semdim_{clean_sem_dim}_{metric}{suffix}.png"
        plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
        print(f"Input type vs word type vs model type scatter plot for {sem_dim} saved to {os.path.join(save_path, file_name)}")
        plt.close()

def plot_inputtype_wordtype_modeltype_all_semdim_scatter(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims):
    """Plot scatter plot for all semantic dimensions in one plot, X axis: metric score, Y axis: input type + word type + model type."""
    from collections import defaultdict
    
    def normalize_model_name(model_name):
        if 'gpt' in model_name.lower():
            return 'GPT'
        return model_name
    
    # Extract data by semantic dimension, model, input type, and word type
    semdim_model_input_word_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    semdim_model_input_word_words = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for model_name, model_data in data.items():
        normalized_model = normalize_model_name(model_name)
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                        continue
                    
                    semdim_model_input_word_scores[sem_dim][normalized_model][input_type][word_type].append(extract_metric(entry, metric))
                    semdim_model_input_word_words[sem_dim][normalized_model][input_type][word_type].append(extract_num_words(entry))
    
    # Calculate averages for each semantic dimension
    semdim_model_input_word_avgs = {}
    for sem_dim in semdim_model_input_word_scores:
        semdim_model_input_word_avgs[sem_dim] = {}
        for model_name in semdim_model_input_word_scores[sem_dim]:
            semdim_model_input_word_avgs[sem_dim][model_name] = {}
            for input_type in ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']:
                semdim_model_input_word_avgs[sem_dim][model_name][input_type] = {}
                for word_type in ['common', 'rare', 'constructed']:
                    if (input_type in semdim_model_input_word_scores[sem_dim][model_name] and 
                        word_type in semdim_model_input_word_scores[sem_dim][model_name][input_type]):
                        scores = semdim_model_input_word_scores[sem_dim][model_name][input_type][word_type]
                        weights = semdim_model_input_word_words[sem_dim][model_name][input_type][word_type]
                        if scores:
                            avg = weighted_avg(scores, weights)
                            semdim_model_input_word_avgs[sem_dim][model_name][input_type][word_type] = avg
    
    if not semdim_model_input_word_avgs:
        return
    
    input_types = ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']
    word_types = ['common', 'rare', 'constructed']
    models = list(set(model for semdim_data in semdim_model_input_word_avgs.values() for model in semdim_data.keys()))
    sem_dims = list(semdim_model_input_word_avgs.keys())
    
    fig, ax = plt.subplots(figsize=(12, 12))  # Increased height for more Y-axis categories
    
    # Colors for word types, markers for models
    word_type_colors = ['#ffd43b', '#20c997', '#f06595']  # common, rare, constructed
    model_markers = ['o', 's', '^', 'D', '*', 'v', '<', '>', 'p', 'h']
    
    # Calculate y positions for each combination: input_type + word_type + model_type
    y_positions = {}
    current_y = 0
    y_labels = []
    y_ticks = []
    
    for i, input_type in enumerate(input_types):
        for j, word_type in enumerate(word_types):
            for k, model_name in enumerate(models):
                y_positions[(input_type, word_type, model_name)] = current_y
                y_labels.append(f"{input_type}-{word_type}-{model_name}")
                y_ticks.append(current_y)
                current_y += 1
            current_y += 0.5  # Add space between word types
        current_y += 1  # Add space between input types
    
    # Plot each combination
    for i, word_type in enumerate(word_types):
        for j, model_name in enumerate(models):
            x_coords = []
            y_coords = []
            for input_type in input_types:
                for sem_dim in sem_dims:
                    if (sem_dim in semdim_model_input_word_avgs and
                        model_name in semdim_model_input_word_avgs[sem_dim] and
                        input_type in semdim_model_input_word_avgs[sem_dim][model_name] and 
                        word_type in semdim_model_input_word_avgs[sem_dim][model_name][input_type]):
                        x_coords.append(semdim_model_input_word_avgs[sem_dim][model_name][input_type][word_type])
                        y_coords.append(y_positions[(input_type, word_type, model_name)])
            if x_coords:
                color = word_type_colors[i]
                marker = model_markers[j % len(model_markers)]
                label = f"{model_name}-{word_type}" if i == 0 else None
                ax.scatter(x_coords, y_coords, c=color, marker=marker, s=20, alpha=0.8, edgecolors='black', linewidth=0.5, label=label)
    
    ax.set_xlabel(get_metric_label(metric), fontsize=13)
    ax.set_ylabel('Input Type - Word Type - Model Type', fontsize=13)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, va='center', fontsize=9)
    
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.grid(True, alpha=0.3, axis='x')
    
    # Create legend elements
    legend_elements = []
    for i, word_type in enumerate(word_types):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=word_type_colors[i], 
                                        markersize=8, label=word_type))
    for j, model_name in enumerate(models):
        legend_elements.append(plt.Line2D([0], [0], marker=model_markers[j % len(model_markers)], color='w', markerfacecolor='gray', 
                                        markersize=8, label=model_name))
    
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
    
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    file_name = f"inputtype_wordtype_modeltype_all_semdim_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Input type vs word type vs model type vs all semantic dimensions scatter plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def plot_inputtype_modeltype_wordtype_all_semdim_scatter(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims):
    """Plot scatter plot for all semantic dimensions in one plot, X axis: metric score, Y axis: input type + model type + word type."""
    from collections import defaultdict
    
    def normalize_model_name(model_name):
        if 'gpt' in model_name.lower():
            return 'GPT'
        return model_name
    
    # Extract data by semantic dimension, model, input type, and word type
    semdim_model_input_word_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    semdim_model_input_word_words = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for model_name, model_data in data.items():
        normalized_model = normalize_model_name(model_name)
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                        continue
                    
                    semdim_model_input_word_scores[sem_dim][normalized_model][input_type][word_type].append(extract_metric(entry, metric))
                    semdim_model_input_word_words[sem_dim][normalized_model][input_type][word_type].append(extract_num_words(entry))
    
    # Calculate averages for each semantic dimension
    semdim_model_input_word_avgs = {}
    for sem_dim in semdim_model_input_word_scores:
        semdim_model_input_word_avgs[sem_dim] = {}
        for model_name in semdim_model_input_word_scores[sem_dim]:
            semdim_model_input_word_avgs[sem_dim][model_name] = {}
            for input_type in ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']:
                semdim_model_input_word_avgs[sem_dim][model_name][input_type] = {}
                for word_type in ['common', 'rare', 'constructed']:
                    if (input_type in semdim_model_input_word_scores[sem_dim][model_name] and 
                        word_type in semdim_model_input_word_scores[sem_dim][model_name][input_type]):
                        scores = semdim_model_input_word_scores[sem_dim][model_name][input_type][word_type]
                        weights = semdim_model_input_word_words[sem_dim][model_name][input_type][word_type]
                        if scores:
                            avg = weighted_avg(scores, weights)
                            semdim_model_input_word_avgs[sem_dim][model_name][input_type][word_type] = avg
    
    if not semdim_model_input_word_avgs:
        return
    
    input_types = ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']
    word_types = ['common', 'rare', 'constructed']
    models = list(set(model for semdim_data in semdim_model_input_word_avgs.values() for model in semdim_data.keys()))
    sem_dims = list(semdim_model_input_word_avgs.keys())
    
    fig, ax = plt.subplots(figsize=(12, 12))  # Increased height for more Y-axis categories
    
    # Colors for word types, markers for models
    word_type_colors = ['#ffd43b', '#20c997', '#f06595']  # common, rare, constructed
    model_markers = ['o', 's', '^', 'D', '*', 'v', '<', '>', 'p', 'h']
    
    # Calculate y positions for each combination: input_type + model_type + word_type
    y_positions = {}
    current_y = 0
    y_labels = []
    y_ticks = []
    
    for i, input_type in enumerate(input_types):
        for j, model_name in enumerate(models):
            for k, word_type in enumerate(word_types):
                y_positions[(input_type, model_name, word_type)] = current_y
                y_labels.append(f"{input_type}-{model_name}-{word_type}")
                y_ticks.append(current_y)
                current_y += 1
            current_y += 0.5  # Add space between model types
        current_y += 1  # Add space between input types
    
    # Plot each combination
    for i, word_type in enumerate(word_types):
        for j, model_name in enumerate(models):
            x_coords = []
            y_coords = []
            for input_type in input_types:
                for sem_dim in sem_dims:
                    if (sem_dim in semdim_model_input_word_avgs and
                        model_name in semdim_model_input_word_avgs[sem_dim] and
                        input_type in semdim_model_input_word_avgs[sem_dim][model_name] and 
                        word_type in semdim_model_input_word_avgs[sem_dim][model_name][input_type]):
                        x_coords.append(semdim_model_input_word_avgs[sem_dim][model_name][input_type][word_type])
                        y_coords.append(y_positions[(input_type, model_name, word_type)])
            if x_coords:
                color = word_type_colors[i]
                marker = model_markers[j % len(model_markers)]
                label = f"{model_name}-{word_type}" if i == 0 else None
                ax.scatter(x_coords, y_coords, c=color, marker=marker, s=20, alpha=0.8, edgecolors='black', linewidth=0.5, label=label)
    
    ax.set_xlabel(get_metric_label(metric), fontsize=13)
    ax.set_ylabel('Input Type - Model Type - Word Type', fontsize=13)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, va='center', fontsize=9)
    
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.grid(True, alpha=0.3, axis='x')
    
    # Create legend elements
    legend_elements = []
    for i, word_type in enumerate(word_types):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=word_type_colors[i], 
                                        markersize=8, label=word_type))
    for j, model_name in enumerate(models):
        legend_elements.append(plt.Line2D([0], [0], marker=model_markers[j % len(model_markers)], color='w', markerfacecolor='gray', 
                                        markersize=8, label=model_name))
    
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
    
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    file_name = f"inputtype_modeltype_wordtype_all_semdim_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Input type vs model type vs word type vs all semantic dimensions scatter plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def plot_inputtype_per_wordtype_all_models(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims):
    from collections import defaultdict
    input_types = ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']
    word_types = ['common', 'rare', 'constructed']
    inputtype_wordtype_scores = defaultdict(lambda: defaultdict(list))
    inputtype_wordtype_words = defaultdict(lambda: defaultdict(list))
    for model_name, model_data in data.items():
        for word_type, word_data in model_data.items():
            if word_type not in word_types:
                continue
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed:
                        if sem_dim not in constructed_dims:
                            continue
                    inputtype_wordtype_scores[word_type][input_type].append(extract_metric(entry, metric))
                    inputtype_wordtype_words[word_type][input_type].append(extract_num_words(entry))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = ['#ffd43b', '#20c997', '#f06595']
    for idx, word_type in enumerate(word_types):
        avgs = []
        for input_type in input_types:
            scores = inputtype_wordtype_scores[word_type][input_type]
            weights = inputtype_wordtype_words[word_type][input_type]
            if scores:
                avg = weighted_avg(scores, weights)
                avgs.append(avg)
            else:
                avgs.append(None)
        ax = axes[idx]
        x = [i for i, v in enumerate(avgs) if v is not None]
        y = [v for v in avgs if v is not None]
        ax.scatter(x, y, color=colors[idx], s=60, edgecolors='black', linewidth=1)
        ax.set_title(word_type)
        ax.set_xticks(range(len(input_types)))
        ax.set_xticklabels(input_types, rotation=30, ha='right')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.axhline(y=0.35, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
        ax.axhline(y=0.65, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(get_metric_label(metric))
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    file_name = f"inputtype_per_wordtype_all_models_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Input type per word type (all models) plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def plot_high_performance_semantic_dimensions(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims):
    """Plot semantic dimensions where both rare and constructed word types have macro F1 >= 0.55"""
    wordtype_scores, wordtype_words = extract_wordtype_data(data, metric, filter_constructed, constructed_dims)
    
    high_performance_dims = []
    for sem_dim in wordtype_scores:
        rare_scores = wordtype_scores[sem_dim].get('rare', [])
        constructed_scores = wordtype_scores[sem_dim].get('constructed', [])
        rare_weights = wordtype_words[sem_dim].get('rare', [])
        constructed_weights = wordtype_words[sem_dim].get('constructed', [])
        
        if rare_scores and constructed_scores:
            rare_avg = weighted_avg(rare_scores, rare_weights)
            constructed_avg = weighted_avg(constructed_scores, constructed_weights)
            
            if rare_avg >= 0.55 and constructed_avg >= 0.55:
                high_performance_dims.append((sem_dim, rare_avg, constructed_avg))
    
    if not high_performance_dims:
        print("No semantic dimensions found with both rare and constructed F1 >= 0.55")
        return
    
    high_performance_dims.sort(key=lambda x: (x[1] + x[2]) / 2, reverse=True)
    
    sem_dims = [item[0] for item in high_performance_dims]
    rare_scores = [item[1] for item in high_performance_dims]
    constructed_scores = [item[2] for item in high_performance_dims]
    
    x = np.arange(len(sem_dims))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(sem_dims) * 0.4)))
    
    bars1 = ax.barh(x - width/2, rare_scores, width, label='Rare', color='#ffd43b', edgecolor='black')
    bars2 = ax.barh(x + width/2, constructed_scores, width, label='Constructed', color='#20c997', edgecolor='black')
    
    ax.set_yticks(x)
    ax.set_yticklabels(sem_dims, fontsize=12)
    ax.set_xlabel(get_metric_label(metric), fontsize=13)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    ax.axvline(x=0.55, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Threshold (0.55)')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(x=0.35, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
    ax.axvline(x=0.65, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    for i, (rare_score, constructed_score) in enumerate(zip(rare_scores, constructed_scores)):
        ax.text(rare_score + 0.01, i - width/2, f'{rare_score:.3f}', ha='left', va='center', fontsize=10)
        ax.text(constructed_score + 0.01, i + width/2, f'{constructed_score:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    suffix = "_filter" if filter_constructed else ""
    file_name = f"high_performance_semantic_dimensions_{metric}{suffix}.png"
    save_plot(fig, save_path, file_name)

def plot_inference_results_by_input_model_word(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims):
    """Scatter plot showing inference results by input type, model type, and word type"""
    def normalize_model_name(model_name):
        if 'gpt' in model_name.lower():
            return 'GPT'
        return model_name
    
    from collections import defaultdict
    input_model_word_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    input_model_word_words = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for model_name, model_data in data.items():
        normalized_model = normalize_model_name(model_name)
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                        continue
                    input_model_word_scores[input_type][normalized_model][word_type].append(extract_metric(entry, metric))
                    input_model_word_words[input_type][normalized_model][word_type].append(extract_num_words(entry))
    
    input_model_word_avgs = {}
    for input_type in ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']:
        input_model_word_avgs[input_type] = {}
        for model_name in set(normalize_model_name(m) for m in data.keys()):
            input_model_word_avgs[input_type][model_name] = {}
            for word_type in ['constructed', 'rare', 'common']:
                if (input_type in input_model_word_scores and 
                    model_name in input_model_word_scores[input_type] and 
                    word_type in input_model_word_scores[input_type][model_name]):
                    scores = input_model_word_scores[input_type][model_name][word_type]
                    weights = input_model_word_words[input_type][model_name][word_type]
                    if scores:
                        avg = weighted_avg(scores, weights)
                        input_model_word_avgs[input_type][model_name][word_type] = avg
    
    if not input_model_word_avgs:
        return
    
    input_types = ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']
    models = list(set(normalize_model_name(m) for m in data.keys()))
    word_types = ['constructed', 'rare', 'common']
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = ['#ffd43b', '#20c997', '#f06595', '#845ef7', '#fd7e14']
    markers = ['o', 's', '^', 'D', '*']
    
    x_positions = []
    y_values = []
    colors_plot = []
    markers_plot = []
    labels = []
    
    current_x = 0
    separator_positions = []
    
    for i, input_type in enumerate(input_types):
        input_start = current_x
        
        for j, model_name in enumerate(models):
            for k, word_type in enumerate(word_types):
                if (input_type in input_model_word_avgs and 
                    model_name in input_model_word_avgs[input_type] and 
                    word_type in input_model_word_avgs[input_type][model_name]):
                    
                    x_positions.append(current_x)
                    y_values.append(input_model_word_avgs[input_type][model_name][word_type])
                    colors_plot.append(colors[j % len(colors)])
                    markers_plot.append(markers[k % len(markers)])
                    labels.append(f"{input_type}-{model_name}-{word_type}")
                    current_x += 1
        
        separator_positions.append((input_start + current_x - 1) / 2)
        current_x += 2
    
    for i, (x, y, color, marker) in enumerate(zip(x_positions, y_values, colors_plot, markers_plot)):
        ax.scatter(x, y, c=color, marker=marker, s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    for sep_pos in separator_positions:
        ax.axvline(x=sep_pos, color='gray', linestyle='-', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Input Type - Model - Word Type', fontsize=13)
    ax.set_ylabel(get_metric_label(metric), fontsize=13)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

    
    ax.grid(True, alpha=0.3)
    
    legend_elements = []
    for j, model_name in enumerate(models):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[j % len(colors)], 
                                        markersize=8, label=f'Model: {model_name}'))
    for k, word_type in enumerate(word_types):
        legend_elements.append(plt.Line2D([0], [0], marker=markers[k % len(markers)], color='w', markerfacecolor='gray', 
                                        markersize=8, label=f'Word: {word_type}'))
    
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
    
    plt.tight_layout()
    
    suffix = "_filter" if filter_constructed else ""
    file_name = f"inference_results_by_input_model_word_{metric}{suffix}.png"
    save_plot(fig, save_path, file_name)

def plot_inference_results_by_semantic_input_model_word(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims):
    """Scatter plot showing inference results by semantic dimension, input type, model type, and word type"""
    def normalize_model_name(model_name):
        if 'gpt' in model_name.lower():
            return 'GPT'
        return model_name
    
    from collections import defaultdict
    semdim_input_model_word_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    semdim_input_model_word_words = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for model_name, model_data in data.items():
        normalized_model = normalize_model_name(model_name)
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                        continue
                    semdim_input_model_word_scores[sem_dim][input_type][normalized_model][word_type].append(extract_metric(entry, metric))
                    semdim_input_model_word_words[sem_dim][input_type][normalized_model][word_type].append(extract_num_words(entry))
    
    semdim_input_model_word_avgs = {}
    for sem_dim in semdim_input_model_word_scores:
        semdim_input_model_word_avgs[sem_dim] = {}
        for input_type in ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']:
            semdim_input_model_word_avgs[sem_dim][input_type] = {}
            for model_name in set(normalize_model_name(m) for m in data.keys()):
                semdim_input_model_word_avgs[sem_dim][input_type][model_name] = {}
                for word_type in ['constructed', 'rare', 'common']:
                    if (input_type in semdim_input_model_word_scores[sem_dim] and 
                        model_name in semdim_input_model_word_scores[sem_dim][input_type] and 
                        word_type in semdim_input_model_word_scores[sem_dim][input_type][model_name]):
                        scores = semdim_input_model_word_scores[sem_dim][input_type][model_name][word_type]
                        weights = semdim_input_model_word_words[sem_dim][input_type][model_name][word_type]
                        if scores:
                            avg = weighted_avg(scores, weights)
                            semdim_input_model_word_avgs[sem_dim][input_type][model_name][word_type] = avg
    
    if not semdim_input_model_word_avgs:
        return
    
    input_types = ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']
    models = list(set(normalize_model_name(m) for m in data.keys()))
    word_types = ['constructed', 'rare', 'common']
    sem_dims = list(semdim_input_model_word_avgs.keys())
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(sem_dims)))
    markers = ['o', 's', '^', 'D', '*']
    
    x_positions = []
    y_values = []
    colors_plot = []
    markers_plot = []
    labels = []
    
    current_x = 0
    separator_positions = []
    
    for i, input_type in enumerate(input_types):
        input_start = current_x
        
        for j, model_name in enumerate(models):
            for k, word_type in enumerate(word_types):
                for l, sem_dim in enumerate(sem_dims):
                    if (input_type in semdim_input_model_word_avgs[sem_dim] and 
                        model_name in semdim_input_model_word_avgs[sem_dim][input_type] and 
                        word_type in semdim_input_model_word_avgs[sem_dim][input_type][model_name]):
                        
                        x_positions.append(current_x)
                        y_values.append(semdim_input_model_word_avgs[sem_dim][input_type][model_name][word_type])
                        colors_plot.append(colors[l % len(colors)])
                        markers_plot.append(markers[k % len(markers)])
                        labels.append(f"{sem_dim}-{input_type}-{model_name}-{word_type}")
                        current_x += 1
        
        separator_positions.append((input_start + current_x - 1) / 2)
        current_x += 2
    
    for i, (x, y, color, marker) in enumerate(zip(x_positions, y_values, colors_plot, markers_plot)):
        ax.scatter(x, y, c=[color], marker=marker, s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    for sep_pos in separator_positions:
        ax.axvline(x=sep_pos, color='gray', linestyle='-', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Semantic Dimension - Input Type - Model - Word Type', fontsize=13)
    ax.set_ylabel(get_metric_label(metric), fontsize=13)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

    
    ax.grid(True, alpha=0.3)
    
    legend_elements = []
    for l, sem_dim in enumerate(sem_dims):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[l % len(colors)], 
                                        markersize=8, label=f'SemDim: {sem_dim}'))
    for k, word_type in enumerate(word_types):
        legend_elements.append(plt.Line2D([0], [0], marker=markers[k % len(markers)], color='w', markerfacecolor='gray', 
                                        markersize=8, label=f'Word: {word_type}'))
    
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right', ncol=2)
    
    plt.tight_layout()
    
    suffix = "_filter" if filter_constructed else ""
    file_name = f"inference_results_by_semantic_input_model_word_{metric}{suffix}.png"
    save_plot(fig, save_path, file_name)

def plot_grouped_horizontal_bar(
    avg_dict, group, metric='f1', title=None, save_path=None, 
    top_n=3, bottom_n=3, sem_dims=None, filter_constructed=False
):
    """
    avg_dict: {(group, sem_dim): (avg, total_words)}
    group: tuple, e.g. ('Qwen2.5-Omni-3B', 'original_and_audio')
    """
    # 1. 해당 group의 semantic dimension만 추출
    semdim_avgs = {}
    for (g, sem_dim), (avg, total_words) in avg_dict.items():
        if g == group:
            if sem_dims is None or sem_dim in sem_dims:
                semdim_avgs[sem_dim] = avg
    if not semdim_avgs:
        print(f"[WARN] No data for group {group}")
        return

    # 2. 내림차순 정렬
    sorted_items = sorted(semdim_avgs.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_items)
    # top_n, bottom_n 처리
    if top_n is not None and bottom_n is not None and (top_n + bottom_n) < n:
        selected = sorted_items[:top_n] + sorted_items[-bottom_n:]
    elif top_n is not None and top_n < n:
        selected = sorted_items[:top_n]
    elif bottom_n is not None and bottom_n < n:
        selected = sorted_items[-bottom_n:]
    else:
        selected = sorted_items

    sem_dims_plot, avgs = zip(*selected)
    
    # Use #5c940d (green) and #e03131 (red) as base colors, keep interpolation logic
    from matplotlib.colors import to_rgb

    green_rgb = to_rgb("#5c940d")
    red_rgb = to_rgb("#e03131")
    white_rgb = (1, 1, 1)

    colors = []
    for v in avgs:
        if v > 0.5:
            # Interpolate between white and green, as before
            alpha = min(1.0, (v - 0.5) * 2)
            color = tuple((1 - alpha) * w + alpha * g for w, g in zip(white_rgb, green_rgb))
            colors.append(color)
        elif v < 0.5:
            # Interpolate between white and red, as before
            alpha = min(1.0, (0.5 - v) * 2)
            color = tuple((1 - alpha) * w + alpha * r for w, r in zip(white_rgb, red_rgb))
            colors.append(color)
        else:
            colors.append(white_rgb)

    fig, ax = plt.subplots(figsize=(8, max(6, len(sem_dims_plot)*0.4)))
    bars = ax.barh(range(len(avgs)), avgs, color=colors, edgecolor='black')
    ax.set_yticks(range(len(avgs)))
    ax.set_yticklabels(sem_dims_plot, fontsize=12, va='center')
    ax.set_xlabel(get_metric_label(metric), fontsize=13)
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax.invert_yaxis()  # 내림차순 정렬이 위에서 아래로 보이게
    if title:
        ax.set_title(title, fontsize=15, pad=15)
    plt.tight_layout()
    
    group_str = '_'.join(str(g) for g in group)
    full_save_path = os.path.join(save_path, metric, group_str)
    os.makedirs(full_save_path, exist_ok=True)
    suffix = ""
    if top_n is not None or bottom_n is not None:
        suffix = f"_top{top_n or 0}_bottom{bottom_n or 0}"
    file_name = f"{len(sem_dims_plot)}_dims{suffix}.png"
    save_plot(fig, full_save_path, file_name)

def plot_wordtype_semantic_dimension_bars(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims):
    """Plot horizontal bar charts for each word type showing semantic dimension performance"""
    from collections import defaultdict
    import numpy as np
    
    # Extract data by word type and semantic dimension
    wordtype_semdim_scores = defaultdict(lambda: defaultdict(list))
    wordtype_semdim_words = defaultdict(lambda: defaultdict(list))
    
    for model_name, model_data in data.items():
        for word_type, word_data in model_data.items():
            if word_type not in ['common', 'rare', 'constructed']:
                continue
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                        continue
                    wordtype_semdim_scores[word_type][sem_dim].append(extract_metric(entry, metric))
                    wordtype_semdim_words[word_type][sem_dim].append(extract_num_words(entry))
    
    # Calculate averages for each word type and semantic dimension
    wordtype_semdim_avgs = {}
    for word_type in ['common', 'rare', 'constructed']:
        wordtype_semdim_avgs[word_type] = {}
        for sem_dim in wordtype_semdim_scores[word_type]:
            scores = wordtype_semdim_scores[word_type][sem_dim]
            weights = wordtype_semdim_words[word_type][sem_dim]
            if scores:
                avg = weighted_avg(scores, weights)
                wordtype_semdim_avgs[word_type][sem_dim] = avg
    
    if not wordtype_semdim_avgs:
        return
    
    word_types = ['common', 'rare', 'constructed']
    
    # 1. 각 semantic dimension별로 constructed, common, rare의 평균을 내서 정렬
    all_sem_dims = set()
    for word_type_data in wordtype_semdim_avgs.values():
        all_sem_dims.update(word_type_data.keys())
    
    semdim_avg = {}
    for sem_dim in all_sem_dims:
        vals = []
        for word_type in word_types:
            vals.append(wordtype_semdim_avgs.get(word_type, {}).get(sem_dim, 0.0))
        semdim_avg[sem_dim] = np.mean(vals)
    sorted_sem_dims = sorted(semdim_avg.items(), key=lambda x: x[1], reverse=True)
    reference_sem_dims = [item[0] for item in sorted_sem_dims]
    
    # Get top 3 semantic dimensions for bold formatting
    top3_sem_dims = set(reference_sem_dims[:3])
    
    red_rgb = (1.0, 0.42, 0.42)   # #ff6b6b
    green_rgb = (0.58, 0.85, 0.18) # #94d82d
    white_rgb = (1.0, 1.0, 1.0)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    
    for idx, word_type in enumerate(word_types):
        if word_type not in wordtype_semdim_avgs or not wordtype_semdim_avgs[word_type]:
            continue
            
        ax = axes[idx]
        semdim_data = wordtype_semdim_avgs[word_type]
        
        scores = []
        colors = []
        y_labels = []
        
        for sem_dim in reference_sem_dims:
            score = semdim_data.get(sem_dim, 0.0)
            scores.append(score)
            # 색상 interpolation
            if score < 0.5:
                alpha = min(1.0, (0.5 - score) * 2)
                color = tuple((1 - alpha) * w + alpha * r for w, r in zip(white_rgb, red_rgb))
            elif score > 0.5:
                alpha = min(1.0, (score - 0.5) * 2)
                color = tuple((1 - alpha) * w + alpha * g for w, g in zip(white_rgb, green_rgb))
            else:
                color = white_rgb
            colors.append(color)
            
            # Add bold formatting for top 3 semantic dimensions
            if sem_dim in top3_sem_dims:
                y_labels.append(f"$\\bf{{{sem_dim}}}$")
            else:
                y_labels.append(sem_dim)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(scores)), scores, color=colors, edgecolor='black')
        
        # Set labels and formatting
        ax.set_yticks(range(len(scores)))
        ax.set_yticklabels(y_labels, fontsize=10, va='center')
        ax.set_xlabel(get_metric_label(metric), fontsize=12)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        
        # Add vertical line at 0.5
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Set title
        ax.set_title(f'{word_type.capitalize()} Words', fontsize=14, pad=15)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Invert y-axis for descending order (highest average at top)
        ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    file_name = f"wordtype_semantic_dimension_bars_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Word type semantic dimension bars plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def plot_semantic_dimension_variance_by_wordtype(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims):
    """Plot semantic dimensions sorted by variance (max-min difference) across word types"""
    from collections import defaultdict
    
    # Extract data by word type and semantic dimension
    wordtype_semdim_scores = defaultdict(lambda: defaultdict(list))
    wordtype_semdim_words = defaultdict(lambda: defaultdict(list))
    
    for model_name, model_data in data.items():
        for word_type, word_data in model_data.items():
            if word_type not in ['common', 'rare', 'constructed']:
                continue
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                        continue
                    wordtype_semdim_scores[word_type][sem_dim].append(extract_metric(entry, metric))
                    wordtype_semdim_words[word_type][sem_dim].append(extract_num_words(entry))
    
    # Calculate averages for each word type and semantic dimension
    wordtype_semdim_avgs = {}
    for word_type in ['common', 'rare', 'constructed']:
        wordtype_semdim_avgs[word_type] = {}
        for sem_dim in wordtype_semdim_scores[word_type]:
            scores = wordtype_semdim_scores[word_type][sem_dim]
            weights = wordtype_semdim_words[word_type][sem_dim]
            if scores:
                avg = weighted_avg(scores, weights)
                wordtype_semdim_avgs[word_type][sem_dim] = avg
    
    if not wordtype_semdim_avgs:
        return
    
    # Calculate variance (max-min difference) for each semantic dimension across word types
    semdim_variance = {}
    semdim_wordtype_scores = {}
    
    # Get all semantic dimensions that appear in at least one word type
    all_sem_dims = set()
    for word_type_data in wordtype_semdim_avgs.values():
        all_sem_dims.update(word_type_data.keys())
    
    for sem_dim in all_sem_dims:
        scores = []
        for word_type in ['common', 'rare', 'constructed']:
            if sem_dim in wordtype_semdim_avgs[word_type]:
                scores.append(wordtype_semdim_avgs[word_type][sem_dim])
            else:
                scores.append(0.0)  # Default value for missing data
        
        if len(scores) >= 2:  # Need at least 2 scores to calculate variance
            variance = max(scores) - min(scores)
            semdim_variance[sem_dim] = variance
            semdim_wordtype_scores[sem_dim] = scores
    
    if not semdim_variance:
        return
    
    # Sort semantic dimensions by variance (ascending order: small to large difference)
    sorted_sem_dims = sorted(semdim_variance.items(), key=lambda x: x[1])
    sem_dims_ordered = [item[0] for item in sorted_sem_dims]
    variances = [item[1] for item in sorted_sem_dims]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, max(12, len(sem_dims_ordered) * 0.6)))
    
    word_types = ['common', 'rare', 'constructed']
    colors = ['#ffd43b', '#20c997', '#f06595']  # Yellow, Teal, Pink
    
    # Create horizontal bar plot
    x_positions = np.arange(len(sem_dims_ordered))
    width = 0.25
    
    for i, word_type in enumerate(word_types):
        scores = []
        for sem_dim in sem_dims_ordered:
            if sem_dim in semdim_wordtype_scores:
                scores.append(semdim_wordtype_scores[sem_dim][i])
            else:
                scores.append(0.0)
        
        bars = ax.barh(x_positions + i * width, scores, width, 
                      label=word_type.capitalize(), color=colors[i], 
                      edgecolor='black', alpha=0.8)
    
    # Add variance values as text on the right side
    for i, (sem_dim, variance) in enumerate(zip(sem_dims_ordered, variances)):
        ax.text(1.005, i + width, f'{variance:.3f}', ha='left', va='center', 
                fontsize=10, transform=ax.get_xaxis_transform())
    
    # Customize the plot
    ax.set_yticks(x_positions + width)
    ax.set_yticklabels(sem_dims_ordered, fontsize=11, va='center')
    ax.set_xlabel(get_metric_label(metric), fontsize=13)
    ax.set_xlim(0, 1.08)  # Extra space for variance text
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    # Add vertical line at 0.5
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Add title and legend
    ax.set_title('Semantic Dimensions by Variance Across Word Types\n(Sorted by Max-Min Difference: Small to Large)', 
                 fontsize=14, pad=15)
    ax.legend(loc='upper right', fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Invert y-axis for ascending order (smallest variance at top)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    file_name = f"semantic_dimension_variance_by_wordtype_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Semantic dimension variance by word type plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def plot_inputtype_semdim_scatter(data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims, line=False):
    """
    Plot scatter plot: X-axis = input types, Y-axis = metric score, one line per semantic dimension (color-coded).
    If line=True, connect points with lines. If line=False (default), only plot scatter points.
    Model type과 word type은 구분 없이 평균을 내고, semantic dimension별로 색상 구분.
    """
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    input_types = ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']
    # Collect scores: sem_dim -> input_type -> [scores]
    semdim_inputtype_scores = defaultdict(lambda: defaultdict(list))
    semdim_inputtype_words = defaultdict(lambda: defaultdict(list))

    for model_name, model_data in data.items():
        for word_type, word_data in model_data.items():
            for input_type, input_data in word_data.items():
                if "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                        continue
                    semdim_inputtype_scores[sem_dim][input_type].append(extract_metric(entry, metric))
                    semdim_inputtype_words[sem_dim][input_type].append(extract_num_words(entry))

    # Calculate averages for each semantic dimension and input type
    semdim_inputtype_avgs = {}
    for sem_dim in semdim_inputtype_scores:
        semdim_inputtype_avgs[sem_dim] = []
        for input_type in input_types:
            scores = semdim_inputtype_scores[sem_dim][input_type]
            weights = semdim_inputtype_words[sem_dim][input_type]
            if scores:
                avg = weighted_avg(scores, weights)
            else:
                avg = np.nan
            semdim_inputtype_avgs[sem_dim].append(avg)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = plt.cm.get_cmap('tab20', len(semdim_inputtype_avgs))
    for idx, (sem_dim, avgs) in enumerate(semdim_inputtype_avgs.items()):
        if line:
            ax.plot(input_types, avgs, marker='o', label=sem_dim, color=colors(idx))
        else:
            ax.scatter(input_types, avgs, label=sem_dim, color=colors(idx), s=30, alpha=0.8)

    ax.set_xlabel('Input Types', fontsize=13)
    ax.set_ylabel(get_metric_label(metric), fontsize=13)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    file_name = f"inputtype_semdim_scatter_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Input type vs semantic dimension scatter plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def plot_inputtype_wordtype_semdim_scatter(
    data, metric='macro_f1_score', save_path=None, filter_constructed=False, constructed_dims=constructed_dims, line=False
):
    """
    Plot scatter: X-axis = input type (text only), Y-axis = metric score.
    Word type별 색상, model type별 마커, semantic dimension별 점 모두 표시.
    X축: input type만 표기, rotation=0, 마지막에 여백.
    input_type별로 3개의 word_type이 가까이 군집되고, input_type 그룹들끼리는 떨어져 보이도록 X좌표를 배치.
    """
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    def normalize_model_name(model_name):
        if 'gpt' in model_name.lower():
            return 'GPT'
        return model_name

    input_types = ['original', 'original_and_audio', 'ipa', 'ipa_and_audio', 'audio']
    word_types = ['common', 'rare', 'constructed']
    models = sorted(set(normalize_model_name(m) for m in data.keys()))
    sem_dims = set()
    for model_data in data.values():
        for word_data in model_data.values():
            for input_data in word_data.values():
                sem_dims.update(input_data.get('dimensions', {}).keys())
    sem_dims = sorted(list(sem_dims))

    # X축 배치: input_type별로 3개씩 가까이, 그룹 간 간격은 넓게
    group_gap = 2.0  # input_type 그룹 간 간격
    inner_gap = 0.3  # word_type 그룹 내 간격
    x_map = {}
    x_labels = []
    x_tick_pos = []
    for i, input_type in enumerate(input_types):
        base = i * group_gap
        for j, word_type in enumerate(word_types):
            x = base + (j - 1) * inner_gap  # -0.3, 0, 0.3
            x_map[(input_type, word_type)] = x
        x_labels.append(input_type)
        x_tick_pos.append(base)

    word_type_colors = {'common': '#ffd43b', 'rare': '#20c997', 'constructed': '#f06595'}
    model_markers = ['o', 's', '^', 'D', '*', 'v', '<', '>', 'p', 'h']
    model_marker_map = {m: model_markers[i % len(model_markers)] for i, m in enumerate(models)}

    x_coords = []
    y_coords = []
    colors = []
    markers = []

    for model_name, model_data in data.items():
        normalized_model = normalize_model_name(model_name)
        marker = model_marker_map[normalized_model]
        for word_type, word_data in model_data.items():
            if word_type not in word_types:
                continue
            color = word_type_colors[word_type]
            for input_type, input_data in word_data.items():
                if input_type not in input_types or "romanized" in input_type:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                        continue
                    score = extract_metric(entry, metric)
                    x = x_map[(input_type, word_type)]
                    x_coords.append(x)
                    y_coords.append(score)
                    colors.append(color)
                    markers.append(marker)

    fig, ax = plt.subplots(figsize=(max(10, len(x_labels)*1.2), 10))
    
    # Add background shading for specific input types (original, ipa, audio)
    highlight_input_types = ['original', 'ipa', 'audio']
    for i, input_type in enumerate(input_types):
        if input_type in highlight_input_types:
            base = i * group_gap
            # Add light gray background rectangle for the entire input type group
            rect = plt.Rectangle((base - group_gap/2, 0), group_gap, 1, 
                               facecolor='lightgray', alpha=0.3, edgecolor='none', zorder=0)
            ax.add_patch(rect)
    
    for m in set(markers):
        idxs = [i for i, mk in enumerate(markers) if mk == m]
        ax.scatter([x_coords[i] for i in idxs], [y_coords[i] for i in idxs],
                   s=30, alpha=0.3, c=[colors[i] for i in idxs], marker=m, edgecolors='black', linewidth=0.5, label=None, zorder=10)

    ax.set_xlabel('Input Type', fontsize=13)
    ax.set_ylabel(get_metric_label(metric), fontsize=13)
    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlim(-group_gap, x_tick_pos[-1] + group_gap)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.grid(True, alpha=0.3, axis='y')
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=word_type_colors[wt], markersize=8, label=wt) for wt in word_types]
    legend_elements += [Line2D([0], [0], marker=model_marker_map[m], color='w', markerfacecolor='gray', markersize=8, label=m) for m in models]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    file_name = f"inputtype_wordtype_semdim_scatter_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Input type/word type/semantic dimension scatter plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def plot_semantic_dimensions_by_input_type(
    data, input_type='ipa', metric='macro_f1_score', save_path=None, 
    filter_constructed=False, constructed_dims=constructed_dims
):
    from collections import defaultdict
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    word_types = ['constructed', 'rare', 'common']
    wordtype_semdim_scores = {wt: defaultdict(list) for wt in word_types}
    wordtype_semdim_words = {wt: defaultdict(list) for wt in word_types}

    for model_name, model_data in data.items():
        for word_type, word_data in model_data.items():
            if word_type not in word_types:
                continue
            for input_type_name, input_data in word_data.items():
                if input_type_name != input_type or "romanized" in input_type_name:
                    continue
                dimensions = input_data.get('dimensions', {})
                for sem_dim, entry in dimensions.items():
                    if filter_constructed and sem_dim not in constructed_dims:
                        continue
                    wordtype_semdim_scores[word_type][sem_dim].append(extract_metric(entry, metric))
                    wordtype_semdim_words[word_type][sem_dim].append(extract_num_words(entry))

    common_rare_scores = defaultdict(list)
    common_rare_words = defaultdict(list)
    for sem_dim in set(list(wordtype_semdim_scores['common'].keys()) + list(wordtype_semdim_scores['rare'].keys())):
        c_scores = wordtype_semdim_scores['common'].get(sem_dim, [])
        c_words = wordtype_semdim_words['common'].get(sem_dim, [])
        r_scores = wordtype_semdim_scores['rare'].get(sem_dim, [])
        r_words = wordtype_semdim_words['rare'].get(sem_dim, [])
        all_scores = c_scores + r_scores
        all_words = c_words + r_words
        if all_scores:
            common_rare_scores[sem_dim] = all_scores
            common_rare_words[sem_dim] = all_words

    def get_avgs(scores_dict, words_dict):
        avgs = {}
        for sem_dim in scores_dict:
            scores = scores_dict[sem_dim]
            weights = words_dict[sem_dim]
            if scores:
                avg = weighted_avg(scores, weights)
                avgs[sem_dim] = avg
        return avgs

    avgs_dict = {
        'constructed': get_avgs(wordtype_semdim_scores['constructed'], wordtype_semdim_words['constructed']),
        'natural': get_avgs(common_rare_scores, common_rare_words)
    }

    green_rgb = to_rgb("#5c940d")
    red_rgb = to_rgb("#e03131")
    white_rgb = (1, 1, 1)

    natural_avgs = avgs_dict['natural']
    constructed_avgs = avgs_dict['constructed']
    if not natural_avgs:
        print(f"[WARN] No data found for natural (common+rare) with input type: {input_type}")
        return
    sorted_sem_dims = sorted(natural_avgs.keys(), key=lambda x: natural_avgs[x], reverse=False)
    n = len(sorted_sem_dims)

    fig = plt.figure(figsize=(14, max(8, n*0.4)))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.20, 1], wspace=0.25)
    ax_nat = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[0, 1])
    ax_con = fig.add_subplot(gs[0, 2])

    nat_scores = [natural_avgs.get(sem_dim, 0.0) for sem_dim in sorted_sem_dims]
    nat_colors = []
    for v in nat_scores:
        if v > 0.5:
            alpha = min(1.0, (v - 0.5) * 2)
            color = tuple((1 - alpha) * w + alpha * g for w, g in zip(white_rgb, green_rgb))
            nat_colors.append(color)
        elif v < 0.5:
            alpha = min(1.0, (0.5 - v) * 2)
            color = tuple((1 - alpha) * w + alpha * r for w, r in zip(white_rgb, red_rgb))
            nat_colors.append(color)
        else:
            nat_colors.append(white_rgb)
    bars_nat = ax_nat.barh(range(n), nat_scores, color=nat_colors, edgecolor='black', align='center')
    ax_nat.set_yticks([])
    ax_nat.set_xlim(0, 1)
    ax_nat.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_nat.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax_nat.grid(True, alpha=0.3, axis='x')
    ax_nat.set_title('natural', fontsize=15, pad=15)
    # ax_nat.invert_yaxis()
    ax_nat.invert_xaxis()
    for i, (bar, avg) in enumerate(zip(bars_nat, nat_scores)):
        width = bar.get_width()
        ax_nat.text(width+0.01, bar.get_y() + bar.get_height()/2, f'{avg:.3f}', ha='right', va='center', fontsize=10)

    con_scores = [constructed_avgs.get(sem_dim, 0.0) for sem_dim in sorted_sem_dims]
    con_colors = []
    for v in con_scores:
        if v > 0.5:
            alpha = min(1.0, (v - 0.5) * 2)
            color = tuple((1 - alpha) * w + alpha * g for w, g in zip(white_rgb, green_rgb))
            con_colors.append(color)
        elif v < 0.5:
            alpha = min(1.0, (0.5 - v) * 2)
            color = tuple((1 - alpha) * w + alpha * r for w, r in zip(white_rgb, red_rgb))
            con_colors.append(color)
        else:
            con_colors.append(white_rgb)
    bars_con = ax_con.barh(range(n), con_scores, color=con_colors, edgecolor='black', align='center')
    ax_con.set_yticks([])
    ax_con.set_xlim(0, 1)
    ax_con.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_con.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax_con.grid(True, alpha=0.3, axis='x')
    ax_con.set_title('constructed', fontsize=15, pad=15)
    # ax_con.invert_yaxis()
    for i, (bar, avg) in enumerate(zip(bars_con, con_scores)):
        width = bar.get_width()
        ax_con.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{avg:.3f}', ha='left', va='center', fontsize=10)

    ax_y.set_ylim(ax_nat.get_ylim())
    ax_y.set_xlim(0, 1)
    ax_y.set_xticks([])
    ax_y.set_yticks(range(n))
    ax_y.set_yticklabels([])
    for i, label in enumerate(sorted_sem_dims):
        ax_y.text(0.6, i, label, fontsize=12, va='center', ha='center', transform=ax_y.transData)
    ax_y.tick_params(left=False, right=False, labelleft=False, labelright=False)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax_y.spines[spine].set_visible(False)
    ax_y.text(0.6, n + 0.5, 'Semantic Dimension', fontsize=15, ha='center', va='bottom', transform=ax_y.transData)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    file_name = f"semantic_dimensions_by_{input_type}_{metric}{suffix}_middlelabel.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"[plot1] Semantic dimensions by {input_type} input (natural vs constructed, middle label) plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

    if not constructed_avgs:
        print(f"[WARN] No data found for constructed with input type: {input_type}")
        return
    sorted_sem_dims_nat = sorted(natural_avgs.keys(), key=lambda x: natural_avgs[x], reverse=False)
    n_nat = len(sorted_sem_dims_nat)
    sorted_sem_dims_con = sorted(constructed_avgs.keys(), key=lambda x: constructed_avgs[x], reverse=False)
    n_con = len(sorted_sem_dims_con)
    fig2, (ax_nat2, ax_con2) = plt.subplots(1, 2, figsize=(14, max(8, max(n_nat, n_con)*0.4)), width_ratios=[1, 1], gridspec_kw={'wspace':0.45})

    # natural subplot
    nat_scores2 = [natural_avgs.get(sem_dim, 0.0) for sem_dim in sorted_sem_dims_nat]
    nat_colors2 = []
    for v in nat_scores2:
        if v > 0.5:
            alpha = min(1.0, (v - 0.5) * 2)
            color = tuple((1 - alpha) * w + alpha * g for w, g in zip(white_rgb, green_rgb))
            nat_colors2.append(color)
        elif v < 0.5:
            alpha = min(1.0, (0.5 - v) * 2)
            color = tuple((1 - alpha) * w + alpha * r for w, r in zip(white_rgb, red_rgb))
            nat_colors2.append(color)
        else:
            nat_colors2.append(white_rgb)
    bars_nat2 = ax_nat2.barh(range(n_nat), nat_scores2, color=nat_colors2, edgecolor='black', align='center')
    ax_nat2.set_yticks(range(n_nat))
    ax_nat2.set_yticklabels(sorted_sem_dims_nat, fontsize=12, va='center')
    ax_nat2.set_xlim(0, 1)
    ax_nat2.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_nat2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax_nat2.grid(True, alpha=0.3, axis='x')
    ax_nat2.set_title('natural', fontsize=15, pad=15)
    for i, (bar, avg) in enumerate(zip(bars_nat2, nat_scores2)):
        width = bar.get_width()
        ax_nat2.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{avg:.3f}', ha='left', va='center', fontsize=10)

    con_scores2 = [constructed_avgs.get(sem_dim, 0.0) for sem_dim in sorted_sem_dims_con]
    con_colors2 = []
    for v in con_scores2:
        if v > 0.5:
            alpha = min(1.0, (v - 0.5) * 2)
            color = tuple((1 - alpha) * w + alpha * g for w, g in zip(white_rgb, green_rgb))
            con_colors2.append(color)
        elif v < 0.5:
            alpha = min(1.0, (0.5 - v) * 2)
            color = tuple((1 - alpha) * w + alpha * r for w, r in zip(white_rgb, red_rgb))
            con_colors2.append(color)
        else:
            con_colors2.append(white_rgb)
    bars_con2 = ax_con2.barh(range(n_con), con_scores2, color=con_colors2, edgecolor='black', align='center')
    ax_con2.set_yticks(range(n_con))
    ax_con2.set_yticklabels(sorted_sem_dims_con, fontsize=12, va='center')
    ax_con2.set_xlim(0, 1)
    ax_con2.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_con2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax_con2.grid(True, alpha=0.3, axis='x')
    ax_con2.set_title('constructed', fontsize=15, pad=15)
    for i, (bar, avg) in enumerate(zip(bars_con2, con_scores2)):
        width = bar.get_width()
        ax_con2.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{avg:.3f}', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    file_name2 = f"semantic_dimensions_by_{input_type}_{metric}{suffix}_nat_con_separate.png"
    plt.savefig(os.path.join(save_path, file_name2), dpi=300, bbox_inches='tight')
    print(f"[plot2] Semantic dimensions by {input_type} input (separately sorted by natural/constructed, with each subplot y-axis label) plot saved to {os.path.join(save_path, file_name2)}")
    plt.close()

def plot_lang_modeltype_inputype(data, metric='macro_f1_score', save_path=None, input_type='ipa', model_name='Qwen2.5-Omni-7B'):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from collections import defaultdict

    lang_wordtype_scores = defaultdict(lambda: defaultdict(list))
    lang_wordtype_words = defaultdict(lambda: defaultdict(list))
    for model, model_data in data.items():
        if model != model_name:
            continue
        for word_type, word_data in model_data.items():
            for input_type_name, input_data in word_data.items():
                if input_type_name != input_type:
                    continue
                lang = input_data.get('language', None)
                if lang is None:
                    dims = input_data.get('dimensions', {})
                    for dim_entry in dims.values():
                        lang = dim_entry.get('language', None)
                        if lang is not None:
                            break
                if lang is None:
                    lang = word_type if word_type in ['en', 'fr', 'ko', 'ja'] else 'unknown'
                dims = input_data.get('dimensions', {})
                for sem_dim, entry in dims.items():
                    lang_wordtype_scores[lang][word_type].append(entry.get(metric, 0.0))
                    lang_wordtype_words[lang][word_type].append(entry.get('count', 1))

    languages = sorted(lang_wordtype_scores.keys())
    word_types = sorted(set(wt for d in lang_wordtype_scores.values() for wt in d.keys()))
    avg_scores = np.zeros((len(languages), len(word_types)))
    for i, lang in enumerate(languages):
        for j, wt in enumerate(word_types):
            scores = np.array(lang_wordtype_scores[lang][wt])
            weights = np.array(lang_wordtype_words[lang][wt])
            if len(scores) > 0 and weights.sum() > 0:
                avg_scores[i, j] = np.sum(scores * weights) / np.sum(weights)
            else:
                avg_scores[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(2.5*len(languages), 6))
    x_labels = []
    x_ticks = []
    bar_width = 0.18
    group_gap = 0.25
    colors = ['#ffd43b', '#20c997', '#f06595', '#845ef7', '#fd7e14']
    xpos = 0
    for i, lang in enumerate(languages):
        for j, wt in enumerate(word_types):
            score = avg_scores[i, j]
            if not np.isnan(score):
                ax.bar(xpos, score, width=bar_width, color=colors[j % len(colors)], edgecolor='black', label=wt if i == 0 else "")
            x_labels.append(f"{lang}\n{wt}")
            x_ticks.append(xpos)
            xpos += 1
        xpos += group_gap
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=12)
    ax.set_ylabel(get_metric_label(metric), fontsize=13)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.grid(True, alpha=0.3, axis='y')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12, loc='upper right')
    ax.set_title(f"{model_name} ({input_type.upper()}) Performance by Language and Word Type", fontsize=15, pad=15)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_name = f"lang_wordtype_{model_name.replace('/', '_')}_{input_type}_{metric}.png"
        plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
        print(f"Plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def main(json_path, metric='macro_f1_score', sem_dims=None, save_path=None, filter_constructed=False, input_type='ipa'):
    data = load_stat_json(json_path)
    all_dims = set()
    for model_data in data.values():
        for word_data in model_data.values():
            for input_data in word_data.values():
                dims = input_data.get('dimensions', {})
                all_dims.update(dims.keys())
    
    if filter_constructed:
        avg_by_model = compute_avg_by_condition(data, ['model'], metric, filter_constructed=True)
        avg_by_wordtype = compute_avg_by_condition(data, ['word_type'], metric, filter_constructed=True)
        avg_by_inputtype = compute_avg_by_condition(data, ['input_type'], metric, filter_constructed=True)
        avg_by_model_input = compute_avg_by_condition(data, ['model', 'input_type'], metric, filter_constructed=True)
        category_avgs = compute_avg_by_category(data, metric, filter_constructed=True)
    else:
        avg_by_model = compute_avg_by_condition(data, ['model'], metric, filter_constructed=False)
        avg_by_wordtype = compute_avg_by_condition(data, ['word_type'], metric, filter_constructed=False)
        avg_by_inputtype = compute_avg_by_condition(data, ['input_type'], metric, filter_constructed=False)
        avg_by_model_input = compute_avg_by_condition(data, ['model', 'input_type'], metric, filter_constructed=False)
        category_avgs = compute_avg_by_category(data, metric, filter_constructed=False)
    
    model_input_groups = set(g for (g, sd) in avg_by_model_input.keys())
    for group in model_input_groups:
        plot_grouped_horizontal_bar(
            avg_by_model_input, group, metric, 
            title=f"Model: {group[0]}, Input Type: {group[1]}",
            save_path=save_path, top_n=25, bottom_n=25, sem_dims=sem_dims, filter_constructed=filter_constructed
        )
    
    # plot_wordtype_performance(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    # plot_inputtype_performance(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    # plot_inputtype_wordtype_scatter(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    # plot_inputtype_wordtype_modeltype_scatter(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    # plot_inputtype_wordtype_modeltype_semdim_scatter(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    # plot_inputtype_wordtype_modeltype_all_semdim_scatter(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    # plot_inputtype_modeltype_wordtype_all_semdim_scatter(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    # plot_inputtype_per_wordtype_all_models(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    
    # # Generate new requested plots
    # plot_high_performance_semantic_dimensions(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    # plot_inference_results_by_input_model_word(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    # plot_inference_results_by_semantic_input_model_word(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    # plot_wordtype_semantic_dimension_bars(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    # plot_semantic_dimension_variance_by_wordtype(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    # plot_inputtype_wordtype_semdim_scatter(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    plot_semantic_dimensions_by_input_type(data, input_type=input_type, metric=metric, save_path=save_path, filter_constructed=filter_constructed)
    plot_lang_modeltype_inputype(data, metric=metric, save_path=save_path, input_type=input_type)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='./results/statistics/semdim_stat.json')
    parser.add_argument('--metric', type=str, default='macro_f1_score', help='f1 or accuracy')
    parser.add_argument('--sem_dims', type=str, default=None, help='Comma-separated list of semantic dimensions to plot (default: all)')
    parser.add_argument('--save_path', type=str, default="./results/plots/rq1", help='Path to save the plots')
    parser.add_argument('--filter_constructed', action='store_true', help='Filter out constructed word semantic dimensions')
    parser.add_argument('--input_type', type=str, default='ipa', help='Input type to plot semantic dimensions for (default: ipa)')
    args = parser.parse_args()
    sem_dims = None
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    if args.sem_dims:
        sem_dims = [s.strip() for s in args.sem_dims.split(',') if s.strip()]
    main(args.json_path, metric=args.metric, sem_dims=sem_dims, save_path=save_path, filter_constructed=args.filter_constructed, input_type=args.input_type)
