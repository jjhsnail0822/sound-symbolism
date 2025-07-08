import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.font_manager import FontProperties

# python src/analysis/plotting/rq1_plot.py --json_path ./results/statistics/semdim_stat.json --metric f1 --sem_dims "sem_dim_01,sem_dim_05,sem_dim_12"

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

def extract_metric(entry, metric='f1'):
    # entry: {"accuracy": 0.7, "precision": 0.7, "recall": 0.7, "f1_score": 0.7, "count": 2800}
    if metric.lower() == 'f1':
        return entry["f1_score"]
    elif metric.lower() == 'accuracy' or metric.lower() == 'acc':
        return entry["accuracy"]
    elif metric.lower() == 'precision':
        return entry["precision"]
    elif metric.lower() == 'recall':
        return entry["recall"]
    else:
        raise ValueError(f"Unknown metric: {metric}")

def extract_num_words(entry):
    return entry["count"]

def compute_avg_by_condition(data, group_keys, metric='f1', filter_constructed=False):
    from collections import defaultdict
    semdim_scores = defaultdict(list)
    semdim_words = defaultdict(list)
    
    constructed_dims = {
        "beautiful-ugly", "delicate-rugged", "tense-relaxed", "simple-complex", 
        "happy-sad", "sharp-round", "fast-slow", "masculine-feminine", 
        "strong-weak", "exciting-calming", "harsh-mellow", "ordinary-unique", 
        "dangerous-safe", "realistic-fantastical", "abrupt-continuous", 
        "passive-active", "big-small", "heavy-light", "hard-soft", 
        "solid-nonsolid", "inhibited-free"
    }
    
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

def plot_horizontal_bar(avg_dict, group, metric='f1', title=None, save_path=None, sem_dims=None, filter_constructed=False):
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
    
    category_colors = {
        "valence": "#FF6B6B",
        "potency": "#4ECDC4",
        "activity": "#45B7D1",
        "other factors": "#96CEB4",
        "stimulus dimensions": "#FF8C42",
        "other dimensions": "#DDA0DD"
    }
    
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
    ax.set_xlabel(f'{metric} score', fontsize=13)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(x=0.35, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
    ax.axvline(x=0.65, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
    
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
    os.makedirs(full_save_path, exist_ok=True)
    
    suffix = "_filter" if filter_constructed else ""
    file_name = f"{len(sem_dims) if sem_dims is not None else 'all'}_dims{suffix}.png"
    plt.savefig(os.path.join(full_save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(full_save_path, file_name)}")
    plt.close()

def compute_avg_by_category(data, metric='f1', filter_constructed=False):
    from collections import defaultdict
    category_scores = defaultdict(list)
    category_words = defaultdict(list)
    
    constructed_dims = {
        "beautiful-ugly", "delicate-rugged", "tense-relaxed", "simple-complex", 
        "happy-sad", "sharp-round", "fast-slow", "masculine-feminine", 
        "strong-weak", "exciting-calming", "harsh-mellow", "ordinary-unique", 
        "dangerous-safe", "realistic-fantastical", "abrupt-continuous", 
        "passive-active", "big-small", "heavy-light", "hard-soft", 
        "solid-nonsolid", "inhibited-free"
    }
    
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

def plot_category_performance(category_avgs, metric='f1', title=None, save_path=None, filter_constructed=False):
    if not category_avgs:
        return
    
    sorted_items = sorted(category_avgs.items(), key=lambda x: x[1][0], reverse=False)
    categories = [item[0] for item in sorted_items]
    avgs = [item[1][0] for item in sorted_items]
    total_words = [item[1][1] for item in sorted_items]
    
    colors = []
    for v in avgs:
        if v >= 0.5:
            green_intensity = min(1.0, (v - 0.5) * 2)
            colors.append((1 - green_intensity * 0.7, 1 - green_intensity * 0.3, 1 - green_intensity * 0.7))
        else:
            red_intensity = min(1.0, (0.5 - v) * 2)
            colors.append((1 - red_intensity * 0.3, 1 - red_intensity * 0.7, 1 - red_intensity * 0.7))
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(categories)*0.4)))
    bars = ax.barh(range(len(avgs)), avgs, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=12, va='center')
    ax.set_ylabel('Semantic Categories', fontsize=13)
    
    ax.set_xlabel(f'{metric} score', fontsize=13)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(x=0.35, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
    ax.axvline(x=0.65, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
    
    for i, (bar, avg, total) in enumerate(zip(bars, avgs, total_words)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{avg:.3f} (n={total})', ha='left', va='center', fontsize=10)
    
    if title:
        ax.set_title(title, fontsize=15, pad=15)
    
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    file_name = f"category_performance_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Category performance plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def plot_wordtype_performance(data, metric='f1', save_path=None, filter_constructed=False):
    from collections import defaultdict
    
    constructed_dims = {
        "beautiful-ugly", "delicate-rugged", "tense-relaxed", "simple-complex", 
        "happy-sad", "sharp-round", "fast-slow", "masculine-feminine", 
        "strong-weak", "exciting-calming", "harsh-mellow", "ordinary-unique", 
        "dangerous-safe", "realistic-fantastical", "abrupt-continuous", 
        "passive-active", "big-small", "heavy-light", "hard-soft", 
        "solid-nonsolid", "inhibited-free"
    }
    
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
                    
                    semdim_scores[sem_dim][word_type].append(extract_metric(entry, metric))
                    semdim_words[sem_dim][word_type].append(extract_num_words(entry))
    
    semdim_avgs = {}
    for sem_dim in semdim_scores:
        semdim_avgs[sem_dim] = {}
        for word_type in ['rare', 'constructed', 'common']:
            if word_type in semdim_scores[sem_dim]:
                scores = semdim_scores[sem_dim][word_type]
                weights = semdim_words[sem_dim][word_type]
                avg = weighted_avg(scores, weights)
                semdim_avgs[sem_dim][word_type] = avg
            else:
                semdim_avgs[sem_dim][word_type] = 0.0
    
    if not semdim_avgs:
        return
    
    word_types = ['constructed', 'rare', 'common']
    sem_dims = list(semdim_avgs.keys())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        "valence": "#FF6B6B",
        "potency": "#4ECDC4",
        "activity": "#45B7D1",
        "other factors": "#96CEB4",
        "stimulus dimensions": "#FF8C42",
        "other dimensions": "#DDA0DD"
    }
    
    for sem_dim in sem_dims:
        values = [semdim_avgs[sem_dim].get(wt, 0.0) for wt in word_types]
        category = find_category(sem_dim)
        color = colors.get(category, 'gray')
        line = ax.plot(word_types, values, marker='o', linewidth=2, markersize=6, color=color, label=sem_dim, alpha=0.8)
        
        constructed_value = semdim_avgs[sem_dim].get('constructed', 0.0)
        ax.text(-0.1, constructed_value, sem_dim, ha='right', va='center', fontsize=9, 
                color=color, alpha=0.8, transform=ax.get_yaxis_transform())
    
    ax.set_xlabel('Word Types', fontsize=13)
    ax.set_ylabel(f'{metric} score', fontsize=13)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(y=0.35, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
    ax.axhline(y=0.65, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
    
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    file_name = f"wordtype_performance_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Word type performance plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def plot_ranking_change(data, metric='f1', save_path=None, filter_constructed=False):
    from collections import defaultdict
    
    constructed_dims = {
        "beautiful-ugly", "delicate-rugged", "tense-relaxed", "simple-complex", 
        "happy-sad", "sharp-round", "fast-slow", "masculine-feminine", 
        "strong-weak", "exciting-calming", "harsh-mellow", "ordinary-unique", 
        "dangerous-safe", "realistic-fantastical", "abrupt-continuous", 
        "passive-active", "big-small", "heavy-light", "hard-soft", 
        "solid-nonsolid", "inhibited-free"
    }
    
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
                    
                    semdim_scores[sem_dim][word_type].append(extract_metric(entry, metric))
                    semdim_words[sem_dim][word_type].append(extract_num_words(entry))
    
    semdim_avgs = {}
    for sem_dim in semdim_scores:
        semdim_avgs[sem_dim] = {}
        for word_type in ['constructed', 'rare', 'common']:
            if word_type in semdim_scores[sem_dim]:
                scores = semdim_scores[sem_dim][word_type]
                weights = semdim_words[sem_dim][word_type]
                avg = weighted_avg(scores, weights)
                semdim_avgs[sem_dim][word_type] = avg
            else:
                semdim_avgs[sem_dim][word_type] = 0.0
    
    if not semdim_avgs:
        return
    
    word_types = ['constructed', 'rare', 'common']
    sem_dims = list(semdim_avgs.keys())
    
    # 각 word type별로 등수 계산
    rankings = {}
    for word_type in word_types:
        scores = [(sem_dim, semdim_avgs[sem_dim].get(word_type, 0.0)) for sem_dim in sem_dims]
        scores.sort(key=lambda x: x[1], reverse=True)  # 성능 높은 순으로 정렬
        rankings[word_type] = {sem_dim: rank + 1 for rank, (sem_dim, _) in enumerate(scores)}
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        "valence": "#FF6B6B",
        "potency": "#4ECDC4",
        "activity": "#45B7D1",
        "other factors": "#96CEB4",
        "stimulus dimensions": "#FF8C42",
        "other dimensions": "#DDA0DD"
    }
    
    for sem_dim in sem_dims:
        values = [rankings[wt][sem_dim] for wt in word_types]
        category = find_category(sem_dim)
        color = colors.get(category, 'gray')
        line = ax.plot(word_types, values, marker='o', linewidth=2, markersize=6, color=color, label=sem_dim, alpha=0.8)
        
        # constructed 지점에서 Y축에 semantic dimension 이름 표시
        constructed_rank = rankings['constructed'][sem_dim]
        ax.text(-0.1, constructed_rank, sem_dim, ha='right', va='center', fontsize=9, 
                color=color, alpha=0.8, transform=ax.get_yaxis_transform())
    
    ax.set_xlabel('Word Types', fontsize=13)
    ax.set_ylabel('Ranking', fontsize=13)
    ax.invert_yaxis()  # 1등이 위쪽에 오도록 Y축 뒤집기
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    suffix = "_filter" if filter_constructed else ""
    file_name = f"ranking_change_{metric}{suffix}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Ranking change plot saved to {os.path.join(save_path, file_name)}")
    plt.close()

def main(json_path, metric='f1', sem_dims=None, save_path=None, filter_constructed=False):
    data = load_stat_json(json_path)
    avg_by_model = compute_avg_by_condition(data, ['model'], metric, filter_constructed)
    for model_name in set(g[0] for (g, sd) in avg_by_model.keys()):
        plot_horizontal_bar(avg_by_model, (model_name,), metric, title=f"Model: {model_name}", sem_dims=sem_dims, save_path=save_path, filter_constructed=filter_constructed)
    avg_by_wordtype = compute_avg_by_condition(data, ['word_type'], metric, filter_constructed)
    for word_type in set(g[0] for (g, sd) in avg_by_wordtype.keys()):
        plot_horizontal_bar(avg_by_wordtype, (word_type,), metric, title=f"Word Type: {word_type}", sem_dims=sem_dims, save_path=save_path, filter_constructed=filter_constructed)
    avg_by_inputtype = compute_avg_by_condition(data, ['input_type'], metric, filter_constructed)
    for input_type in set(g[0] for (g, sd) in avg_by_inputtype.keys()):
        plot_horizontal_bar(avg_by_inputtype, (input_type,), metric, title=f"Input Type: {input_type}", sem_dims=sem_dims, save_path=save_path, filter_constructed=filter_constructed)
    avg_by_model_input = compute_avg_by_condition(data, ['model', 'input_type'], metric, filter_constructed)
    model_input_groups = set(g for (g, sd) in avg_by_model_input.keys())
    for group in model_input_groups:
        plot_horizontal_bar(avg_by_model_input, group, metric, title=f"Model: {group[0]}, Input Type: {group[1]}", sem_dims=sem_dims, save_path=save_path, filter_constructed=filter_constructed)
    
    category_avgs = compute_avg_by_category(data, metric, filter_constructed)
    plot_category_performance(category_avgs, metric, title=f"Category-wise {metric} Performance", save_path=save_path, filter_constructed=filter_constructed)
    
    plot_wordtype_performance(data, metric, save_path=save_path, filter_constructed=filter_constructed)
    plot_ranking_change(data, metric, save_path=save_path, filter_constructed=filter_constructed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='./results/statistics/semdim_stat.json')
    parser.add_argument('--metric', type=str, default='f1', help='f1 or accuracy')
    parser.add_argument('--sem_dims', type=str, default=None, help='Comma-separated list of semantic dimensions to plot (default: all)')
    parser.add_argument('--save_path', type=str, default="./results/plots/rq1", help='Path to save the plots')
    parser.add_argument('--filter_constructed', action='store_true', help='Filter out constructed word semantic dimensions')
    args = parser.parse_args()
    sem_dims = None
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    if args.sem_dims:
        sem_dims = [s.strip() for s in args.sem_dims.split(',') if s.strip()]
    main(args.json_path, metric=args.metric, sem_dims=sem_dims, save_path=save_path, filter_constructed=args.filter_constructed)
