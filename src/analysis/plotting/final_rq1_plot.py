import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

constructed_dims = [
    "beautiful-ugly", "delicate-rugged", "tense-relaxed", "simple-complex", 
    "happy-sad", "sharp-round", "fast-slow", "masculine-feminine", 
    "strong-weak", "exciting-calming", "harsh-mellow", "ordinary-unique", 
    "dangerous-safe", "realistic-fantastical", "abrupt-continuous", 
    "passive-active", "big-small", "heavy-light", "hard-soft", 
    "solid-nonsolid", "inhibited-free"
]

human_eval_result = {
        "big-small": {"macro_f1_score": 0.8458},
        "tense-relaxed": {"macro_f1_score": 0.8194},
        "ordinary-unique": {"macro_f1_score": 0.7948},
        "sharp-round": {"macro_f1_score": 0.7405},
        "simple-complex": {"macro_f1_score": 0.7352},
        "harsh-mellow": {"macro_f1_score": 0.7306},
        "beautiful-ugly": {"macro_f1_score": 0.6944},
        "abrupt-continuous": {"macro_f1_score": 0.6659},
        "heavy-light": {"macro_f1_score": 0.6638},
        "masculine-feminine": {"macro_f1_score": 0.6538},
        "happy-sad": {"macro_f1_score": 0.6478},
        "inhibited-free": {"macro_f1_score": 0.6458},
        "dangerous-safe": {"macro_f1_score": 0.6247},
        "fast-slow": {"macro_f1_score": 0.5875},
        "exciting-calming": {"macro_f1_score": 0.5717},
        "hard-soft": {"macro_f1_score": 0.5678},
        "solid-nonsolid": {"macro_f1_score": 0.5652},
        "strong-weak": {"macro_f1_score": 0.5637},
        "realistic-fantastical": {"macro_f1_score": 0.4330},
}

def get_color(val):
    green = np.array(to_rgb("#5c940d"))
    red = np.array(to_rgb("#e03131"))
    white = np.array([1, 1, 1])
    if val >= 0.5:
        alpha = min(1.0, (val - 0.5) * 2)
        return tuple((1 - alpha) * white + alpha * green)
    else:
        alpha = min(1.0, (0.5 - val) * 2)
        return tuple((1 - alpha) * white + alpha * red)
    
def plot_final_rq1_bar_comparison(human_eval, constructed, natural, metric_label='macro_f1_score', input_type=''):
    dims = [d for d in constructed_dims if d in human_eval]
    # print(f"[DEBUG] dims (from human_eval): {dims}")
    def get_scores(dims, dct, metric_label):
        scores = []
        for dim in dims:
            score_dict = dct.get(dim, None)
            if score_dict is None:
                scores.append(np.nan)
            else:
                v = score_dict.get(metric_label, np.nan)
                scores.append(v)
        return scores

    group_data = {
        'Natural': get_scores(dims, natural, metric_label),
        'Constructed': get_scores(dims, constructed, metric_label),
        'Human (Audio only)': get_scores(dims, human_eval, metric_label),
    }
    two_stage_font_size = 16
    one_stage_font_size = 20
    # print(f"[DEBUG] group_data: {group_data}")
    group_names = list(group_data.keys())
    n_groups = len(group_names)
    fig, axes = plt.subplots(1, n_groups, figsize=(20, 8), sharey=False)
    # fig, axes = plt.subplots(1, n_groups, figsize=(12, 8), sharey=False)
    if n_groups == 1:
        axes = [axes]

    for ax, group in zip(axes, group_names):
        scores = group_data[group]
        filtered = [(d, s) for d, s in zip(dims, scores) if not np.isnan(s)]
        if not filtered:
            ax.set_title(group, fontsize=two_stage_font_size, pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        dims_filtered, scores_filtered = zip(*filtered)
        sort_idx = np.argsort([-s for s in scores_filtered])
        dims_sorted = [dims_filtered[i] for i in sort_idx]
        scores_sorted = [scores_filtered[i] for i in sort_idx]
        colors = [get_color(v) for v in scores_sorted]
        n = len(dims_sorted)
        bars = ax.barh(range(n), scores_sorted, color=colors, edgecolor='black')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Macro-F1 Score", fontsize=one_stage_font_size)
        ax.set_yticks(range(n))
        ax.set_yticklabels(dims_sorted, fontsize=two_stage_font_size)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        ax.set_title(group, fontsize=two_stage_font_size, pad=10)
    plt.tight_layout()
    save_path = f"./results/plots/rq1/final_rq1_bar_comparison_{input_type}_{metric_label}.png"
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    save_path = f"./results/plots/rq1/final_rq1_bar_comparison_{input_type}_{metric_label}.pdf"
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

def aggregate_score_and_count(language_list, data, metric, final_semdim_score, final_semdim_count, final_input_type_list):
    for language in language_list:
        model_list = data[language].keys()
        for model in model_list:
            word_types = data[language][model].keys()
            for word_type in word_types:
                if word_type not in ["constructed", "common", "rare"]:
                    continue
                input_types = data[language][model][word_type].keys()
                for input_type in input_types:
                    if input_type not in ["ipa", "audio", "ipa_and_audio", "original", "original_and_audio"]:
                        continue
                    if input_type not in final_input_type_list:
                        final_input_type_list.append(input_type)
                    semantic_dimensions = data[language][model][word_type][input_type]["dimensions"].keys()
                    for semdim in semantic_dimensions:
                        if semdim not in constructed_dims:
                            continue
                        if semdim not in final_semdim_score[input_type][word_type]:
                            final_semdim_score[input_type][word_type][semdim] = []
                        final_semdim_score[input_type][word_type][semdim].append(data[language][model][word_type][input_type]["dimensions"][semdim][metric])
                        if semdim not in final_semdim_count[input_type][word_type]:
                            final_semdim_count[input_type][word_type][semdim] = []
                        final_semdim_count[input_type][word_type][semdim].append(data[language][model][word_type][input_type]["dimensions"][semdim]["count"])
    return final_semdim_score, final_semdim_count, final_input_type_list

def compute_mean_average_score(final_semdim_score, final_semdim_count, final_semdim_score_weighted, eval_metric:str="macro_f1_score"):
    semdim_wordtype_scores = {
        "constructed": {},
        "natural": {},
    }
    for input_type in final_semdim_score.keys():
        for word_type in final_semdim_score[input_type].keys():
            if word_type in ["common", "rare"]:
                final_word_type = "natural"
            else:
                final_word_type = "constructed"
            for semdim in final_semdim_score[input_type][word_type].keys():
                metrics = final_semdim_score[input_type][word_type][semdim]
                if semdim not in semdim_wordtype_scores[final_word_type]:
                    semdim_wordtype_scores[final_word_type][semdim] = []
                semdim_wordtype_scores[final_word_type][semdim].extend(metrics)
    for word_type in ["constructed", "natural"]:
        for semdim, values in semdim_wordtype_scores[word_type].items():
            if len(values) > 0:
                mean_val = float(np.mean(values))
            else:
                mean_val = float('nan')
            if semdim not in final_semdim_score_weighted["whole"][word_type]:
                final_semdim_score_weighted["whole"][word_type][semdim] = {}
            final_semdim_score_weighted["whole"][word_type][semdim][eval_metric] = mean_val
            print(f"[DEBUG] word_type={word_type}, semdim={semdim}, mean={mean_val}")
    return final_semdim_score_weighted

def aggregate_macro_f1_by_inputtype(json_path, constructed_langs=['art'], natural_langs=['en','fr','ja','ko'], constructed_dims=constructed_dims, metric='macro_f1_score'):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Always use {lang: {model: ...}} structure
    language_list = data.keys()
    final_input_type_list = []
    final_semdim_score = {
        "ipa": {"constructed": {}, "common": {}, "rare": {}},
        "audio": {"constructed": {}, "common": {}, "rare": {}},
        "ipa_and_audio": {"constructed": {}, "common": {}, "rare": {}},
        "original": {"constructed": {}, "common": {}, "rare": {}},
        "original_and_audio": {"constructed": {}, "common": {}, "rare": {}},
    }
    final_semdim_count = {
        "ipa": {"constructed": {}, "common": {}, "rare": {}},
        "audio": {"constructed": {}, "common": {}, "rare": {}},
        "ipa_and_audio": {"constructed": {}, "common": {}, "rare": {}},
        "original": {"constructed": {}, "common": {}, "rare": {}},
        "original_and_audio": {"constructed": {}, "common": {}, "rare": {}},
    }
    final_semdim_score_weighted = {
        "whole": {"constructed": {}, "natural": {}},
    }

    final_semdim_score, final_semdim_count, final_input_type_list = aggregate_score_and_count(language_list, data, metric, final_semdim_score, final_semdim_count, final_input_type_list)
    
    final_semdim_score_weighted = compute_mean_average_score(final_semdim_score, final_semdim_count, final_semdim_score_weighted)

    return final_semdim_score_weighted

if __name__ == "__main__":
    json_path = './results/statistics/semdim_stat.json'
    metric = 'macro_f1_score'
    input_types = ['ipa', 'audio', 'ipa_and_audio', 'original', 'original_and_audio']
    results = aggregate_macro_f1_by_inputtype(json_path, constructed_langs=['art'], natural_langs=['en','fr','ja','ko'], constructed_dims=constructed_dims, metric=metric)
    constructed_scores = results["whole"]["constructed"]
    natural_scores = results["whole"]["natural"]
    # Check if there is any valid data
    print(f"\nPlotting for input_type: whole")
    plot_final_rq1_bar_comparison(human_eval_result, constructed_scores, natural_scores, metric_label='macro_f1_score', input_type='whole')

