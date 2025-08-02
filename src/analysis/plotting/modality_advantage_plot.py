import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
import pandas as pd
from matplotlib.patches import Patch

def load_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def convert_to_dataframe(data):
    columns = ['idx', 'lang', 'model', 'type', 'input_type', 'sem_dim', 'macro_f1', 'accuracy']
    df_rows = []
    idx = int(0)
    for model, types in data.items():
        for type, input_types in types.items():        
            for input_type, sem_dims in input_types.items():
                for sem_dim, langs in sem_dims.items():
                    for lang, stats in langs.items():
                        if 'gpt-4o-audio-preview' == model:
                            model = 'gpt-4o'
                        df_rows.append([idx, lang, model, type, input_type, sem_dim, stats['macro_f1_score'], stats['accuracy']])
                        idx += 1
    df = pd.DataFrame(df_rows, columns=columns)
    df.set_index('idx', inplace=True)
    return df

def print_stats(df):
    print("DataFrame Statistics:")
    print(df.describe())
    print("\nUnique Languages:", df['lang'].unique())
    print("\nUnique Models:", df['model'].unique())
    print("\nUnique Types:", df['type'].unique())
    print("\nUnique Input Types:", df['input_type'].unique())
    print("\nUnique Semantic Dimensions:", df['sem_dim'].unique())


def plot_modality_advantage(df, lang, mod1, mod2, all_models_only=False):
    """
    :param df: DataFrame containing the data
    :param lang: Language to filter the DataFrame
    :param mod1: First modality to compare (e.g., 'audio')
    :param mod2: Second modality to compare (e.g., 'original')  
    """

    if all_models_only:
        models = ['ALL_MODELS']
    else:
        models = df['model'].unique().tolist()
        models.append('ALL_MODELS') 
    n_models = len(models)

    if n_models > 1:
        n_cols = 2
        n_rows = (n_models + 1) // 2
        figsize = (12, 5 * n_rows)  # Adjust height based on number of rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = [axes] if n_cols == 1 else [axes[0]]
        elif n_rows == 1:
            axes = axes if n_models > 1 else [axes]
        else:
            axes = axes.flatten()
    else:
        n_cols = 1
        n_rows = 1
        figsize = (8, 5)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = [axes] if n_cols == 1 else [axes[0]]
        
    
    for i, model in enumerate(models):
        if model == 'ALL_MODELS':
            model_df = df
        else:
            model_df = df[df['model'] == model]
        # print(f'[DEBUG] Model DF Length : {len(model_df)}')

        if lang in ['en', 'ko', 'ja', 'fr', 'art']:
            model_df = model_df[model_df['lang'] == lang]
            # print(f"[DEBUG] Filtered by language: {lang}, remaining rows: {len(model_df)}")
        elif lang in ['constructed', 'natural']:
            model_df = model_df[model_df['type'] == lang]
            # print(f"[DEBUG] Filtered by type: {lang}, remaining rows: {len(model_df)}")
        else:
            print(f"Unknown language type: {lang}. Using all data.")

        dfs = []
        for input_type in [mod1, mod2]:
            input_df = model_df[model_df['input_type'] == input_type]
            semdim_f1 = input_df.groupby(['sem_dim'])['macro_f1'].mean()
            dfs.append(semdim_f1)

        common_dims = dfs[0].index.intersection(dfs[1].index)
        dfs_common = [df[common_dims] for df in dfs]
        
        difference = dfs_common[0] - dfs_common[1]
        sorted_dims = difference.sort_values(ascending=False).index

        dfs_sorted = [df[sorted_dims] for df in dfs_common]
        

        ax = axes[i]
        x_positions = range(len(sorted_dims))

        for j, (df_sorted, marker, color) in enumerate(zip(dfs_sorted, ['o', 's'], ['blue', 'red'], )):
            ax.plot(x_positions, df_sorted.values, 
                    marker=marker, linewidth=2 + j, label=f'{mod1 if j == 0 else mod2}', color=color, alpha=0.8)
            

        difference_sorted = dfs_sorted[0] - dfs_sorted[1]

        for idx in range(len(difference_sorted) - 1):
            if difference_sorted.iloc[idx] > 0 and difference_sorted.iloc[idx + 1] < 0:
                cross_point = idx + difference_sorted.iloc[idx] / (difference_sorted.iloc[idx] - difference_sorted.iloc[idx + 1])
                ax.axvline(x=cross_point, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Crossover')
                break

        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Semantic Dimension')
        ax.set_ylabel('F1 Macro Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(sorted_dims, rotation=45, ha='right')
   
    if n_models < len(axes):
        for j in range(n_models, len(axes)):
            fig.delaxes(axes[j])
    
    plt.suptitle(f'{lang.capitalize()} - {mod1.capitalize()} vs {mod2.capitalize()}\n'
                 f'Comparison of {mod1} and {mod2} modalities across models',
                    fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def statistical_significance_analysis(df, lang=None):
    results = {}

    if lang is not None:
        if lang in ['en', 'ko', 'ja', 'fr', 'art']:
            df = df[df['lang'] == lang]
        elif lang in ['constructed', 'natural']:
            df = df[df['type'] == lang]
        else:
            print(f"Unknown language type: {lang}. Using all data.")
            
    for dim in df['sem_dim'].unique():
        dim_data = df[df['sem_dim'] == dim]
        audio_scores = dim_data[dim_data['input_type']=='audio']['macro_f1'].values
        text_scores = dim_data[dim_data['input_type']=='original']['macro_f1'].values
        
        if len(audio_scores) > 0 and len(text_scores) > 0:
            t_stat, p_value = ttest_rel(audio_scores, text_scores)
            
            w_stat, w_p_value = wilcoxon(audio_scores, text_scores)
            
            mean_diff = np.mean(audio_scores) - np.mean(text_scores)
            pooled_std = np.sqrt(((np.std(audio_scores)**2 + np.std(text_scores)**2) / 2))
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            results[dim] = {
                'mean_difference': mean_diff,
                'p_value': p_value,
                'w_p_value': w_p_value,
                'cohens_d': cohens_d,
                'significance': 'significant' if p_value < 0.05 else 'non-significant'
            }
    
    return results


def modality_advantage_correlation(df, lang1, lang2, mod1, mod2, model=None):
    """
    :param df: DataFrame containing the data
    :param lang1: First language to compare
    :param lang2: Second language to compare
    :param mod1: First modality to compare (e.g., 'audio')
    :param mod2: Second modality to compare (e.g., 'original')  
    """
    dfs = []
    for lang in [lang1, lang2]:
        if model is not None:
            df = df[df['model'] == model]
            print(f"[DEBUG] Filtered by model: {model}, remaining rows: {len(df)}")
        if lang in ['en', 'ko', 'ja', 'fr', 'art']:
            lang_df = df[df['lang'] == lang]
        elif lang in ['constructed', 'natural']:
            lang_df = df[df['type'] == lang]
        else:
            print(f"Unknown language type: {lang}. Using all data.")
            lang_df = df

        dfs_lang = []
        for input_type in [mod1, mod2]:
            input_df = lang_df[lang_df['input_type'] == input_type]
            semdim_f1 = input_df.groupby(['sem_dim'])['macro_f1'].mean()
            dfs_lang.append(semdim_f1)
        
        common_dims = dfs_lang[0].index.intersection(dfs_lang[1].index)
        dfs_lang_common = [df[common_dims] for df in dfs_lang]
        
        difference = dfs_lang_common[0] - dfs_lang_common[1]
        sorted_dims = difference.sort_values(ascending=False).index
        
        dfs_sorted = [df[sorted_dims] for df in dfs_lang_common]
        
        dfs.append(dfs_sorted)

    lang1_dfs, lang2_dfs = dfs
    lang1_advantage = lang1_dfs[0] - lang1_dfs[1]
    lang2_advantage = lang2_dfs[0] - lang2_dfs[1]
    common_dims = lang1_advantage.index.intersection(lang2_advantage.index)
    lang1_advantage_common = lang1_advantage[common_dims]
    lang2_advantage_common = lang2_advantage[common_dims]
    lang1_advantage_values = lang1_advantage_common.values
    lang2_advantage_values = lang2_advantage_common.values
    
    pearson_corr, pearson_p = stats.pearsonr(lang1_advantage_values, lang2_advantage_values)    
    spearman_corr, spearman_p = stats.spearmanr(lang1_advantage_values, lang2_advantage_values) 

    dimension_groups = {
        'audio_dominant': [],
        'text_dominant': [],
        'mixed_audio_text': [],
        'mixed_text_audio': [],
        'neutral': []
    }
    colors = []
    for i in range(len(lang1_advantage_values)):
        val1 = lang1_advantage_values[i]
        val2 = lang2_advantage_values[i]
        
        if val1 > 0 and val2 > 0:
            colors.append('darkblue')
            dimension_groups['audio_dominant'].append(common_dims[i])
        elif val1 < 0 and val2 < 0:
            colors.append('darkred')
            dimension_groups['text_dominant'].append(common_dims[i])
        elif val1 > 0 and val2 < 0:
            colors.append('lightblue')
            dimension_groups['mixed_audio_text'].append(common_dims[i])
        elif val1 < 0 and val2 > 0:
            colors.append('lightcoral')
            dimension_groups['mixed_text_audio'].append(common_dims[i])
        else:
            colors.append('gray')
            dimension_groups['neutral'].append(common_dims[i])

    plt.figure(figsize=(12, 10))
    plt.scatter(lang1_advantage_values, lang2_advantage_values,
                alpha=0.7, s=100, c=colors, edgecolors='black') 
    
    for i, dim in enumerate(common_dims):
        plt.annotate(dim, (lang1_advantage_values[i], lang2_advantage_values[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=10, alpha=0.9)
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    min_val = min(min(lang1_advantage_values), min(lang2_advantage_values))
    max_val = max(max(lang1_advantage_values), max(lang2_advantage_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'black', linestyle='--', 
             alpha=0.7, linewidth=2, label='Perfect correlation (y=x)')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(lang1_advantage_values, lang2_advantage_values)
    line_x = np.array([min_val, max_val])
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, 'g-', alpha=0.8, linewidth=2, 
             label=f'Regression line (R²={r_value**2:.3f})')      
    
    plt.xlabel(f'{lang1.upper()} {mod1.capitalize()} Advantage (F1 {mod1} - F1 {mod2})', fontsize=12)
    plt.ylabel(f'{lang2.upper()} {mod1.capitalize()} Advantage (F1 {mod1} - F1 {mod2})', fontsize=12)
    plt.title(f'{mod1.capitalize()} vs {mod2.capitalize()} Advantage: {lang1.upper()} vs {lang2.upper()}\n'
              f'Pearson r = {pearson_corr:.3f}, p = {pearson_p:.3f}\n'
              f'Spearman r = {spearman_corr:.3f}, p = {spearman_p:.3f}', fontsize=14)

  
    legend_elements = [
        plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.7, linewidth=2, 
                   label='Perfect correlation'),
        plt.Line2D([0], [0], color='g', alpha=0.8, linewidth=2, 
                   label=f'Regression line (R²={r_value**2:.3f})'),
        Patch(facecolor='darkblue', alpha=0.7, label='Both languages: Audio dominant'),
        Patch(facecolor='darkred', alpha=0.7, label='Both languages: Text dominant'),
        Patch(facecolor='lightblue', alpha=0.7, label=f'{lang1}: Audio, {lang2}: Text'),
        Patch(facecolor='lightcoral', alpha=0.7, label=f'{lang1}: Text, {lang2}: Audio'),
        Patch(facecolor='gray', alpha=0.7, label='Neutral/Mixed')
    ]
    plt.legend(handles=legend_elements, fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Pattern Analysis ===")
    print(f"Pearson correlation: {pearson_corr:.3f}, p-value: {pearson_p:.3f}")
    print(f"Spearman correlation: {spearman_corr:.3f}, p-value: {spearman_p:.3f}")
    print(f"Dimension groups:")
    for group, dims in dimension_groups.items():
        print(f"[{group}: {len(dims)} dimensions]")
        for i, dim in enumerate(dims):
            print(f"{i+1}. {dim} (Color: {colors[i]})")

    
    return pearson_corr, pearson_p, spearman_corr, spearman_p, dimension_groups


def main():
    json_path = 'semdim_stat.json'
    data = load_json(json_path)
    df = convert_to_dataframe(data)
        
    df = df.drop(df[df['lang'] =='all'].index, axis=0)
    df = df.drop(df[df['lang'] == 'all_sum'].index, axis=0)

    print_stats(df)

    modality_advantage_correlation(df, lang1='constructed', lang2='natural', mod1='audio', mod2='original')




