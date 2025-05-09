import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# EXP_NAME = 'word_meaning_pair_matching'
# TITLE = 'Word-Meaning Pair Matching'
EXP_NAME = 'word_meaning_pair_matching_no_dialogue'
TITLE = 'Word-Meaning Pair Matching without Dialogue'

# 1) Collect result file paths
exp_dir = f'analysis/experiments/understanding/{EXP_NAME}'
files = glob.glob(os.path.join(exp_dir, 'all_results_*.json'))

# 2) Data structure: models, categories, and accuracies
models = []
cats = []
data = {}  # data[model][cat] = accuracy

for fp in files:
    arr = json.load(open(fp, 'r', encoding='utf-8'))
    for rec in arr:
        m = rec['model'].split('/')[-1]
        dp = os.path.basename(rec['data_path'])
        # Category name: language + task type
        lang = dp.split('-')[-1].split('.')[0].upper()
        t = '→'.join(['Word','Meaning']) if 'unmasked' in dp else '→'.join(['Meaning','Word'])
        cat = f'{lang} {t}'
        if m not in data:
            data[m] = {}
        data[m][cat] = rec['accuracy']
        if m not in models:
            models.append(m)
        if cat not in cats:
            cats.append(cat)

# Define a fixed model ordering
desired_order = [
    'gemma-3-4b-it',
    'gemma-3-12b-it',
    'gemma-3-27b-it',
    'Qwen2.5-3B-Instruct',
    'Qwen2.5-7B-Instruct',
    'Qwen2.5-14B-Instruct',
    'Qwen2.5-32B-Instruct',
    'Qwen2.5-72B-Instruct',
    'Qwen3-4B',
    'Qwen3-8B',
    'Qwen3-14B',
    'Qwen3-32B',
    'Qwen3-4B-thinking',
    'Qwen3-8B-thinking',
    'Qwen3-14B-thinking',
    'Qwen3-32B-thinking',
    'gpt-4o',
    'gpt-4.1',
    'o4-mini',
    'OLMo-2-1124-7B-Instruct',
    'OLMo-2-1124-13B-Instruct',
    'OLMo-2-0325-32B-Instruct',
]
# Sort models based on desired_order
models = [m for m in desired_order if m in models] + [m for m in models if m not in desired_order]

cats = sorted(cats, key=lambda x: (x.split()[0], x.split()[1]))

# 3) Split categories by task type
w2m_cats = [c for c in cats if 'Word→Meaning' in c]
m2w_cats = [c for c in cats if 'Meaning→Word' in c]

# 4) Plotting: create two subplots for Word→Meaning and Meaning→Word
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

fig.suptitle(f'{TITLE}', fontsize=16, y=0.98)

# Define color palettes for model families
gemma_colors = plt.get_cmap('Blues')([0.4, 0.6, 0.8]) # Lighter to darker blues
qwen25_colors = plt.get_cmap('Reds')([0.3, 0.5, 0.7, 0.9]) # Lighter to darker reds
qwen_colors = plt.get_cmap('Oranges')([0.3, 0.5, 0.7, 0.9]) # Lighter to darker oranges/browns
olmo_colors = plt.get_cmap('Purples')([0.4, 0.6, 0.8]) # Lighter to darker purples
gpt_colors = plt.get_cmap('Greens')([0.4, 0.6, 0.8]) # Lighter to darker greens
# Add more colors if needed, e.g., from 'Greens'
# qwen_colors = plt.get_cmap('Greens')([0.3, 0.5, 0.7, 0.9])

def plot_group(ax, subcats, title):
    width = 0.15
    gap = 0.2 # spacing between language groups
    num_models = len(models)
    group_width = width * num_models + gap
    x = np.arange(len(subcats)) * group_width
    gemma_idx = 0
    qwen25_idx = 0
    qwen_idx = 0
    qwen_thinking_idx = 0
    olmo_idx = 0
    gpt_idx = 0
    for i, m in enumerate(models):
        y = [data[m].get(c, np.nan) for c in subcats]
        # Assign color based on model family
        if 'gemma' in m.lower():
            color = gemma_colors[gemma_idx % len(gemma_colors)]
            gemma_idx += 1
        elif 'qwen2.5' in m.lower():
            color = qwen25_colors[qwen25_idx % len(qwen25_colors)]
            qwen25_idx += 1
        elif 'qwen3' in m.lower():
            color = qwen_colors[qwen_idx % len(qwen_colors)]
            qwen_idx += 1
        elif 'qwen3' in m.lower() and 'thinking' in m.lower():
            color = qwen_colors[qwen_thinking_idx % len(qwen_colors)]
            qwen_thinking_idx += 1
        elif 'olmo' in m.lower():
            color = olmo_colors[olmo_idx % len(olmo_colors)]
            olmo_idx += 1
        elif 'gpt' in m.lower() or 'o' in m.lower():
            color = gpt_colors[gpt_idx % len(gpt_colors)]
            gpt_idx += 1
        else:
            color = plt.get_cmap('tab10').colors[i % 10]

        bars = ax.bar(x + i*width, y, width, label=m, color=color)
        # Annotate each bar with its value as a percentage
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h + 0.005,
                f'{h*100:.1f}',
                ha='center', va='bottom',
                fontsize=4
            )
    ax.set_xticks(x + (width * num_models) / 2)
    ax.set_xticklabels([c.split()[0] for c in subcats], rotation=0, ha='center')
    ax.set_title(title)

plot_group(ax1, w2m_cats, 'Word→Meaning Accuracy')
plot_group(ax2, m2w_cats, 'Meaning→Word Accuracy')

# 5) Draw average lines for left/right 4 bars
avg_w2m = np.nanmean([data[m].get(c, np.nan) for m in models for c in w2m_cats])
avg_m2w = np.nanmean([data[m].get(c, np.nan) for m in models for c in m2w_cats])

ax1.axhline(avg_w2m, color='red', linestyle='--', linewidth=1)
ax1.text(
    0.01, avg_w2m - 0.01,  # Move text slightly below the line
    f'Avg {avg_w2m*100:.1f}%', color='red',
    transform=ax1.get_yaxis_transform(),
    va='top', ha='left', fontsize=6
)

ax2.axhline(avg_m2w, color='red', linestyle='--', linewidth=1)
ax2.text(
    0.01, avg_m2w - 0.01,  # Move text slightly below the line
    f'Avg {avg_m2w*100:.1f}%', color='red',
    transform=ax2.get_yaxis_transform(),
    va='top', ha='left', fontsize=6
)

for ax in (ax1, ax2):
    ax.set_ylabel('Accuracy (%)')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.02)

ax2.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at top for the suptitle
out_png = f'analysis/plots/{EXP_NAME}.png'
plt.savefig(out_png, dpi=300)
plt.show()