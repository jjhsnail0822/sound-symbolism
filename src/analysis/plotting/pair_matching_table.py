import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

# 1) Collect result file paths
exp_dir = os.path.abspath(os.path.join(__file__, '..', '..', 'experiments', 'understanding'))
files = glob.glob(os.path.join(exp_dir, 'all_results_*.json'))

# 2) Load data into dict
models = []
cats = []
data = {}
for fp in files:
    arr = json.load(open(fp, 'r', encoding='utf-8'))
    for rec in arr:
        m = rec['model'].split('/')[-1]
        dp = os.path.basename(rec['data_path'])
        lang = dp.split('-')[-1].split('.')[0].upper()
        t = 'W→M' if 'unmasked' in dp else 'M→W'
        cat = f'{lang} {t}'
        data.setdefault(m, {})[cat] = rec['accuracy'] * 100
        if m not in models: models.append(m)
        if cat not in cats: cats.append(cat)

# 3) Sort models and categories
desired_order = [
    'gemma-3-4b-it','gemma-3-12b-it','gemma-3-27b-it',
    'Qwen2.5-3B-Instruct','Qwen2.5-7B-Instruct','Qwen2.5-14B-Instruct',
    'Qwen2.5-32B-Instruct','Qwen2.5-72B-Instruct',
    'gpt-4o','gpt-4.1','o4-mini'
]
models = [m for m in desired_order if m in models] + [m for m in models if m not in desired_order]
cats = sorted(cats, key=lambda x: (x.split()[0], x.split()[1]))

# 4) Build DataFrame
df = pd.DataFrame(index=models, columns=cats)
for m in models:
    for c in cats:
        df.at[m, c] = data.get(m, {}).get(c, float('nan'))
df = df.round(1)

# Convert numeric values to strings with one decimal place
df_str = df.applymap(lambda x: f"{x:.1f}" if pd.notna(x) else "")

# 5) Group columns by task and add subtotals
# split original columns into word→meaning and meaning→word
w2m = [c for c in cats if 'W→M' in c]
m2w = [c for c in cats if 'M→W' in c]

# extract language codes
langs = sorted({c.split()[0] for c in cats})

# build new DataFrame with MultiIndex columns
parts = []
# W→M block
df_w2m = df[w2m]
parts.append(df_w2m[ [f"{lang} W→M" for lang in langs] ])
parts.append(pd.DataFrame(df_w2m.mean(axis=1), columns=['Avg W→M']))
# M→W block
df_m2w = df[m2w]
parts.append(df_m2w[ [f"{lang} M→W" for lang in langs] ])
parts.append(pd.DataFrame(df_m2w.mean(axis=1), columns=['Avg M→W']))

df2 = pd.concat(parts, axis=1)

# create MultiIndex for columns, grouping M→W block first, then W→M
tuples = []
# M→W block: per-language then average
for sub in langs:
    tuples.append(('M→W', sub))
tuples.append(('M→W', 'Avg'))
# W→M block: per-language then average
for sub in langs:
    tuples.append(('W→M', sub))
tuples.append(('W→M', 'Avg'))
df2.columns = pd.MultiIndex.from_tuples(tuples)

# convert to string with one decimal
df2_str = df2.applymap(lambda x: f"{x:.1f}" if pd.notna(x) else "")

# 6) Plot as table
fig, ax = plt.subplots(figsize=(df2.shape[1]*0.8, len(models)*0.3))
ax.axis('off')
tbl = ax.table(
    cellText=df2_str.values,
    rowLabels=df2.index,
    colLabels=df2.columns,
    cellLoc='center',
    loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
tbl.scale(1, 1.5)

plt.title('Word‑Meaning Pair Matching Accuracy (%)', pad=20)
plt.tight_layout()
out_png = 'results/plots/word_meaning_pair_matching_table.png'
plt.savefig(out_png, dpi=300)
plt.show()