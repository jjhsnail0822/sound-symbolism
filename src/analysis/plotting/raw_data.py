import matplotlib.pyplot as plt

# 1) Set languages and labels
langs = ['en', 'fr', 'ja', 'ko']
labels = ['EN', 'FR', 'JA', 'KO']

# 2) Use predefined counts for each language
counts = [853, 1049, 1418, 4999]

# 3) Plot style settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

fig, ax = plt.subplots(figsize=(8, 8))
bars = ax.bar(labels, counts, color=['#4E79A7', '#F28E2B', '#E15759', '#76B7B2'], edgecolor='black')

# 4) Axes, title, and grid
ax.set_title('Word-Meaning Pair Counts by Language', pad=15)
ax.set_xlabel('Language')
ax.set_ylabel('Count of Word-Meaning Pairs')
ax.grid(axis='y', linestyle='--', alpha=0.5)

max_count = max(counts)
ax.set_ylim(0, max_count * 1.1)

# 5) Annotate bar heights
for bar in bars:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        h + max_count * 0.02,
        f'{h}',
        ha='center', va='bottom'
    )

plt.tight_layout()
plt.savefig('results/plots/word_meaning_counts.png', dpi=300)
plt.show()