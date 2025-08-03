import json
from collections import defaultdict
import pandas as pd

# JSON file path
JSON_FILE_PATH = 'data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json'
ART_JSON_FILE_PATH = 'data/processed/art/semantic_dimension/semantic_dimension_binary_gt.json'

# Define the poles for each dimension. The first element is labeled '1', the second is '2'.
DIMENSION_POLES = {
    "good-bad": ["good", "bad"],
    "beautiful-ugly": ["beautiful", "ugly"],
    "pleasant-unpleasant": ["pleasant", "unpleasant"],
    "strong-weak": ["strong", "weak"],
    "big-small": ["big", "small"],
    "rugged-delicate": ["rugged", "delicate"],
    "active-passive": ["active", "passive"],
    "fast-slow": ["fast", "slow"],
    "sharp-round": ["sharp", "round"],
    "realistic-fantastical": ["realistic", "fantastical"],
    "structured-disorganized": ["structured", "disorganized"],
    "ordinary-unique": ["ordinary", "unique"],
    "interesting-uninteresting": ["interesting", "uninteresting"],
    "simple-complex": ["simple", "complex"],
    "abrupt-continuous": ["abrupt", "continuous"],
    "exciting-calming": ["exciting", "calming"],
    "hard-soft": ["hard", "soft"],
    "happy-sad": ["happy", "sad"],
    "harsh-mellow": ["harsh", "mellow"],
    "heavy-light": ["heavy", "light"],
    "inhibited-free": ["inhibited", "free"],
    "masculine-feminine": ["masculine", "feminine"],
    "solid-nonsolid": ["solid", "nonsolid"],
    "tense-relaxed": ["tense", "relaxed"],
    "dangerous-safe": ["dangerous", "safe"]
}

def analyze_dimension_distribution(nat_file_path, art_file_path):
    """
    Analyzes the distribution of answers for each dimension from JSON files.
    """
    # Initialize a dictionary to store data by word_group
    stats = {
        'common': defaultdict(lambda: {'1': 0, '2': 0, 'total': 0}),
        'rare': defaultdict(lambda: {'1': 0, '2': 0, 'total': 0}),
        'constructed': defaultdict(lambda: {'1': 0, '2': 0, 'total': 0}),
        'total': defaultdict(lambda: {'1': 0, '2': 0, 'total': 0})
    }

    def process_file(file_path, is_art_file=False):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Iterate over all languages and words
        for lang in data:
            for item in data[lang]:
                if 'dimensions' not in item or not item['dimensions']:
                    continue

                # If word_group is missing, treat it as 'common'
                if is_art_file:
                    word_group = 'constructed'
                else:
                    word_group = item.get('word_group', 'common')

                for dim_name, dim_data in item['dimensions'].items():
                    if dim_name in DIMENSION_POLES:
                        poles = DIMENSION_POLES[dim_name]
                        answer = dim_data.get('answer')

                        if answer == poles[0]:
                            label = '1'
                        elif answer == poles[1]:
                            label = '2'
                        else:
                            continue

                        # Update word_group and total statistics
                        stats[word_group][dim_name][label] += 1
                        stats[word_group][dim_name]['total'] += 1
                        stats['total'][dim_name][label] += 1
                        stats['total'][dim_name]['total'] += 1

    process_file(nat_file_path, is_art_file=False)
    process_file(art_file_path, is_art_file=True)
    
    # Create a 'natural' group by combining 'common' and 'rare'
    stats['natural'] = defaultdict(lambda: {'1': 0, '2': 0, 'total': 0})
    for dim_name in DIMENSION_POLES:
        for group in ['common', 'rare']:
            if dim_name in stats[group]:
                stats['natural'][dim_name]['1'] += stats[group][dim_name]['1']
                stats['natural'][dim_name]['2'] += stats[group][dim_name]['2']
                stats['natural'][dim_name]['total'] += stats[group][dim_name]['total']

    return stats

def create_distribution_table(stats):
    """
    Creates a pandas DataFrame from the analyzed statistics.
    """
    table_data = []
    for group in ['natural', 'constructed', 'total']:
        for dim_name, poles in DIMENSION_POLES.items():
            # Use .get() to use a default value (0) if data is missing
            dim_stats = stats[group].get(dim_name, {'1': 0, '2': 0, 'total': 0})
            
            table_data.append({
                'Word Group': group.capitalize(),
                'Dimension': dim_name,
                'Count 1': dim_stats['1'],
                'Count 2': dim_stats['2'],
                'Total': dim_stats['total']
            })
    
    df = pd.DataFrame(table_data)
    return df

def create_comparison_table(df):
    """
    Creates a single comparison table for Common, Rare, and Total groups.
    """
    # Pivot the table to have dimensions as index and groups as columns, without sorting the index.
    pivot_df = df.pivot_table(
        index='Dimension', 
        columns='Word Group', 
        values=['Count 1', 'Count 2', 'Total'],
        aggfunc='sum',
        sort=False
    )
    
    # Swap the column levels to bring Word Group to the top
    pivot_df = pivot_df.swaplevel(0, 1, axis=1)
    
    # Sort the columns to group by Word Group (Natural, Constructed, Total)
    pivot_df = pivot_df.reindex(columns=['Natural', 'Constructed', 'Total'], level=0)
    
    return pivot_df

if __name__ == '__main__':
    # Analyze data
    statistics = analyze_dimension_distribution(JSON_FILE_PATH, ART_JSON_FILE_PATH)
    
    # Create DataFrame
    distribution_df = create_distribution_table(statistics)
    
    # Create a single comparison table
    comparison_table = create_comparison_table(distribution_df)

    # Print the comparison table to the terminal
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print("\n--- Semantic Dimension Distribution Comparison ---")
        print(comparison_table)
        # print total for each word group
        print("\n--- Total Counts by Word Group ---")
        print(comparison_table.sum())