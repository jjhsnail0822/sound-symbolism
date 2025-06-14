import json
import statistics # For median calculation
import os # For path manipulation

# --- Helper function to load JSON data from a file ---
def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- Helper function to save JSON data to a file ---
def save_json_file(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- Helper function to calculate a specific quantile of infinigram_count ---
def get_quantile_infinigram_count(data_list, quantile_n=4, quantile_index=0):
    """
    Calculates a specific quantile of 'infinigram_count' from a list of dictionaries.
    For Q1 (25th percentile), use quantile_n=4, quantile_index=0.
    Returns 0 if the list is empty or counts cannot be extracted.
    """
    if not data_list:
        return 0
    
    counts = []
    for item in data_list:
        # Ensure 'infinigram_count' exists and is a number
        count = item.get('infinigram_count')
        if isinstance(count, (int, float)):
            counts.append(count)
        # else:
            # Optionally, handle items without 'infinigram_count' or with invalid types
            # print(f"Warning: Item without valid 'infinigram_count': {item}")
            
    if not counts: # If no valid counts were found
        return 0
    
    # Ensure there are enough data points for quantiles if n > 1
    if len(counts) < quantile_n and quantile_n > 1: # For quantiles, need at least n items for n-quantiles
        # Fallback or specific handling for small datasets, e.g., return min, max, or median
        # For simplicity, returning median if not enough data for specified quantiles
        return statistics.median(counts) if counts else 0

    return statistics.quantiles(counts, n=quantile_n)[quantile_index]

# --- Main script ---

base_data_path = 'data/processed/nat'

# 1. Load data from JSON files
en_file_path = os.path.join(base_data_path, 'en.json')
fr_file_path = os.path.join(base_data_path, 'fr.json')
ja_file_path = os.path.join(base_data_path, 'ja.json')
ko_file_path = os.path.join(base_data_path, 'ko.json')

en_data = load_json_file(en_file_path)
fr_data = load_json_file(fr_file_path)
ja_data = load_json_file(ja_file_path)
ko_data = load_json_file(ko_file_path)

# 2. Calculate Q1 (25th percentile) for each language
q1_en = get_quantile_infinigram_count(en_data, quantile_n=4, quantile_index=0)
q1_fr = get_quantile_infinigram_count(fr_data, quantile_n=4, quantile_index=0)
q1_ja = get_quantile_infinigram_count(ja_data, quantile_n=4, quantile_index=0)
q1_ko = get_quantile_infinigram_count(ko_data, quantile_n=4, quantile_index=0)

print(f"Q1 (25th percentile) for English: {q1_en}")
print(f"Q1 (25th percentile) for French: {q1_fr}")
print(f"Q1 (25th percentile) for Japanese: {q1_ja}")
print(f"Q1 (25th percentile) for Korean: {q1_ko}")

# 3. Initialize lists for common and rare items
# "common" will now mean above Q1, "rare" will mean at or below Q1
more_frequent_items = [] # Renamed from common_items for clarity
less_frequent_items = [] # Renamed from rare_items for clarity

# 4. Process and classify data for each language
datasets = {
    'en': (en_data, q1_en),
    'fr': (fr_data, q1_fr),
    'ja': (ja_data, q1_ja),
    'ko': (ko_data, q1_ko)
}

for lang_code, (data, q1_val) in datasets.items():
    for item in data:
        # Create a copy to avoid modifying the original data_list items in memory
        processed_item = item.copy() 
        
        # Add the language field
        processed_item['language'] = lang_code
        
        # Get infinigram_count, default to 0 if not present or not a number
        current_count = processed_item.get('infinigram_count', 0)
        if not isinstance(current_count, (int, float)):
            current_count = 0 # Treat non-numeric or missing as 0 for comparison

        # Classify based on Q1
        if current_count > q1_val:
            more_frequent_items.append(processed_item)
        else: # count <= q1_val
            less_frequent_items.append(processed_item)

# 5. Save the classified data to new JSON files in the current working directory
#    or specify a different path if needed.
output_more_frequent_file = 'data/processed/nat/common_words.json' # Updated filename
output_less_frequent_file = 'data/processed/nat/rare_words.json' # Updated filename

save_json_file(output_more_frequent_file, more_frequent_items)
save_json_file(output_less_frequent_file, less_frequent_items)

print(f"\nProcessed {len(more_frequent_items)} items above Q1 and saved to {output_more_frequent_file}")
print(f"Processed {len(less_frequent_items)} items at or below Q1 and saved to {output_less_frequent_file}")