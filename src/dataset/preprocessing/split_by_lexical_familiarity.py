import json
import os # For path manipulation

# --- Helper function to load JSON data from a file ---
def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- Helper function to save JSON data to a file ---
def save_json_file(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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

# 2. Initialize lists for common and rare items
# "rare": infinigram_count's 'total' <= 25
# "common": Words that are not rare (infinigram_count's 'total' > 25)
common_items = []
rare_items = []

datasets = {
    'en': en_data,
    'fr': fr_data,
    'ja': ja_data,
    'ko': ko_data
}

# 3. Process and classify data for each language
for lang_code, data in datasets.items():
    for item in data:
        processed_item = item.copy()
        processed_item['language'] = lang_code
        
        # Extract 'total' count from 'infinigram_count'
        total_count = item['infinigram_count']['total']

        # Classify as rare or common
        if total_count <= 25:
            rare_items.append(processed_item)
        else:
            common_items.append(processed_item)

# 4. Save the classified data to new JSON files
output_common_file = 'data/processed/nat/common_words.json'
output_rare_file = 'data/processed/nat/rare_words.json'

save_json_file(output_common_file, common_items)
save_json_file(output_rare_file, rare_items)

print(f"\nProcessed {len(common_items)} common items and saved to {output_common_file}")
print(f"Processed {len(rare_items)} rare items and saved to {output_rare_file}")