import json

def filter_clusters_and_select_top_words(input_filepath, output_filepath):
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_filepath}")
        return

    processed_clusters = []
    required_languages_list = ['en', 'fr', 'ja', 'ko']
    required_languages_set = set(required_languages_list)

    for cluster_obj in data:
        words_in_cluster = cluster_obj.get("words", [])
        
        top_word_by_lang = {}

        lang_candidates = {lang: [] for lang in required_languages_set}
        for word_data in words_in_cluster:
            lang = word_data.get("language")
            if lang in required_languages_set:
                if isinstance(word_data.get("infinigram_count"), (int, float)):
                    lang_candidates[lang].append(word_data)

        all_required_languages_present = True
        for lang_code in required_languages_list:
            if not lang_candidates[lang_code]:
                all_required_languages_present = False
                break
            best_word_for_lang = max(lang_candidates[lang_code], key=lambda x: x.get("infinigram_count", -1))
            top_word_by_lang[lang_code] = best_word_for_lang
            
        if all_required_languages_present:
            new_cluster = {key: value for key, value in cluster_obj.items() if key != "words"}
            new_cluster["words"] = [top_word_by_lang[lang] for lang in required_languages_list] # 정의된 순서대로 단어 추가
            
            processed_clusters.append(new_cluster)

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(processed_clusters, f, ensure_ascii=False, indent=4)
        print(f"Processed data saved to {output_filepath}")
    except IOError:
        print(f"Error: Could not write to output file at {output_filepath}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return

input_file = "dataset/1_preprocess/nat/crosslingual_with_en/crosslingual_clustered_with_count.json"
output_file = "dataset/1_preprocess/nat/crosslingual_with_en/crosslingual_clustered_filtered_top_words.json"

filter_clusters_and_select_top_words(input_file, output_file)