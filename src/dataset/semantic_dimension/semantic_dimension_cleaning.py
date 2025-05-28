import json
import os
import glob
from collections import defaultdict

def analyze_feature_concordance_across_files(file_paths):
    """
    Analyzes how many files show consistent values for each semantic feature,
    both globally and on a per-language basis.

    Args:
        file_paths (list): List of JSON file paths to analyze.

    Returns:
        tuple: (global_agreement_counts, global_total_features,
                per_lang_agreement_counts, per_lang_total_unique_features,
                file_errors)
            - global_agreement_counts (defaultdict(int)):
                Key is the number of agreeing files, value is the count of unique features
                globally having that agreement number.
            - global_total_features (int): Total number of unique features analyzed globally.
            - per_lang_agreement_counts (defaultdict(lambda: defaultdict(int))):
                Outer key is language code. Inner dict's key is the number of
                agreeing files, value is the count of unique features for that language.
            - per_lang_total_unique_features (defaultdict(int)):
                Key is language code, value is the total number of unique features
                for that language.
            - file_errors (list): List of files that caused errors during loading or processing.
    """
    all_feature_occurrences = defaultdict(lambda: defaultdict(set))
    loaded_file_count = 0
    file_errors = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            loaded_file_count += 1
            file_basename = os.path.basename(file_path)

            for lang, entries in data.items():
                for item in entries:
                    if isinstance(item, dict) and 'word' in item and 'dimensions' in item and isinstance(item['dimensions'], dict):
                        word = item['word']
                        for dim_key, dim_value in item['dimensions'].items():
                            feature_signature = (lang, word, dim_key)
                            all_feature_occurrences[feature_signature][dim_value].add(file_basename)
        except FileNotFoundError:
            # Keep Korean error messages for console output during script run as per original
            print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
            file_errors.append(file_path)
        except json.JSONDecodeError:
            print(f"오류: 잘못된 JSON 형식 - {file_path}")
            file_errors.append(file_path)
        except Exception as e:
            print(f"오류: '{file_path}' 처리 중 예기치 않은 오류 발생 - {e}")
            file_errors.append(file_path)

    if loaded_file_count == 0:
        return defaultdict(int), 0, defaultdict(lambda: defaultdict(int)), defaultdict(int), file_errors

    global_agreement_counts = defaultdict(int)
    per_lang_agreement_counts = defaultdict(lambda: defaultdict(int))
    per_lang_total_unique_features = defaultdict(int)
    
    global_total_features = len(all_feature_occurrences)

    for feature_signature, value_to_files_map in all_feature_occurrences.items():
        lang, _, _ = feature_signature
        per_lang_total_unique_features[lang] += 1
        
        if not value_to_files_map:
            continue

        max_files_for_this_feature = 0
        for dim_value, files_set in value_to_files_map.items():
            if len(files_set) > max_files_for_this_feature:
                max_files_for_this_feature = len(files_set)
        
        if max_files_for_this_feature > 0:
            global_agreement_counts[max_files_for_this_feature] += 1
            per_lang_agreement_counts[lang][max_files_for_this_feature] += 1
            
    return global_agreement_counts, global_total_features, per_lang_agreement_counts, per_lang_total_unique_features, file_errors


if __name__ == "__main__":
    directory_path = 'data/processed/nat/semantic_dimension/'
    json_files = glob.glob(os.path.join(directory_path, '*.json'))
    num_total_files = len(json_files)

    if num_total_files < 1:
        print(f"No JSON files to analyze in '{directory_path}'.")
    else:
        print(f"Starting semantic feature concordance analysis for {num_total_files} JSON files in '{directory_path}'...\n")
        
        g_agreement, g_total_feat, pl_agreement, pl_total_feat, errors = analyze_feature_concordance_across_files(json_files)

        if errors:
            print(f"\nNote: Errors occurred while processing the following {len(errors)} file(s):")
            for err_file in errors:
                print(f"  - {err_file}")
            print("")

        analyzed_file_count = num_total_files - len(errors)

        # --- Global Summary ---
        if g_total_feat == 0 and not errors:
             print("No unique semantic features were analyzed globally.")
        elif g_total_feat > 0:
            print(f"Analyzed a total of {g_total_feat} unique semantic features globally (language, word, dimension key combinations).")
            print("=" * 70)
            print("Global distribution of semantic feature value agreement across files:")
            print("=" * 70)

            if analyzed_file_count <= 0 and g_total_feat > 0:
                print("No files could be effectively analyzed for global statistics due to errors.")
            
            for num_agreed_files in sorted(g_agreement.keys(), reverse=True):
                count = g_agreement[num_agreed_files]
                percentage = (count / g_total_feat) * 100 if g_total_feat > 0 else 0
                
                if num_agreed_files == 1:
                    print(f"- Number of features with unique values or appearing in only {num_agreed_files} file: {count} ({percentage:.2f}%)")
                else:
                    print(f"- Number of features with values agreeing across {num_agreed_files} files: {count} ({percentage:.2f}%)")
            
            print("=" * 70)
            if analyzed_file_count > 0:
                 print(f"(Total number of files targeted for global analysis: {analyzed_file_count})")
            print("\n")


        elif errors and g_total_feat == 0:
            print("No semantic features were analyzed globally due to file loading errors.")
            print("\n")

        # --- Per-Language Summary ---
        if not pl_total_feat and not errors and g_total_feat == 0 : # If global was also empty and no errors
            pass # Message already printed by global summary
        elif not pl_total_feat and errors and g_total_feat == 0:
             pass # Message already printed by global summary
        elif pl_total_feat:
            print("--- Per-Language Semantic Feature Concordance Analysis ---")
            for lang in sorted(pl_total_feat.keys()):
                lang_total_features = pl_total_feat[lang]
                lang_agreement_counts = pl_agreement[lang]

                print(f"\n-- Language: {lang} --")
                if lang_total_features == 0:
                    print(f"No unique semantic features were analyzed for language: {lang}.")
                    continue

                print(f"Analyzed a total of {lang_total_features} unique semantic features for language: {lang}.")
                print("-" * 50)
                print(f"Distribution for {lang}:")
                print("-" * 50)

                for num_agreed_files in sorted(lang_agreement_counts.keys(), reverse=True):
                    count = lang_agreement_counts[num_agreed_files]
                    percentage = (count / lang_total_features) * 100 if lang_total_features > 0 else 0
                    
                    if num_agreed_files == 1:
                        print(f"- Features unique or in {num_agreed_files} file: {count} ({percentage:.2f}%)")
                    else:
                        print(f"- Features agreeing across {num_agreed_files} files: {count} ({percentage:.2f}%)")
                print("-" * 50)
                if analyzed_file_count > 0:
                    print(f"(Based on {analyzed_file_count} successfully loaded files)")
        
        if not g_total_feat and not pl_total_feat and errors: # If both global and per-lang are empty due to errors
            print("No features could be analyzed due to errors in all files.")