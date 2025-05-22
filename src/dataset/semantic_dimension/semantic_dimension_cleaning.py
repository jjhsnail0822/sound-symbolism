import json

def count_dimension_feature_mismatches(file_path1, file_path2):
    """
    두 JSON 파일 간에 각 언어별로 "dimensions" 내의 feature 값 불일치 개수와
    비교된 전체 feature 개수를 계산합니다.

    Args:
        file_path1 (str): 첫 번째 JSON 파일 경로.
        file_path2 (str): 두 번째 JSON 파일 경로.

    Returns:
        dict: 언어 코드를 키로, (불일치 feature 수, 전체 비교 feature 수) 튜플을 값으로 하는 딕셔너리.
              오류 발생 시 문자열 메시지를 반환합니다.
    """
    try:
        with open(file_path1, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        with open(file_path2, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
    except FileNotFoundError as e:
        return f"Error: File not found - {e.filename}"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format in one of the files - {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

    analysis_results = {}
    all_languages = sorted(list(set(data1.keys()) | set(data2.keys())))

    for lang in all_languages:
        lang_mismatch_features = 0
        lang_total_compared_features = 0
        
        list1 = data1.get(lang, [])
        list2 = data2.get(lang, [])

        words1_data = {}
        for item in list1:
            if isinstance(item, dict) and 'word' in item and 'dimensions' in item and isinstance(item['dimensions'], dict):
                words1_data[item['word']] = item['dimensions']

        words2_data = {}
        for item in list2:
            if isinstance(item, dict) and 'word' in item and 'dimensions' in item and isinstance(item['dimensions'], dict):
                words2_data[item['word']] = item['dimensions']

        common_words = set(words1_data.keys()) & set(words2_data.keys())

        if not common_words:
            analysis_results[lang] = (0, 0)
            continue

        for word in common_words:
            dims1 = words1_data[word]
            dims2 = words2_data[word]

            # 두 dimensions 객체의 모든 고유한 키들을 가져옵니다.
            all_dim_keys = set(dims1.keys()) | set(dims2.keys())
            
            for key in all_dim_keys:
                lang_total_compared_features += 1
                
                val1 = dims1.get(key) # 키가 없으면 None 반환
                val2 = dims2.get(key) # 키가 없으면 None 반환

                # 값이 다르거나 (None vs 값 포함), 한쪽에만 키가 존재하는 경우 불일치로 간주
                if val1 != val2:
                    lang_mismatch_features += 1
            
        analysis_results[lang] = (lang_mismatch_features, lang_total_compared_features)
        
    return analysis_results

if __name__ == "__main__":
    # 제공된 파일 경로
    file_path_qwen = 'data/processed/nat/semantic_dimension/semantic_dimension_gt_Qwen3-32B.json'
    file_path_gemma = 'data/processed/nat/semantic_dimension/semantic_dimension_gt_gemma-3-27b-it.json'

    results = count_dimension_feature_mismatches(file_path_qwen, file_path_gemma)

    if isinstance(results, str): # 오류 메시지가 반환된 경우
        print(results)
    else:
        print("Dimension feature mismatch analysis per language:")
        for lang, (mismatches, total) in results.items():
            if total > 0:
                percentage = (mismatches / total) * 100
                print(f"- {lang}: {mismatches} mismatches out of {total} features ({percentage:.2f}%)")
            else:
                # 공통 단어가 없거나, 공통 단어는 있지만 dimensions 필드가 없거나 비어있는 경우,
                # 또는 dimensions 필드는 있지만 그 안에 feature가 하나도 없는 경우.
                print(f"- {lang}: {mismatches} mismatches out of {total} features (No features to compare, or all dimensions empty/non-existent for common words)")