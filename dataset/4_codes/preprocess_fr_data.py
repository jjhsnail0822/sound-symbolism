import json
import os
import argparse
from collections import defaultdict, Counter

def check_word_in_meaning(data):
    """단어가 의미 안에 포함되어 있는지 확인"""
    problematic_items = []
    
    for item in data:
        word = item.get('word', '').strip().lower()
        meanings = item.get('meaning', [])
        
        # 한 글자 단어는 검사에서 제외
        if len(word) <= 1:
            continue
            
        for meaning in meanings:
            # 의미가 문자열인지 확인
            if not isinstance(meaning, str):
                continue
                
            # 단어가 의미 안에 포함되어 있는지 확인 (대소문자 구분 없이)
            if word in meaning.lower():
                problematic_items.append({
                    'word': word,
                    'meaning': meaning,
                    'full_item': item
                })
                break
    
    return problematic_items

def process_fr_data(input_file=None, output_file=None):
    # 기본 파일 경로
    if input_file is None:
        input_file = "../1_preprocess/nat/fr.json"
    if output_file is None:
        output_file = "../1_preprocess/nat/fr_processed.json"
    
    # 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"Error: 파일을 찾을 수 없습니다: {input_file}")
        return
    
    # JSON 파일 읽기
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: JSON 파일 형식이 올바르지 않습니다: {input_file}")
        print(f"상세 오류: {e}")
        return
    except Exception as e:
        print(f"Error: 파일을 읽는 중 오류가 발생했습니다: {e}")
        return
    
    print(f"원본 데이터 항목 수: {len(data)}")
    
    # 단어가 의미 안에 포함되어 있는지 확인
    problematic_items = check_word_in_meaning(data)
    if problematic_items:
        print(f"\n=== 단어가 의미 안에 포함된 항목 ({len(problematic_items)}개) ===")
        for i, item in enumerate(problematic_items, 1):
            print(f"{i}. 단어: '{item['word']}', 의미: '{item['meaning']}'")
            if i >= 20:  # 처음 20개만 출력
                print(f"... 외 {len(problematic_items) - 20}개")
                break
        
        # 문제가 있는 단어 목록
        problem_words = set(item['word'] for item in problematic_items)
        print(f"\n문제가 있는 단어 목록 ({len(problem_words)}개): {', '.join(sorted(problem_words))}")
    else:
        print("\n단어가 의미 안에 포함된 항목이 없습니다.")
    
    # 단어별로 항목 그룹화
    word_groups = defaultdict(list)
    for item in data:
        word = item.get('word', '')
        if word:
            word_groups[word].append(item)
    
    # 처리된 데이터를 저장할 리스트
    processed_data = []
    
    # 단어별로 처리
    for word, items in word_groups.items():
        # 의미가 없는 항목 필터링
        valid_items = [item for item in items if item.get('meaning') and len(item.get('meaning', [])) > 0]
        
        if not valid_items:
            continue
        
        # 의미가 가장 많은 항목 선택
        if len(valid_items) > 1:
            # 의미 수에 따라 정렬
            valid_items.sort(key=lambda x: len(x.get('meaning', [])), reverse=True)
            # 가장 의미가 많은 항목 선택
            processed_data.append(valid_items[0])
        else:
            # 항목이 하나뿐이면 그대로 추가
            processed_data.append(valid_items[0])
    
    # 통계 계산
    total_words = len(processed_data)
    total_meanings = sum(len(item.get('meaning', [])) for item in processed_data)
    
    # URL별 의미 수 계산
    url_stats = defaultdict(int)
    for item in processed_data:
        url = item.get('url', 'unknown')
        meanings_count = len(item.get('meaning', []))
        url_stats[url] += meanings_count
    
    # 결과 출력
    print(f"\n=== 처리 결과 통계 ===")
    print(f"총 단어 수: {total_words}")
    print(f"총 의미 수: {total_meanings}")
    print(f"단어당 평균 의미 수: {total_meanings / total_words:.2f}")
    
    print(f"\n=== URL별 의미 수 ===")
    for url, count in sorted(url_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"{url}: {count}개 의미 ({count / total_meanings * 100:.2f}%)")
    
    # 처리된 데이터 저장
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"\n처리된 데이터가 저장되었습니다: {output_file}")
    except Exception as e:
        print(f"Error: 파일을 저장하는 중 오류가 발생했습니다: {e}")
        return
    
    # 중복 단어 통계
    duplicate_words = sum(1 for items in word_groups.values() if len(items) > 1)
    print(f"\n중복된 단어 수: {duplicate_words}")
    print(f"제거된 항목 수: {len(data) - total_words}")
    
    # 의미 수 분포
    meaning_counts = [len(item.get('meaning', [])) for item in processed_data]
    meaning_counter = Counter(meaning_counts)
    
    print(f"\n=== 의미 수 분포 ===")
    for count, freq in sorted(meaning_counter.items()):
        print(f"의미 {count}개: {freq}개 단어 ({freq / total_words * 100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='프랑스어 의성어/의태어 데이터 전처리')
    parser.add_argument('--input', help='입력 JSON 파일 경로')
    parser.add_argument('--output', help='출력 JSON 파일 경로')
    
    args = parser.parse_args()
    process_fr_data(args.input, args.output) 