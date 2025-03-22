import csv
import json
import os
import re

DIR_PATH = 'dataset/0_raw/nat/standard_kor_dic_20250306'
files_dir = os.listdir(DIR_PATH)
# remove hidden files
files_dir = [f for f in files_dir if not f.startswith('.')]
# sort files by filename '0000_number.json'
files_dir.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

# open all json files in the directory
ds = []
for filename in files_dir:
    with open(f'{DIR_PATH}/{filename}', 'r', encoding='utf-8') as f:
        data = json.load(f)
        ds.extend(data['channel']['item'])

def extract_definition_list(d):
    definitions = []
    for pos_info in d['word_info']['pos_info']:
        for comm_pattern_info in pos_info['comm_pattern_info']:
            for sense_info in comm_pattern_info['sense_info']:
                definition = sense_info['definition']
                definition = definition.split(' ‘')[0].strip() # remove sentences like '‘가득01「1」’보다 센 느낌을 준다.'
                definitions.append(definition)
    return definitions

def is_onomatopoeia(original_word, definition_list):
    invalid_word_list = ['-이', '-히']
    valid_def_list = [' 소리.', ' 모양.', ' 상태.', ' 느낌.'] # onomatopoeia: 소리, ideophone: 모양, 상태, 느낌 according to the Korean dictionary publishing standard
    
    if any(invalid in original_word for invalid in invalid_word_list):
        return False
    for definition in definition_list:
        if any(valid in definition for valid in valid_def_list):
            return True
    return False

onomatopoeias = []
for d in ds:
    if d['word_info']['pos_info'][0]['pos'] == '부사' and d['word_info']['word_type'] == '고유어':
        definitions = extract_definition_list(d)
        if is_onomatopoeia(d['word_info']['word'], definitions):
            word = d['word_info']['word'].replace('-', '') # remove '-' e.g., '가랑-가랑'
            word = re.sub(r'\d+', '', word)  # remove numbers e.g., '가랑가랑01'
            if onomatopoeias and onomatopoeias[-1]['word'] == word:
                onomatopoeias[-1]['definitions'].extend(definitions) # merge definitions if same word
            else:
                onomatopoeias.append({'word': word, 'definitions': definitions}) # add new word

# save to csv
with open('dataset/0_raw/nat/ko.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'meaning', 'ref'])
    for o in onomatopoeias:
        writer.writerow([o['word'], o['definitions'][0], '표준국어대사전 20250306'])

for o in onomatopoeias:
    o['url'] = 'https://stdict.korean.go.kr'
    o['found'] = True
    o['meaning'] = o['definitions'][0]
    o['ref'] = '표준국어대사전 20250306'

# save to json
with open('dataset/1_preprocess/nat/ko_past.json', 'w', encoding='utf-8') as f:
    json.dump(onomatopoeias, f, ensure_ascii=False, indent=4)