import csv
import json
import os
import pykakasi
import re
from tqdm import tqdm

DIR_PATH_NIKKOKU = 'dataset/0_raw/nat/seisenban_nikkoku' # 1089 onomatopoeias
DIR_PATH_KOJIEN = 'dataset/0_raw/nat/kojien_7ed' # 950 onomatopoeias
DIR_PATH_DAIJISEN = 'dataset/0_raw/nat/digital_daijisen' # 982 onomatopoeias
DIR_PATH_DAIJIRIN = 'dataset/0_raw/nat/daijirin_4ed' # 1081 onomatopoeias

kks = pykakasi.kakasi()
def hira2kata(s):
    return ''.join([kk['kana'] for kk in kks.convert(s)])

def convert_dash_to_hira(s):
    for i, char in enumerate(s):
        if char == 'ー' and i > 0:
            prev = s[i-1]
            if prev in 'あかさたなはまやらわがざだばぱぁゃ':
                s = s[:i] + 'あ' + s[i+1:]
            elif prev in 'いきしちにひみりぎじぢびぴぃ':
                s = s[:i] + 'い' + s[i+1:]
            elif prev in 'うくすつぬふむゆるぐずづぶぷぅゅ':
                s = s[:i] + 'う' + s[i+1:]
            elif prev in 'えけせてねへめれげぜでべぺぇ':
                s = s[:i] + 'え' + s[i+1:]
            elif prev in 'おこそとのほもよろをごぞどぼぽぉょ':
                s = s[:i] + 'お' + s[i+1:]
    return s

def open_dictionary(DIR_PATH):
    files_dir = os.listdir(DIR_PATH)
    # remove hidden files
    files_dir = [f for f in files_dir if not f.startswith('.')]
    # sort files by filename '0000_number.json'
    files_dir.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

    # open all json files in the directory
    ds = []
    for filename in files_dir:
        with open(f'{DIR_PATH}/{filename}', 'r', encoding='utf-8') as f:
            data = json.load(f)
            ds.extend(data)
    return ds

# open csv file
with open('dataset/0_raw/nat/ja.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    ja_words = [row[0] for row in reader]
    ja_words = ja_words[1:] # remove header

ds_nikkoku = open_dictionary(DIR_PATH_NIKKOKU)
ds_kojien = open_dictionary(DIR_PATH_KOJIEN)
ds_daijisen = open_dictionary(DIR_PATH_DAIJISEN)
ds_daijirin = open_dictionary(DIR_PATH_DAIJIRIN)

onomatopoeias = [{'word': word, 'nikkoku': [], 'kojien': [], 'daijisen': [], 'daijirin': []} for word in ja_words]

# fill in the onomatopoeias
def fill_onomatopoeias(ds, ds_name):
    mapping_hira = {}
    mapping_kata = {}
    mapping_hira_without_dash = {}
    mapping_kata_without_dash = {}
    progress = {}
    for o in onomatopoeias:
        mapping_hira[o['word']] = o
        mapping_kata[hira2kata(o['word'])] = o
        mapping_hira_without_dash[convert_dash_to_hira(o['word'])] = o
        mapping_kata_without_dash[hira2kata(convert_dash_to_hira(o['word']))] = o
        progress[o['word']] = 0
    conditions = [
        # idx, mapping, min_progress
        (0, mapping_hira, 0), # hiragana word
        (0, mapping_kata, 1), # katakana word
        (1, mapping_hira, 2), # hiragana pronunciation
        (1, mapping_kata, 3), # katakana pronunciation
        (0, mapping_hira_without_dash, 4), # hiragana word without dash
        (0, mapping_kata_without_dash, 5), # katakana word without dash
        (1, mapping_hira_without_dash, 6), # hiragana pronunciation without dash
        (1, mapping_kata_without_dash, 7), # katakana pronunciation without dash
    ]
    for d in ds:
        for idx, mapping, min_progress in conditions:
            word_val = d[idx]
            o = mapping.get(word_val)
            if o and progress[o['word']] <= min_progress:
                o[ds_name].extend(d[5])
                progress[o['word']] = min_progress + 1
                break
        # with 'と'
        for idx, mapping, min_progress in conditions:
            if not d[idx].endswith('と'):
                continue
            # remove 'と' from the end of the word
            word_val = d[idx].rstrip('と')
            o = mapping.get(word_val)
            if o and progress[o['word']] <= min_progress:
                o[ds_name].extend(d[5])
                progress[o['word']] = min_progress + 1
                break
    return

fill_onomatopoeias(ds_nikkoku, 'nikkoku')
fill_onomatopoeias(ds_kojien, 'kojien')
fill_onomatopoeias(ds_daijisen, 'daijisen')
fill_onomatopoeias(ds_daijirin, 'daijirin')

# use nikkoku as criteria
def get_nikkoku_senses(text):
    # make duplicate \n's to \n
    text = re.sub(r'\n+', '\n', text)
    text = text.split('\n')
    if text[1][-1] != '。' and len(text) > 2:
        text[1] += text[2] # add the next line if the current line does not end with '。' (e.g., parentheses)
    sense = re.sub(r'(〘副〙|〘感動〙|１|２|３|４|５|６|７|８|９|①|②|③|④|⑤|⑥|⑦|⑧|⑨)', '', text[1]).strip()
    # remove between 「」 or ()
    sense = re.sub(r'(（[^）]*）|「[^」]*」)', '', sense).strip()
    # remove if any onopatopoeia word is in the sense with look-ahead
    for o in onomatopoeias:
        sense = re.sub(f'(?<=。){o['word']}。', '', sense).strip()
    return sense

sense_keys = [
    '音','声','表す','さま',
]
for o in tqdm(onomatopoeias):
    if o['nikkoku'] == []:
        o['nikkoku_simple'] = ''
        o['ref'] = ''
        continue
    o['nikkoku_simple'] = get_nikkoku_senses(o['nikkoku'][0])
    o['ref'] = 'nikkoku'
    for sense in o['nikkoku']:
        if any([key in sense for key in sense_keys]):
            o['nikkoku_simple'] = get_nikkoku_senses(sense)
            break

# save to csv file
with open('dataset/0_raw/nat/ja_temp.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'meaning', 'ref', 'revised', 'nikkoku_simple', 'nikkoku', 'kojien', 'daijisen', 'daijirin'])
    for o in onomatopoeias:
        writer.writerow([o['word'], o['nikkoku_simple'], o['ref'], '', o['nikkoku_simple'], str(o['nikkoku']), str(o['kojien']), str(o['daijisen']), str(o['daijirin'])])

# now, manually check the csv file and fix the meanings

