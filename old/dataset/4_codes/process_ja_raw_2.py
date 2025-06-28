import csv
import json

# This file is used to convert the raw csv file to a json file.
# There should be a prepared csv file in the dataset/0_raw directory.

with open('dataset/0_raw/nat/ja.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    ja_words = [row for row in reader]
    ja_words = ja_words[1:]  # remove header

# csv structure:
# word, meaning, ref, nikkoku_simple, nikkoku, kojien, daijisen, daijirin

data = []
for word in ja_words:
    word_dict = {
        'word': word[0],
        'meaning': word[1],
        'ref': word[2],
        'url': '精選版 日本国語大辞典' if word[2] == 'nikkoku' \
            else '広辞苑 第七版' if word[2] == 'kojien' \
                else 'デジタル大辞泉' if word[2] == 'daijisen' \
                    else '大辞林 第四版' if word[2] == 'daijirin' \
                        else word[2],
        'found': True if word[1] != '' else False,
        # 'definitions': {
        #     'nikkoku_simple': word[3],
        #     'nikkoku': eval(word[4]),
        #     'kojien': eval(word[5]),
        #     'daijisen': eval(word[6]),
        #     'daijirin': eval(word[7]),
        # }
    }
    data.append(word_dict)

# save to json file
with open('dataset/1_preprocess/nat/ja.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)