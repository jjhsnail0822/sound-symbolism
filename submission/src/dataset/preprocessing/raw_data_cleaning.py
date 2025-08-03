import json

# clean raw data to remove found==false items
langs = ['en', 'fr', 'ko', 'ja']

datas = {}
for lang in langs:
    with open(f'data/processed/nat/{lang}_with_not_found.json', 'r', encoding='utf-8') as f:
        datas[lang] = json.load(f)

new_datas = {}
for lang in langs:
    new_datas[lang] = []
    for data in datas[lang]:
        if data['found'] == True:
            new_datas[lang].append(data)

# save new_datas to json files
for lang in langs:
    with open(f'data/processed/nat/{lang}.json', 'w', encoding='utf-8') as f:
        json.dump(new_datas[lang], f, ensure_ascii=False, indent=4)

# save how many items are left in each language
with open('results/statistics/statistics.txt', 'w', encoding='utf-8') as f:
    for lang in langs:
        f.write(f'{lang}: {len(datas[lang])} -> {len(new_datas[lang])}\n')
