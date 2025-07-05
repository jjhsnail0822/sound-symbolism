import json
from argparse import ArgumentParser
from pathlib import Path
parser = ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--output_dir', type=str)

args = parser.parse_args()

data_dir = Path(args.data_dir)

ipa_dicts =list( data_dir.glob('*.txt'))
print("[INFO] ipa_dicts: ", [path.name for path in ipa_dicts])

result_dicts = {}
for ipa_dict in ipa_dicts:
    lang = ipa_dict.stem

    result_dict = {}
    with open(ipa_dict, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"[INFO] {lang}: {len(lines)} words")

    for line in lines:
        line = line.strip()
        if line:
            word, ipas = line.split('\t')
            ipas = ipas.split(',')
            for ipa in ipas:
                ipa = ipa.replace("ˈ", "").replace("ː", "").replace("/", "")
                ipa = ipa.replace("eɪ", "ej").replace("oʊ", "ow").replace("ɫ", 'l')
                ipa = ipa.replace("ɭ","l").replace("ɾ", 'l')
                ipa = ipa.replace("g", "ɡ")
                result_dict.update({ipa:word})
                
    result_dicts[lang] = result_dict
output_path = Path(args.output_dir) / 'ipa_to_word.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result_dicts, f, ensure_ascii=False, indent=4)

print("[INFO] Saved to: ", output_path)