import re
import json
import math
from pathlib import Path
import unicodedata 

def parse_textgrid(textgrid_path:str):
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        data = f.read()    
    try:
        # Xmin
        xmin = re.search(r"xmin = ([0-9.]+)", data ).group(1)
        # Xmax
        xmax = re.search(r"xmax = ([0-9.]+)", data ).group(1)

        # Items
        words, phones = re.search(r'item \[[1]\]:([\s\S]+)item \[2\]:([\s\S]+)$', data).groups()

        # Intervals
        pattern = r'intervals \[([0-9])\]:\s*xmin = ([0-9.]+)\s*xmax = ([0-9.]+)\s*text = "(.*)"' 
        words_interval = [dict(interval = int(idx), xmin=float(xmin), xmax=float(xmax), text=text) for idx, xmin, xmax, text in re.findall(pattern, words)]
        phones_interval = [dict(interval = int(idx), xmin=float(xmin), xmax=float(xmax), text=text) for idx, xmin, xmax, text in re.findall(pattern, phones)]

        result_dict = {
            'xmin': float(xmin),
            'xmax' : float(xmax),
            'words' : words_interval,
            'phones' : phones_interval
        }
        return result_dict
    except Exception as e:
        print(textgrid_path)
        return {}



def textgrid_to_array(xmax, phones_interval, frame_duration_ms=40):
   
    frame_duration_s = frame_duration_ms / 1000.0
    total_frames = math.ceil(xmax / frame_duration_s)

    results = []

    for i in range(total_frames):
        frame_start = i * frame_duration_s
        frame_end = (i + 1) * frame_duration_s
        frame_center = (frame_start + frame_end) / 2

        ipa = ""
        for interval in phones_interval:
            if interval['xmin'] <= frame_center <= interval['xmax']:
                ipa = interval["text"]
        
        if ipa == "":
            ipa = None
        results.append(ipa)

    return results

def main():

    lang = 'art' # 'en', 'fr', 'ja', 'art
    frame_duration_ms = 40
    root = Path('/home/sunahan/workspace/sound-symbolism')

    
    data_dir  = root / 'data' / 'processed'
    if lang == 'art':
        textgrid_dir = data_dir / 'art' / 'textgrids'
        output_dir = data_dir / 'art' / 'alignment'
        output_path = output_dir /'constructed_words.json' 
    else:
        textgrid_dir = data_dir / 'nat' / 'textgrids' / lang
        output_dir = data_dir / 'nat' / 'alignment'
        output_path = output_dir / f'{lang}.json'

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    oov = []
    textgrid_files = textgrid_dir.glob("*.TextGrid")
    for textgrid_path in textgrid_files:
        word = unicodedata.normalize('NFKC', str(textgrid_path.stem))
        data = parse_textgrid(textgrid_path=textgrid_path)
        phones = textgrid_to_array(xmax = data['xmax'], 
                                   phones_interval=data['phones'],
                                   frame_duration_ms=frame_duration_ms)

        if 'spn' in phones:
            oov.append(word)

        result_dict = {
            'word':  word,
            'sampling_rate': 16000,
            'frame_duration_ms': frame_duration_ms,
            'phones': phones,
            'mfa_raw': data
        }
        results.append(result_dict)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("[INFO] File saved to: ", output_path)

    print(f"[WARN] Out of Vocabulary #{len(oov)}: {oov}")

if __name__ == "__main__":
    main()