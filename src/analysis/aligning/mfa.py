import csv
import time
import subprocess
import unicodedata
from pathlib import Path

class MFAWrapRunner():
    '''
    CLI 도구인 MFA를 Python에서 Wrapping하여 명령어 실행 
    '''
    def __init__(self):
        self.check_mfa_installation()
        self.model_name = None

        if not self.check_mfa_installation():
            raise ModuleNotFoundError("MFA is not installed. Run `conda install -c conda-forge montreal-forced-aligner`")      

    def run(self, lang, corpus_dir, output_dir, dict_path=None, validate=False, pretrained=True, clean=True):
        self.set_model_name(lang)

        if not self.check_model_downloaded():
            download = True
        else:
            download = False

        if download:
            self.download_model('acoustic')
        
            if pretrained:
                self.download_model('dictionary')


        if validate:
            print("[INFO] Validate Corpus")
            self.validate_corpus(corpus_dir)
            print("[INFO] Finished")

            self.validate_dictionary(dict_path)

        print("[INFO] Start Aligning")
        tic = time.time()
        if pretrained:
            self.align_with_pretrained_model(corpus_dir, output_dir, clean)
        else:
            self.align_with_custom_dict(corpus_dir, dict_path, output_dir, clean)
        toc = time.time()
        processed_time = toc - tic
        print(f"[INFO] Finished: {processed_time}s ")

    
    def check_mfa_installation(self):
        try:
            result = subprocess.run(['mfa', 'version'], capture_output=True, text=True)
            print("[DEBUG] process running result: ", result)
            if result.returncode == 0:
                print(f'MFA is installed: {result.stdout.strip()}')
                return True
            else:
                print("MFA is not installed")
                return False
        except FileNotFoundError:
            return False

    def set_model_name(self, lang):
        model_map = {
            'en': 'english_us_arpa',
            'fr': 'french_mfa',
            'ja': 'japanese_mfa',
            'ko': 'korean_mfa'
        }
        if lang == 'art':
            lang = 'en'
        if lang not in model_map:
            raise ValueError
        
        self.model_name = model_map[lang]

    def check_model_downloaded(self, lang=None):
        if lang is not None:
            self.set_model_name(lang)

        try:
            cmd = ['mfa', 'model', 'download', 'acoustic', self.model_name]
            result = self.run_mfa_command(cmd)

            if result.returncode == 0:
                print(f'[INFO] Model is already downloaded')
                return True
            else:
                print(f'[INFO] Model is not downladed yet')
                return False
        except Exception as e:
            print(e)
            raise e

 
    def download_model(self, model_type):
        if self.model_name is None:
            print('run `set_model_name(lang)` method first')

        cmd = ['mfa', 'model', 'download', model_type, self.model_name]   
        self.run_mfa_command(cmd)

    
    def validate_corpus(self, corpus_dir):
        cmd = ['mfa', 'validate', corpus_dir, self.model_name, self.model_name ]
        self.run_mfa_command(cmd)

    def validate_dictionary(self, dict_path):
        cmd = ['mfa', 'validate_dictionary', dict_path]
        self.run_mfa_command(cmd)

    def align_with_custom_dict(self, corpus_dir, dict_path, output_dir,  clean):
        cmd = ['mfa', 'align', corpus_dir, dict_path, self.model_name, output_dir ]
        if clean:
            cmd.append('--clean')
        
        return self.run_mfa_command(cmd)
    
    
    def align_with_pretrained_model(self, corpus_dir, output_dir, clean):
        '''
        사전 훈련된 모델로 정렬하여 'word-ipa' 딕셔너리 필요 X
        '''
        if self.model_name is None:
            print('call `set_model_name(lang)` method first.')
            return 
        
        cmd = ['mfa', 'align', corpus_dir, self.model_name, self.model_name, output_dir]
         
        if clean:
            cmd.append('--clean')

        return self.run_mfa_command(cmd)
    
    def run_mfa_command(self, cmd):
        try:
            print(f'[INFO] Running... `{" ".join(cmd)}`')
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print('[INFO] Finished')
            if result.stdout:
                print('[INFO]', result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f'[Error] {e.stderr}')
            return -1
   
    def list_models(self, model_type):
        cmd = ['mfa', 'model', 'download', model_type, '--list']
        return self.run_mfa_command(cmd)


class MFADataPrep:
    def __init__(self, corpus_dir):
        self.corpus_dir = Path(corpus_dir)
    
    def prepare_data(self):
        audio_files = self.corpus_dir.glob("*.wav")
        for audio_file in audio_files:
            transcription = unicodedata.normalize('NFKC', audio_file.stem)
            lab_filepath = self.corpus_dir / f'{transcription}.lab'
            self._create_lab_file(transcription, lab_filepath)
        

    def _create_lab_file(self, trascription, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(trascription)
        print(f"[INFO] LAB file saved: {output_path}")

    def create_custom_dict(self, word_ipa_pairs, dict_path):
       
        with open(dict_path, 'w', encoding='utf-8') as f:
            for word, ipa in word_ipa_pairs:
                if word != 'alphabet':
                    ipa = ipa.upper()
                    f.write(f"{word}\t{' '.join(ipa)}\n")
        print(f"[INFO] Pronounciation dictionary saved: {dict_path}")

        


def main():
    corpus_dir = './example'
    output_dir = './output'
    pretrained = True
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    prep = MFADataPrep(corpus_dir=corpus_dir)
    prep.prepare_data()

    # Artificial(Constructed) Words (Nonwords)#####
    dict_path = './nonwords_pronunciation_dict.txt'

    # words_path = '/home/sunahan/workspace/sound-symbolism/data/processed/art/outputs/constructed_nonwords.csv'
    # with open(words_path, 'r', encoding='utf-8') as f:
    #     reader = csv.reader(f)
    #     word_ipa_pairs = [ (word, ipa) for _, ipa, word in reader]
    #    prep.create_custom_dict(word_ipa_pairs,dict_path)
    pretrained = False
    ################################################
    
    mfa = MFAWrapRunner()
    mfa.run(lang='en', corpus_dir=corpus_dir, output_dir=output_dir, dict_path=str(dict_path), pretrained=pretrained)




if __name__ == '__main__':
    main()