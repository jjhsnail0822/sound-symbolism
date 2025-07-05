import json
import os
from argparse import ArgumentParser
from gtts import gTTS
from pydub import AudioSegment
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--word_type", type=str, choices=['art', 'nat'], default='nat')
parser.add_argument("--base_dir", type=str, default='data/processed/')

# List of languages to process
# langs = ['en', 'fr', 'ja', 'ko']
langs = ['ko']
# Base path for the output audio files

class TTSGenerator:
    """
    A class to generate Text-to-Speech audio using gTTS
    and save it as a WAV file.
    """
    def __init__(self, language, output_path:str = None):
        """
        Initializes the TTSGenerator.
        Args:
            language (str): The language for TTS generation (e.g., 'en', 'ko').
        """
        self.language = language
        self.output_path = output_path
           

    def generate(self, text):
        """
        Generates a WAV audio file for the given text.
        It first creates an MP3 file using gTTS, then converts it to WAV,
        and finally deletes the temporary MP3 file.
        Args:
            text (str): The text to convert to speech.
        """
        try:
            # Define output and temporary file paths
            wav_output_file = f'{self.output_path}/{text}.wav'
            mp3_temp_file = f'{self.output_path}/{text}.mp3'
            
            # If the WAV file already exists, skip generation
            if os.path.exists(wav_output_file):
                return

            # Generate TTS and save as a temporary MP3 file
            tts_obj = gTTS(text=text, lang=self.language, slow=False)
            tts_obj.save(mp3_temp_file)

            # Convert the MP3 file to WAV using pydub
            audio = AudioSegment.from_mp3(mp3_temp_file)
            audio.export(wav_output_file, format="wav")

            # Remove the temporary MP3 file
            os.remove(mp3_temp_file)

        except Exception as e:
            print(f"An error occurred while generating TTS for '{text}': {e}")
        return

# Main script execution
if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = f"{args.base_dir}/{args.word_type}"
   

    if args.word_type == 'nat':
        # Loop through each language
        for lang in langs:
            # Create the output directory for the language if it doesn't exist
            lang_output_path = f"{base_dir}/tts/{lang}"
            os.makedirs(lang_output_path, exist_ok=True)

            # Load the corresponding JSON data file
            json_path = f'{base_dir}/{lang}.json'
            if not os.path.exists(json_path):
                print(f"Warning: JSON file not found at {json_path}. Skipping language {lang}.")
                continue
                
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Initialize the TTS generator for the current language
            tts = TTSGenerator(lang)
            
            # Generate audio for each word in the data
            for item in tqdm(data, desc=f"Generating TTS for {lang}"):
                if 'word' in item:
                    tts.generate(item['word'])
                else:
                    print(f"Warning: 'word' key not found in item: {item}")
    elif args.word_type=='art':
        output_dir = f"{base_dir}/tts"
        os.makedirs(output_dir, exist_ok=True)

        json_path =f'{base_dir}/semantic_dimension/semantic_dimension_binary_gt.json'
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tts = TTSGenerator('en', output_path = output_dir)

        for item in tqdm(data['art'], desc='Generating TTS for artificial nonwords'):
            if 'word' in item:
                tts.generate(item['word'])
            else:
                print(f"Warning: 'word' key not found in item: {item}")
