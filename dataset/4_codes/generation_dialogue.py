import os
import json
import argparse
import time
<<<<<<< HEAD
from pprint import pprint
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
import random
import re
import ast
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env.local file
env_path = Path('/scratch2/sheepswool/workspace/sound-symbolism/.env.local')
load_dotenv(dotenv_path=env_path)

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
=======
from tqdm import tqdm
import openai
import pandas as pd
import random
from typing import List, Dict, Any, Optional

# Bring api key from .env.local
api_key = os.getenv("OPENAI_API_KEY")
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0

class DialogueGenerator:
    def __init__(self, language: str, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the dialogue generator
        
        Args:
            language (str): Language code (en/fr/ko/ja)
            api_key (str, optional): OpenAI API key. If None, will try to get from environment variable
            model (str): OpenAI model to use
        """
        self.language = language.lower()
        if self.language not in ['en', 'fr', 'ko', 'ja']:
            raise ValueError("Language must be one of 'en', 'fr', 'ko', 'ja'.")
        
        # Set paths
        self.input_path = os.path.join('../1_preprocess/nat', f"{self.language}.json")
<<<<<<< HEAD
        self.output_path = '../2_dialogue/nat'
        self.output_file = os.path.join(self.output_path, f"{self.language}.json")
=======
        self.output_path = os.path.join('../2_dialogue/nat', f"{self.language}")
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
<<<<<<< HEAD
        # Set OpenAI API key and client
        if api_key:
            self.api_key = api_key
        elif OPENAI_API_KEY:
            self.api_key = OPENAI_API_KEY
            print(f"Using API key from environment: {OPENAI_API_KEY[:5]}...")
        else:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
=======
        # Set OpenAI API key
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
        
        # Set model
        self.model = model
        
        # Load language-specific templates
        self.templates = self._get_language_templates()
        
        # Load data
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter data to only include entries with meanings
            filtered_data = [item for item in data if item.get('meaning') and len(item.get('meaning', [])) > 0 and item.get('found', False)]
            
            print(f"Loaded {len(filtered_data)} entries with meanings out of {len(data)} total entries")
            return filtered_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def _get_language_templates(self) -> Dict[str, Dict[str, str]]:
        """Get language-specific templates for prompts"""
        templates = {
            'en': {
                'system_prompt': "You are a helpful assistant that creates natural dialogues between two people. The dialogue should include the onomatopoeic word '{word}' which means '{meaning}'. Make the dialogue sound natural and contextually appropriate.",
                'user_prompt': """Condition 1: Create a dialogue between two people that naturally incorporates the onomatopoeic word '{word}' which means '{meaning}'.\n \
<<<<<<< HEAD
                    Condition 2: The dialogue must be {num_utterances} utterances (= {num_turns} turns) long and feel natural.\n \
                    Condition 3: The dialogue must not include greetings or farewells.\n \
                    Condition 4: The onomatopoeic word must appear in the {ss_idx}th utterance.\n \
                    Condition 5: The dialogue must not directly include the meaning of the onomatopoeic word.""",
                'dialogue_format': """['John: {}\nEmma: {}'], ['Michael: {}\nSarah: {}'], ['David: {}\nOlivia: {}']"""
=======
                    Condition 2: The dialogue should be {num_utterances} utterances (= {num_turns} turns) long and feel natural.\n \
                    Condition 3: The dialogue should not include greetings or farewells.\n \
                    Condition 4: The onomatopoeic word should appear in the {ss_idx}th utterance.\n \
                    Condition 5: The dialogue should not directly include the meaning of the onomatopoeic word.""",
                'dialogue_format': ["John: {}\nEmma: {}", "Michael: {}\nSarah: {}", "David: {}\nOlivia: {}"]
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
            },
            'fr': {
                'system_prompt': "Vous êtes un assistant utile qui crée des dialogues naturels entre deux personnes. Le dialogue doit inclure le mot onomatopéique '{word}' qui signifie '{meaning}'. Faites en sorte que le dialogue semble naturel et approprié au contexte.",
                'user_prompt': """Condition 1: Créez un dialogue entre deux personnes qui incorpore naturellement le mot onomatopéique '{word}' qui signifie '{meaning}'.\n \
                    Condition 2: Le dialogue doit comporter {num_utterances} énoncés (= {num_turns} tours de parole) et sembler naturel.\n \
                    Condition 3: Le dialogue ne doit pas inclure de salutations ni d'adieux.\n \
                    Condition 4: Le mot onomatopéique doit apparaître dans le {ss_idx}ème énoncé.\n \
                    Condition 5: Le dialogue ne doit pas inclure directement la signification du mot onomatopéique.""",
<<<<<<< HEAD
                'dialogue_format': """['Pierre: {}\nSophie: {}'], ['Thomas: {}\nCamille: {}'], ['Antoine: {}\nJulie: {}']"""
=======
                'dialogue_format': ["Pierre: {}\nSophie: {}", "Thomas: {}\nCamille: {}", "Antoine: {}\nJulie: {}"]
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
            },
            'ko': {
                'system_prompt': "당신은 두 사람 사이의 자연스러운 대화를 만드는 도우미입니다. 대화에는 '{meaning}'을(를) 의미하는 의성어/의태어 '{word}'가 포함되어야 합니다. 대화가 자연스럽고 문맥에 적절하게 들리도록 만드세요.",
                'user_prompt': """조건1: '{meaning}'을(를) 의미하는 의성어/의태어 '{word}'를 자연스럽게 포함하는 두 사람 사이의 대화를 만들어주세요.\n \
                    조건2: 대화는 {num_utterances}번의 발화(={num_turns}턴)로 이루어져야 하며 자연스럽게 느껴져야 합니다.\n \
<<<<<<< HEAD
                    조건3: 의성어/의태어는 반드시 {ss_idx}번째 발화에 < > 안에 표기되어 나타나야 합니다.\n \
                    조건4: 대화에는 인사말과 헤어지는 표현을 포함하면 안 됩니다. \n \
                    조건5: 대화 내에는 의성어/의태어의 의미가 직접적으로 포함되면 안 됩니다.
                    조건6: 대화에 등장할 수 있는 인물의 이름은 다음과 같습니다. 철수, 영희, 민수, 지혜, 준호, 수진""",
                'dialogue_format': """[ {"화자": {사람 이름}, "발화 번호": {1}, "발화": {발화 내용}},
                    {"화자": {사람 이름}, "발화 번호": {2}, "발화": {발화 내용}},
                    {"화자": {사람 이름}, "발화 번호": {3}, "발화": {발화 내용}},
                    {"화자": {사람 이름}, "발화 번호": {4}, "발화": {발화 내용}},
                    {"화자": {사람 이름}, "발화 번호": {5}, "발화": {발화 내용}},
                    {"화자": {사람 이름}, "발화 번호": {6}, "발화": {발화 내용}},
                    {"화자": {사람 이름}, "발화 번호": {7}, "발화": {발화 내용}},
                    {"화자": {사람 이름}, "발화 번호": {8}, "발화": {발화 내용}},
                ]"""
=======
                    조건3: 대화에는 인사말과 헤어지는 표현을 포함하면 안 됩니다. \n \
                    조건4: 의성어/의태어는 {ss_idx}번째 발화에 나타나야 합니다.\n \
                    조건5: 대화 내에는 의성어/의태어의 의미가 직접적으로 포함되면 안 됩니다.""",
                'dialogue_format': ["철수: {}\n영희: {}", "민수: {}\n지혜: {}", "준호: {}\n수진: {}"]
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
            },
            'ja': {
                'system_prompt': "あなたは二人の間で自然な対話を作成する役立つアシスタントです。対話には「{meaning}」を意味するオノマトペ「{word}」を含める必要があります。対話が自然で文脈に適切に聞こえるようにしてください。",
                'user_prompt': """条件1: 「{meaning}」を意味するオノマトペ「{word}」を自然に取り入れた二人の間の対話を作成してください。\n \
                    条件2: 対話は{num_utterances}回の発話(={num_turns}ターン)で構成され、自然に感じられるようにしてください。\n \
                    条件3: 対話には挨拶や別れの表現を含めないでください。\n \
<<<<<<< HEAD
                    条件4: オノマトペは必ず{ss_idx}番目の発話に現れるようにしてください。\n \
                    条件5: 対話の中にオノマトペの意味を直接含めないでください。""",
                'dialogue_format': """['太郎: {}\n花子: {}'], ['健太: {}\n美咲: {}'], ['翔太: {}\nさくら: {}']"""
=======
                    条件4: オノマトペは{ss_idx}番目の発話に現れるようにしてください。\n \
                    条件5: 対話の中にオノマトペの意味を直接含めないでください。""",
                'dialogue_format': ["太郎: {}\n花子: {}", "健太: {}\n美咲: {}", "翔太: {}\nさくら: {}"]
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
            }
        }
        return templates
    
    def generate_dialogue(self, word_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dialogue for a given word"""
        word = word_data.get('word', '')
        
        # Get the first meaning if it's a list, or use the meaning directly
        meaning = word_data.get('meaning', [])
        if isinstance(meaning, list) and meaning:
            meaning = meaning[0]
        elif not meaning:
            meaning = "an onomatopoeic word"
        
        num_utterances = 8
        ss_idx = random.randint(1, num_utterances)
        num_turns = num_utterances // 2
<<<<<<< HEAD
        
        # Format prompts
        system_prompt = self.templates[self.language]['system_prompt'].format(word=word, meaning=meaning)
        user_prompt = self.templates[self.language]['user_prompt'].format(
            word=word, 
            meaning=meaning, 
            num_utterances=num_utterances, 
            num_turns=num_turns,
            ss_idx=ss_idx
        )
        dialogue_format = random.choice(self.templates[self.language]['dialogue_format'])
        
        try:
            # Call OpenAI API using the new client format
            response = self.client.chat.completions.create(
=======
        # Format prompts
        system_prompt = self.templates[self.language]['system_prompt'].format(word=word, meaning=meaning)
        user_prompt = self.templates[self.language]['user_prompt'].format(word=word, meaning=meaning, num_utterances=num_utterances, num_turns=num_turns, ss_idx=ss_idx)
        dialogue_format = self.templates[self.language]['dialogue_format']
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "user", "content": dialogue_format}
                ],
                temperature=1,
                max_tokens=1000
            )
            
<<<<<<< HEAD
            # Extract dialogue from response (new format)
            dialogue = response.choices[0].message.content.strip()
            dialogue = dialogue.replace("```", "").replace("json", "").strip()
            
            # Parse dialogue into structured format
            parsed_dialogue = self._parse_dialogue(dialogue)
            
            # Check if the dialogue includes the word in the dialogue
            while word not in parsed_dialogue[ss_idx-1]["text"] \
                or any(word in utterance["text"] for utterance in parsed_dialogue[min(ss_idx, num_utterances-1):]) \
                or any(word in utterance["text"] for utterance in parsed_dialogue[:max(ss_idx-1, 0)]):
                    
                print("Result before change")
                pprint(parsed_dialogue)
                print(f"Word: {word} Index: {ss_idx} Meaning: {meaning}")
                # Retry generation
                response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "user", "content": dialogue_format}
                    ],
                    temperature=1,
                    max_tokens=1000
                )
                dialogue = response.choices[0].message.content.strip()
                parsed_dialogue = self._parse_dialogue(dialogue)
                print("Result of reattempt")
                pprint(parsed_dialogue)
                print(f"Word: {word} Index: {ss_idx} Meaning: {meaning}")
                time.sleep(2)
                breakpoint()
=======
            # Extract dialogue from response
            dialogue = response.choices[0].message.content.strip()
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
            
            # Create result dictionary
            result = {
                'word': word,
                'meaning': meaning,
<<<<<<< HEAD
                'dialogue': parsed_dialogue,
=======
                'dialogue': dialogue,
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
                "meta_data": {
                    'language': self.language,
                    "num_utterances": num_utterances,
                    "ss_idx": ss_idx,
                    "success": True,
                },
            }
            
            return result
        except Exception as e:
            print(f"Error generating dialogue for word '{word}': {e}")
<<<<<<< HEAD
            breakpoint()
=======
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
            # Return empty dialogue on error
            return {
                'word': word,
                'meaning': meaning,
                'dialogue': "",
                "meta_data": {
                    'language': self.language,
                    "num_utterances": num_utterances,
                    "ss_idx": ss_idx,
                    "success": False,
                },
            }
    
<<<<<<< HEAD
    def _parse_dialogue(self, dialogue_text: str) -> List[Dict[str, str]]:
        """Parse dialogue text into a structured list of utterances"""
        result = []
        
        # Clean up the dialogue text
        dialogue_text = dialogue_text.strip()
        
        # Try to parse as a list if it looks like one
        if dialogue_text.startswith('[') and dialogue_text.endswith(']'):
            try:
                # Try to safely evaluate as a Python literal
                parsed_list = ast.literal_eval(dialogue_text)
                if isinstance(parsed_list, list):
                    for item in parsed_list:
                        if isinstance(item, str) and ':' in item:
                            breakpoint()
                            parts = item.split(':', 1)
                            if len(parts) == 2:
                                speaker = parts[0].strip()
                                text = parts[1].strip()
                                index = int(text.split()[0])
                                result.append({"speaker": speaker, "index": index, "text": text})
                    return result
            except (SyntaxError, ValueError):
                pass
        
        # If not a list or parsing failed, try line-by-line parsing
        lines = dialogue_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                text = parts[1].strip()
                result.append({"speaker": speaker, "text": text})
        
        return result
    
=======
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
    def generate_all_dialogues(self, limit: Optional[int] = None, shuffle: bool = True) -> List[Dict[str, Any]]:
        """Generate dialogues for all words in the data"""
        if not self.data:
            print("No data available to generate dialogues")
            return []
        
        # Optionally shuffle the data
        data_to_process = self.data.copy()
        if shuffle:
            random.shuffle(data_to_process)
        
        # Limit the number of items to process if specified
        if limit and limit > 0:
            data_to_process = data_to_process[:limit]
        
        results = []
        
<<<<<<< HEAD
        temp_stop_idx = 10
        # Process each word with progress bar
        for i, item in tqdm(enumerate(data_to_process), desc=f"Generating {self.language} dialogues", total=len(data_to_process)):
            # Generate dialogue
            result = self.generate_dialogue(item)
            
            # Add to results list
            results.append(result)
            
            # Save intermediate results every 10 items
            if (i + 1) % 10 == 0:
                self._save_intermediate_results(results)
            
            # Add a delay to avoid rate limiting
            time.sleep(1)
            
            if i == temp_stop_idx:
                break
        
        return results
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]]) -> None:
        """Save intermediate results to avoid losing progress"""
        try:
            # Create temporary filename
            temp_filename = f"{self.output_file}.temp"
            
            with open(temp_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            print(f"Saved intermediate results ({len(results)} items)")
        except Exception as e:
            print(f"Error saving intermediate results: {e}")
    
    def save_results(self, dialogues: List[Dict[str, Any]]) -> bool:
        """Save all dialogues to a single JSON file"""
        if not dialogues:
            print("No dialogues to save")
            return False
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(dialogues, f, ensure_ascii=False, indent=4)
            print(f"All dialogues saved to {self.output_file}")
            return True
        except Exception as e:
            print(f"Error saving dialogues to file: {e}")
            return False
=======
        # Process each word with progress bar
        for item in tqdm(data_to_process, desc=f"Generating {self.language} dialogues"):
            # Generate dialogue
            result = self.generate_dialogue(item)
            
            # Save individual dialogue to file
            self._save_dialogue_to_file(result)
            
            # Add to results list
            results.append(result)
            
            # Add a delay to avoid rate limiting
            time.sleep(1)
        
        return results
    
    def _save_dialogue_to_file(self, dialogue_data: Dict[str, Any]) -> None:
        """Save a single dialogue to a JSON file"""
        word = dialogue_data.get('word', 'unknown')
        # Replace characters that might be problematic in filenames
        safe_word = ''.join(c if c.isalnum() else '_' for c in word)
        
        # Create filename
        filename = os.path.join(self.output_path, f"{safe_word}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dialogue_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error saving dialogue for word '{word}' to file: {e}")
    
    def save_all_dialogues(self, dialogues: List[Dict[str, Any]]) -> None:
        """Save all dialogues to a single JSON file"""
        if not dialogues:
            print("No dialogues to save")
            return
        
        # Create filename for all dialogues
        filename = os.path.join(self.output_path, f"all_dialogues_{self.language}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dialogues, f, ensure_ascii=False, indent=4)
            print(f"All dialogues saved to {filename}")
        except Exception as e:
            print(f"Error saving all dialogues to file: {e}")
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
    
    def run(self, limit: Optional[int] = None, shuffle: bool = True) -> None:
        """Run the dialogue generation process"""
        print(f"Starting dialogue generation for language: {self.language}")
        
        # Generate dialogues
        dialogues = self.generate_all_dialogues(limit=limit, shuffle=shuffle)
        
        # Save all dialogues to a single file
<<<<<<< HEAD
        success = self.save_results(dialogues)
=======
        self.save_all_dialogues(dialogues)
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
        
        print(f"Dialogue generation completed. Generated {len(dialogues)} dialogues.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dialogue Generation Tool')
    parser.add_argument('--language', '-l', required=True, choices=['en', 'fr', 'ko', 'ja'], 
                        help='Language code (en/fr/ko/ja)')
<<<<<<< HEAD
    parser.add_argument('--model', '-m', default="gpt-4o", 
                        help='OpenAI model to use (default: gpt-4o)')
=======
    parser.add_argument('--model', '-m', default="gpt-3.5-turbo", 
                        help='OpenAI model to use (default: gpt-3.5-turbo)')
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
    parser.add_argument('--limit', '-n', type=int, help='Limit the number of dialogues to generate')
    parser.add_argument('--no-shuffle', action='store_true', help='Do not shuffle the data before processing')
    
    args = parser.parse_args()
    
    generator = DialogueGenerator(
        language=args.language,
<<<<<<< HEAD
=======
        api_key=args.api_key,
>>>>>>> d3469426fdda53a1a3933b1ac1ed69fb3c6270c0
        model=args.model
    )
    
    generator.run(limit=args.limit, shuffle=not args.no_shuffle) 