import os
import json
import argparse
import time
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
env_path = Path('/scratch2/sheepswool/workspace/.env.local')
load_dotenv(dotenv_path=env_path)

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class DialogueGenerator:
    def __init__(self, language: str, api_key: Optional[str] = None, model: str = "gpt-4o"):
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
        self.output_path = '../2_dialogue/nat'
        self.output_file = os.path.join(self.output_path, f"{self.language}.json")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Set OpenAI API key and client
        if api_key:
            self.api_key = api_key
        elif OPENAI_API_KEY:
            self.api_key = OPENAI_API_KEY
            # print(f"Using API key from environment: {OPENAI_API_KEY[:5]}...")
        else:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
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
                'user_prompt_set1': """Condition 1: Create a dialogue between two people that naturally incorporates the onomatopoeic word '{word}' which means '{meaning}'.\n
                    Condition 2: The dialogue must be {num_utterances} utterances (= {num_turns} turns) long and feel natural.\n
                    Let's think step by step.""",
                'user_prompt_set2': """Condition 3: The onomatopoeic word must appear in the {ss_idx}th utterance. The context can be changed as needed to naturally incorporate the word.\n
                    Condition 4: Check if the dialogue directly includes the meaning of the onomatopoeic word, and if possible, avoid that expression by creating natural and creative examples.\n
                    Let's think step by step.""",
                'user_prompt_set3': """Condition 5: The characters in the dialogue should have names that exist among speakers of the language. There should be 2 characters.\n
                    Condition 6: If the dialogue includes greetings or farewells, replace those utterances.\n
                    Let's think step by step.""",
                'dialogue_format': """[ {"speaker": "Person1", "utterance_number": 1, "text": "utterance content"},
                    {"speaker": "Person2", "utterance_number": 2, "text": "utterance content"},
                    {"speaker": "Person1", "utterance_number": 3, "text": "utterance content"},
                    {"speaker": "Person2", "utterance_number": 4, "text": "utterance content"},
                    {"speaker": "Person1", "utterance_number": 5, "text": "utterance content"},
                    {"speaker": "Person2", "utterance_number": 6, "text": "utterance content"}
                ]"""
            },
            'fr': {
                'system_prompt': "Vous êtes un assistant utile qui crée des dialogues naturels entre deux personnes. Le dialogue doit inclure le mot onomatopéique '{word}' qui signifie '{meaning}'. Faites en sorte que le dialogue semble naturel et approprié au contexte.",
                'user_prompt_set1': """Condition 1: Créez un dialogue entre deux personnes qui incorpore naturellement le mot onomatopéique '{word}' qui signifie '{meaning}'.\n
                    Condition 2: Le dialogue doit comporter {num_utterances} énoncés (= {num_turns} tours de parole) et sembler naturel.\n
                    Réfléchissons étape par étape.""",
                'user_prompt_set2': """Condition 3: Le mot onomatopéique doit apparaître dans le {ss_idx}ème énoncé. Le contexte peut être modifié si nécessaire pour incorporer naturellement le mot.\n
                    Condition 4: Vérifiez si le dialogue inclut directement la signification du mot onomatopéique, et si possible, évitez cette expression en créant des exemples naturels et créatifs.\n
                    Réfléchissons étape par étape.""",
                'user_prompt_set3': """Condition 5: Les personnages du dialogue doivent avoir des noms qui existent parmi les locuteurs de la langue. Il doit y avoir 2 personnages.\n
                    Condition 6: Si le dialogue inclut des salutations ou des adieux, remplacez ces énoncés.\n
                    Réfléchissons étape par étape.""",
                'dialogue_format': """[ {"speaker": "Personne1", "utterance_number": 1, "text": "contenu de l'énoncé"},
                    {"speaker": "Personne2", "utterance_number": 2, "text": "contenu de l'énoncé"},
                    {"speaker": "Personne1", "utterance_number": 3, "text": "contenu de l'énoncé"},
                    {"speaker": "Personne2", "utterance_number": 4, "text": "contenu de l'énoncé"},
                    {"speaker": "Personne1", "utterance_number": 5, "text": "contenu de l'énoncé"},
                    {"speaker": "Personne2", "utterance_number": 6, "text": "contenu de l'énoncé"}
                ]"""
            },
            'ko': {
                'system_prompt': "당신은 두 사람 사이의 자연스러운 대화를 만드는 도우미입니다. 대화에는 '{meaning}'을(를) 의미하는 의성어/의태어 '{word}'가 포함되어야 합니다. 대화가 자연스럽고 문맥에 적절하게 들리도록 만드세요.",
                'user_prompt_set1': """조건1: '{meaning}'을(를) 의미하는 의성어/의태어 '{word}'를 자연스럽게 포함하는 두 사람의 대화를 만들어주세요.\n
                    조건2: 대화는 {num_utterances}번의 발화(={num_turns}턴)로 이루어져야 합니다.\n
                    단계적으로 생각해봅시다.""",
                'user_prompt_set2': """조건3: 의성어/의태어는 반드시 {ss_idx}번째 발화에 표기되어야 합니다. 단어가 자연스럽게 포함되기 위하여 기존의 문맥이 바뀌어도 됩니다.\n
                    조건4: 주어진 대화에서 의성/의태어의 의미가 직접적으로 포함되어 있는지 확인하고, 가능한 경우 그 표현을 피하여 자연스럽고 창의적인 예문을 만들어주세요.\n
                    단계적으로 생각해봅시다.""",
                'user_prompt_set3': """조건5: 대화에 등장할 수 있는 이름은 그 언어 사용자에서 존재하는 이름으로 하세요. 인물은 2명입니다.\n
                    조건6: 대화에서 인사말과 헤어지는 표현을 포함하는 경우, 그 발화를 대체해야 합니다.\n
                    단계적으로 생각해봅시다.""",
                'dialogue_format': """[ {"화자": "사람1", "발화 번호": 1, "발화": "발화 내용"},
                    {"화자": "사람2", "발화 번호": 2, "발화": "발화 내용"},
                    {"화자": "사람1", "발화 번호": 3, "발화": "발화 내용"},
                    {"화자": "사람2", "발화 번호": 4, "발화": "발화 내용"},
                    {"화자": "사람1", "발화 번호": 5, "발화": "발화 내용"},
                    {"화자": "사람2", "발화 번호": 6, "발화": "발화 내용"}
                ]"""
            },
            'ja': {
                'system_prompt': "あなたは二人の間で自然な対話を作成する役立つアシスタントです。対話には「{meaning}」を意味するオノマトペ「{word}」を含める必要があります。対話が自然で文脈に適切に聞こえるようにしてください。",
                'user_prompt_set1': """条件1: 「{meaning}」を意味するオノマトペ「{word}」を自然に取り入れた二人の間の対話を作成してください。\n
                    条件2: 対話は{num_utterances}回の発話(={num_turns}ターン)で構成され、自然に感じられるようにしてください。\n
                    ステップバイステップで考えましょう。""",
                'user_prompt_set2': """条件3: オノマトペは必ず{ss_idx}番目の発話に表記されるようにしてください。単語を自然に含めるために、既存の文脈が変わっても構いません。\n
                    条件4: 対話の中にオノマトペの意味が直接含まれているかを確認し、可能であればその表現を避け、自然で創造的な例文を作成してください。\n
                    ステップバイステップで考えましょう。""",
                'user_prompt_set3': """条件5: 対話に登場する名前はその言語の話者に存在する名前にしてください。登場人物は2人です。\n
                    条件6: 対話に挨拶や別れの表現が含まれている場合、その発話を置き換えてください。\n
                    ステップバイステップで考えましょう。""",
                'dialogue_format': """[ {"話者": "人物1", "発話番号": 1, "発話": "発話内容"},
                    {"話者": "人物2", "発話番号": 2, "発話": "発話内容"},
                    {"話者": "人物1", "発話番号": 3, "発話": "発話内容"},
                    {"話者": "人物2", "発話番号": 4, "発話": "発話内容"},
                    {"話者": "人物1", "発話番号": 5, "発話": "発話内容"},
                    {"話者": "人物2", "発話番号": 6, "発話": "発話内容"}
                ]"""
            }
        }
        return templates
    
    def generate_dialogue(self, word_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dialogue for a given word using a step-by-step approach"""
        word = word_data.get('word', '')
        
        # Get the first meaning if it's a list, or use the meaning directly
        meaning = word_data.get('meaning', [])
        if isinstance(meaning, list) and meaning:
            meaning = meaning[0]
        elif not meaning:
            meaning = "an onomatopoeic word"
        
        num_utterances = 6
        ss_idx = random.randint(1, num_utterances)
        num_turns = num_utterances // 2
        
        # Format system prompt
        system_prompt = self.templates[self.language]['system_prompt'].format(word=word, meaning=meaning)
        
        try:
            # STEP 1: Generate initial dialogue with first set of conditions
            user_prompt_set1 = self.templates[self.language]['user_prompt_set1'].format(
                word=word, 
                meaning=meaning, 
                num_utterances=num_utterances, 
                num_turns=num_turns
            )
            
            response1 = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_set1},
                    {"role": "user", "content": self.templates[self.language]['dialogue_format']}
                ],
                temperature=1,
                max_tokens=1000
            )
            
            dialogue1 = response1.choices[0].message.content.strip()
            
            # STEP 2: Apply second set of conditions to the initial dialogue
            user_prompt_set2 = self.templates[self.language]['user_prompt_set2'].format(
                ss_idx=ss_idx
            )
            
            response2 = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_set1},
                    {"role": "assistant", "content": dialogue1},
                    {"role": "user", "content": user_prompt_set2}
                ],
                temperature=1,
                max_tokens=1000
            )
            
            dialogue2 = response2.choices[0].message.content.strip()
            
            # STEP 3: Apply final set of conditions
            user_prompt_set3 = self.templates[self.language]['user_prompt_set3']
            
            response3 = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_set1},
                    {"role": "assistant", "content": dialogue1},
                    {"role": "user", "content": user_prompt_set2},
                    {"role": "assistant", "content": dialogue2},
                    {"role": "user", "content": user_prompt_set3}
                ],
                temperature=1,
                max_tokens=1000
            )
            
            final_dialogue = response3.choices[0].message.content.strip()
            
            # Parse the final dialogue
            parsed_dialogue = self._parse_dialogue(final_dialogue)
            
            # Verify that the word appears in the correct utterance
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                # Check if the word is in the correct utterance and not in others
                word_in_correct_place = False
                word_in_wrong_place = False
                
                for i, utterance in enumerate(parsed_dialogue):
                    utterance_text = utterance.get("text", "") if isinstance(utterance, dict) else ""
                    if i == ss_idx - 1:
                        if word in utterance_text:
                            word_in_correct_place = True
                    else:
                        if word in utterance_text:
                            word_in_wrong_place = True
                
                if word_in_correct_place and not word_in_wrong_place:
                    break
                
                # If validation fails, retry with more explicit instructions
                retry_count += 1
                
                # Create a more explicit prompt for fixing the dialogue
                fix_prompt = f"""The dialogue needs to be fixed. The word '{word}' must ONLY appear in the {ss_idx}th utterance, not in any other utterance.
                    Current dialogue: {final_dialogue}
                    Please rewrite the dialogue to fix this issue. Make sure the word '{word}' appears only in utterance #{ss_idx}.
                    {self.templates[self.language]['dialogue_format']}"""
                
                fix_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": fix_prompt}
                    ],
                    temperature=1,
                    max_tokens=1000
                )
                
                final_dialogue = fix_response.choices[0].message.content.strip()
                parsed_dialogue = self._parse_dialogue(final_dialogue)
                
                if retry_count >= max_retries:
                    print(f"Warning: Could not place '{word}' correctly after {max_retries} attempts.")
            
            # Create result dictionary
            result = {
                'word': word,
                'meaning': meaning,
                'dialogue': parsed_dialogue,
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
            # Return empty dialogue on error
            return {
                'word': word,
                'meaning': meaning,
                'dialogue': [],
                "meta_data": {
                    'language': self.language,
                    "num_utterances": num_utterances,
                    "ss_idx": ss_idx,
                    "success": False,
                },
            }
    
    def _parse_dialogue(self, dialogue_text: str) -> List[Dict[str, str]]:
        """Parse dialogue text into a structured list of utterances"""
        result = []
        
        # Clean up the dialogue text
        dialogue_text = dialogue_text.strip()
        
        # Remove any markdown code block formatting
        dialogue_text = re.sub(r'```(?:json)?\s*|\s*```', '', dialogue_text)
        
        # Try to parse as JSON if it looks like a JSON array
        if dialogue_text.startswith('[') and dialogue_text.endswith(']'):
            try:
                # Try to parse as JSON
                parsed_list = json.loads(dialogue_text)
                if isinstance(parsed_list, list):
                    for item in parsed_list:
                        if isinstance(item, dict):
                            # Handle different key formats based on language
                            if self.language == 'ko':
                                # Check if we need to parse from the text field
                                if 'speaker' in item and 'text' in item and item['text'].startswith('{"화자"'):
                                    try:
                                        # Extract the embedded JSON from the text field
                                        text_content = item['text']
                                        # Find where the actual text part begins
                                        parts = text_content.split('발화": "', 1)
                                        if len(parts) == 2:
                                            # Extract the actual text (removing the trailing "})
                                            actual_text = parts[1].rsplit('"}', 1)[0]
                                            
                                            # Extract speaker name
                                            speaker_match = re.search(r'"화자"\s*:\s*"([^"]+)"', text_content)
                                            speaker = speaker_match.group(1) if speaker_match else ""
                                            
                                            # Extract index
                                            index_match = re.search(r'"발화 번호"\s*:\s*(\d+)', text_content)
                                            index = int(index_match.group(1)) if index_match else 0
                                            
                                            result.append({
                                                "speaker": speaker,
                                                "index": index,
                                                "text": actual_text
                                            })
                                        else:
                                            # Fallback if parsing fails
                                            result.append({
                                                "speaker": item.get('speaker', ''),
                                                "index": item.get('index', 0),
                                                "text": item.get('text', '')
                                            })
                                    except Exception as e:
                                        print(f"Error parsing Korean dialogue: {e}")
                                        result.append({
                                            "speaker": item.get('speaker', ''),
                                            "index": item.get('index', 0),
                                            "text": item.get('text', '')
                                        })
                                else:
                                    # Normal case where fields are already properly separated
                                    speaker = item.get('화자', '')
                                    utterance_num = item.get('발화 번호', 0)
                                    text = item.get('발화', '')
                                    
                                    result.append({
                                        "speaker": speaker,
                                        "index": utterance_num,
                                        "text": text
                                    })
                            elif self.language == 'ja':
                                speaker = item.get('話者', '')
                                utterance_num = item.get('発話番号', 0)
                                text = item.get('発話', '')
                                
                                result.append({
                                    "speaker": speaker,
                                    "index": utterance_num,
                                    "text": text
                                })
                            else:  # en, fr
                                speaker = item.get('speaker', '')
                                utterance_num = item.get('utterance_number', 0)
                                text = item.get('text', '')
                                
                                result.append({
                                    "speaker": speaker,
                                    "index": utterance_num,
                                    "text": text
                                })
                    return result
            except json.JSONDecodeError:
                # If JSON parsing fails, continue to other methods
                pass
        
        # If JSON parsing failed, try to parse using ast.literal_eval
        try:
            parsed_list = ast.literal_eval(dialogue_text)
            if isinstance(parsed_list, list):
                for item in parsed_list:
                    if isinstance(item, dict):
                        # Similar handling as above
                        if self.language == 'ko':
                            speaker = item.get('화자', '')
                            utterance_num = item.get('발화 번호', 0)
                            text = item.get('발화', '')
                        elif self.language == 'ja':
                            speaker = item.get('話者', '')
                            utterance_num = item.get('発話番号', 0)
                            text = item.get('発話', '')
                        else:  # en, fr
                            speaker = item.get('speaker', '')
                            utterance_num = item.get('utterance_number', 0)
                            text = item.get('text', '')
                        
                        result.append({
                            "speaker": speaker,
                            "index": utterance_num,
                            "text": text
                        })
                return result
        except (SyntaxError, ValueError):
            # If ast parsing fails, continue to line-by-line parsing
            pass
        
        # If structured parsing failed, try line-by-line parsing as a last resort
        lines = dialogue_text.split('\n')
        utterance_index = 1
        
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                text = parts[1].strip()
                result.append({
                    "speaker": speaker,
                    "index": utterance_index,
                    "text": text
                })
                utterance_index += 1
        
        return result
    
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
    
    def run(self, limit: Optional[int] = None, shuffle: bool = True) -> None:
        """Run the dialogue generation process"""
        print(f"Starting dialogue generation for language: {self.language}")
        
        # Generate dialogues
        dialogues = self.generate_all_dialogues(limit=limit, shuffle=shuffle)
        
        # Save all dialogues to a single file
        success = self.save_results(dialogues)
        
        print(f"Dialogue generation completed. Generated {len(dialogues)} dialogues.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dialogue Generation Tool')
    parser.add_argument('--language', '-l', required=True, choices=['en', 'fr', 'ko', 'ja'], 
                        help='Language code (en/fr/ko/ja)')
    parser.add_argument('--model', '-m', default="gpt-4o", 
                        help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--limit', '-n', type=int, help='Limit the number of dialogues to generate')
    parser.add_argument('--no-shuffle', action='store_true', help='Do not shuffle the data before processing')
    
    args = parser.parse_args()
    
    generator = DialogueGenerator(
        language=args.language,
        model=args.model
    )
    
    generator.run(limit=args.limit, shuffle=not args.no_shuffle) 