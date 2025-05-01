import os
from pathlib import Path
from dotenv import load_dotenv
import json
import time
import re
from openai import OpenAI, RateLimitError, APIError, APITimeoutError

# --- Configuration ---
# Load environment variables from .env.local file
env_path = Path('.env.local')
load_dotenv(dotenv_path=env_path)

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)  # Added timeout to client

MODEL_NAME = "gpt-4.1"
INPUT_DIR = "dataset/2_dialogue/nat"
OUTPUT_DIR = "dataset/2_dialogue/nat"

# Define languages to process and their full names for the prompt
LANGUAGES_TO_PROCESS = {
    "fr": "French",
    "ja": "Japanese",
    "ko": "Korean"
}

# Retry mechanism configuration
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 15  # Increased delay for retries

# --- Prompt Loading ---
try:
    with open('analysis/experiments/prompts.json', 'r', encoding='utf-8') as f:
        prompts = json.load(f)
        if 'dialogue_translation' not in prompts or 'user_prompt' not in prompts['dialogue_translation']:
            raise KeyError("Prompt structure 'dialogue_translation.user_prompt' not found in prompts.json")
        BASE_PROMPT_TEMPLATE = prompts['dialogue_translation']['user_prompt']
except FileNotFoundError:
    raise FileNotFoundError("Error: prompts.json not found. Make sure the path 'analysis/experiments/prompts.json' is correct.")
except json.JSONDecodeError as e:
    raise json.JSONDecodeError(f"Error decoding prompts.json: {e.msg}", e.doc, e.pos)
except KeyError as e:
    raise KeyError(f"Error accessing prompt in prompts.json: {e}")

# --- Translation Function (Handles API call and basic parsing) ---
def translate_dialogue_api_call(dialogue_obj, source_language_name):
    """
    Makes the API call for translation and handles basic response cleaning/parsing.
    Separated to facilitate retries on specific errors.

    Returns:
        dict: Parsed translated dialogue object from API.
    Raises:
        RateLimitError, APIError, APITimeoutError, json.JSONDecodeError, ValueError (for empty response), Exception
    """
    dialogue_json_string = json.dumps(dialogue_obj, ensure_ascii=False, indent=4)
    prompt = BASE_PROMPT_TEMPLATE.format(
        source_language=source_language_name,
        json_dialogue_string=dialogue_json_string
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    translated_json_string = response.choices[0].message.content.strip()

    # Clean potential markdown code blocks
    if translated_json_string.startswith("```json"):
        translated_json_string = translated_json_string[7:]
    if translated_json_string.endswith("```"):
        translated_json_string = translated_json_string[:-3]
    translated_json_string = translated_json_string.strip()

    if not translated_json_string:
        # Raise an error for empty response to trigger retry
        raise ValueError("Received empty JSON string from API.")

    # Parse JSON - can raise json.JSONDecodeError
    translated_dialogue_obj = json.loads(translated_json_string)
    return translated_dialogue_obj

# --- Consistency Check Functions ---
def check_keys(original_utterances, translated_utterances):
    """Checks if keys in translated utterances match the original."""
    if not original_utterances or not translated_utterances:
        return False  # Cannot compare if one is empty
    if len(original_utterances) != len(translated_utterances):
        print("    Consistency Check Failed: Number of utterances mismatch.")
        return False

    original_keys = set(original_utterances[0].keys()) if original_utterances else set()
    for i, trans_utt in enumerate(translated_utterances):
        trans_keys = set(trans_utt.keys())
        if trans_keys != original_keys:
            print(f"    Consistency Check Failed: Key mismatch in utterance {i+1}. Expected {original_keys}, Got {trans_keys}")
            return False
    return True

def check_brackets(translated_utterances, ss_idx):
    """Checks bracket presence only in the target utterance."""
    bracket_pattern = re.compile(r"\[.*?\]")  # Simple check for brackets
    found_bracket_in_target = False
    found_bracket_elsewhere = False

    for utterance in translated_utterances:
        text = utterance.get('text', '')
        has_brackets = bool(bracket_pattern.search(text))
        utterance_idx = utterance.get('index')

        if utterance_idx == ss_idx:
            if has_brackets:
                found_bracket_in_target = True
            else:
                # If brackets are MISSING in the target utterance, it's a failure
                print(f"    Consistency Check Failed: Brackets missing in target utterance (index {ss_idx}). Text: '{text}'")
                return False
        elif has_brackets:
            # If brackets are PRESENT in a non-target utterance, it's a failure
            found_bracket_elsewhere = True
            print(f"    Consistency Check Failed: Unexpected brackets found in non-target utterance (index {utterance_idx}). Text: '{text}'")
            return False  # Fail fast

    if not found_bracket_in_target:
        # This case is covered above, but as a safeguard
        print(f"    Consistency Check Failed: Brackets missing in target utterance (index {ss_idx}) after checking all.")
        return False

    # If we reach here, brackets were found ONLY in the target utterance
    return True

# --- Main Processing Loop ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_errors = 0
    global_translation_failures = 0  # Count dialogues that failed all retries

    for lang_code, lang_name in LANGUAGES_TO_PROCESS.items():
        input_filepath = os.path.join(INPUT_DIR, f"{lang_code}.json")
        output_filename = f"{lang_code}2en.json"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)

        print(f"\n--- Processing language: {lang_name} ({lang_code}.json) ---")

        if not os.path.exists(input_filepath):
            print(f"  Input file not found: {input_filepath}. Skipping.")
            continue

        try:
            with open(input_filepath, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            if not isinstance(source_data, list):
                print(f"  Error: Expected a list in {input_filepath}, but got {type(source_data)}. Skipping.")
                continue
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Error reading or parsing source JSON file {input_filepath}: {e}. Skipping language.")
            total_errors += 1
            continue

        current_lang_translated_data = []
        language_had_errors = False

        for i, item in enumerate(source_data):
            subject_word = item.get('word')
            if not subject_word:
                print(f"    Skipping item {i+1}: 'word' key missing.")
                continue
            print(f"  Processing item {i+1}/{len(source_data)} (Word: {subject_word})...")

            if 'dialogues' not in item or not isinstance(item['dialogues'], list):
                print(f"    Skipping item {i+1}: 'dialogues' key missing or not a list.")
                continue

            translated_dialogues_for_item = []
            item_translation_failed = False  # Tracks if any dialogue within the item failed permanently

            for j, dialogue_entry in enumerate(item['dialogues']):
                if 'dialogue' not in dialogue_entry or not isinstance(dialogue_entry['dialogue'], list):
                    print(f"    Skipping dialogue {j+1} in item {i+1}: 'dialogue' key missing or not a list.")
                    continue
                if 'meta_data' not in dialogue_entry or 'ss_idx' not in dialogue_entry['meta_data']:
                    print(f"    Skipping dialogue {j+1} in item {i+1}: 'meta_data' or 'ss_idx' missing.")
                    continue

                ss_idx = dialogue_entry['meta_data']['ss_idx']
                preprocessed_dialogue_list = []
                original_utterances_for_check = []  # Store original structure for key check

                # --- Preprocessing Step ---
                try:
                    # Deep copy for preprocessing
                    original_utterances = json.loads(json.dumps(dialogue_entry['dialogue']))
                    original_utterances_for_check = json.loads(json.dumps(dialogue_entry['dialogue']))  # Keep a clean copy
                    for utterance in original_utterances:
                        if utterance.get('index') == ss_idx:
                            original_text = utterance.get('text', '')
                            pattern = r'\b' + re.escape(subject_word) + r'\b'
                            replacement = '[' + subject_word + ']'
                            replaced_text = re.sub(pattern, replacement, original_text, flags=re.IGNORECASE)
                            if replacement not in replaced_text:
                                replaced_text = original_text.replace(subject_word, replacement)
                            utterance['text'] = replaced_text
                        preprocessed_dialogue_list.append(utterance)
                except Exception as preproc_e:
                    print(f"    Error during preprocessing dialogue {j+1} in item {i+1}: {preproc_e}. Skipping dialogue.")
                    item_translation_failed = True
                    language_had_errors = True
                    total_errors += 1
                    continue  # Skip to the next dialogue_entry

                # --- Translation with Retry Logic ---
                translated_dialogue_part = None
                for attempt in range(MAX_RETRIES):
                    print(f"    Translating dialogue {j+1}/{len(item['dialogues'])} (Attempt {attempt + 1}/{MAX_RETRIES})...")
                    dialogue_to_translate = {"dialogue": preprocessed_dialogue_list}
                    try:
                        # Make the API call
                        translated_dialogue_part = translate_dialogue_api_call(dialogue_to_translate, lang_name)

                        # --- Consistency Checks ---
                        if not translated_dialogue_part or 'dialogue' not in translated_dialogue_part:
                            print("    Consistency Check Failed: API response missing 'dialogue' key.")
                            raise ValueError("Invalid structure in translated response")  # Trigger retry

                        translated_utterances = translated_dialogue_part['dialogue']

                        # 1. Key Check
                        if not check_keys(original_utterances_for_check, translated_utterances):
                            # Error message printed within check_keys
                            raise ValueError("Consistency Check Failed: Keys mismatch.")  # Trigger retry

                        # 2. Bracket Check
                        if not check_brackets(translated_utterances, ss_idx):
                            # Error message printed within check_brackets
                            raise ValueError("Consistency Check Failed: Brackets mismatch.")  # Trigger retry

                        # If all checks pass
                        print(f"      Translation and Consistency Checks successful (Attempt {attempt + 1}).")
                        # debug
                        # print(f"Translated dialogue: {translated_dialogue_part}")
                        break  # Exit retry loop on success

                    except RateLimitError as e:
                        print(f"      Rate limit reached (Attempt {attempt + 1}). Waiting 60s. Error: {e}")
                        time.sleep(60)
                        # Let the loop continue for retry
                    except (APIError, APITimeoutError) as e:
                        print(f"      API Error/Timeout (Attempt {attempt + 1}). Waiting {RETRY_DELAY_SECONDS}s. Error: {e}")
                        time.sleep(RETRY_DELAY_SECONDS)
                        # Let the loop continue for retry
                    except (json.JSONDecodeError, ValueError) as e:  # Catch JSON errors and our custom ValueErrors
                        print(f"      Parsing/Consistency/Empty Response Error (Attempt {attempt + 1}). Waiting {RETRY_DELAY_SECONDS}s. Error: {e}")
                        time.sleep(RETRY_DELAY_SECONDS)
                        # Let the loop continue for retry
                    except Exception as e:
                        print(f"      An unexpected error occurred (Attempt {attempt + 1}). Waiting {RETRY_DELAY_SECONDS}s. Error: {e}")
                        total_errors += 1
                        time.sleep(RETRY_DELAY_SECONDS)
                        # Let the loop continue for retry

                # --- After Retry Loop ---
                if translated_dialogue_part:
                    # Reconstruct the full dialogue entry if translation succeeded
                    new_dialogue_entry = dialogue_entry.copy()
                    new_dialogue_entry['dialogue'] = translated_dialogue_part['dialogue']
                    translated_dialogues_for_item.append(new_dialogue_entry)
                else:
                    # If loop finished without success
                    print(f"      Translation failed for dialogue {j+1} after {MAX_RETRIES} attempts.")
                    item_translation_failed = True  # Mark item as failed
                    language_had_errors = True
                    global_translation_failures += 1  # Increment global failure count

            # --- After processing all dialogues for an item ---
            if not item_translation_failed:
                translated_item = item.copy()
                translated_item['dialogues'] = translated_dialogues_for_item
                current_lang_translated_data.append(translated_item)
            else:
                print(f"    Skipping item {i+1} (Word: {subject_word}) for final output due to dialogue translation failures.")

        # --- After processing all items for a language ---
        if current_lang_translated_data:
            print(f"\n  Saving translated data for {lang_name} to {output_filepath}...")
            try:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(current_lang_translated_data, f, ensure_ascii=False, indent=4)
                print(f"  Successfully saved {len(current_lang_translated_data)} translated items to {output_filename}.")
            except Exception as e:
                print(f"  Error writing output file {output_filepath}: {e}")
                total_errors += 1
        elif not language_had_errors:
            print(f"  No data was translated successfully for {lang_name}. Output file not created.")
        else:
            print(f"  No data saved for {lang_name} due to processing errors.")

    # --- Final Summary ---
    print("\n--- Translation process finished ---")
    if global_translation_failures > 0:
        print(f"Completed with {global_translation_failures} dialogues failing all translation attempts.")
    if total_errors > 0:
        print(f"Encountered {total_errors} other errors during processing (reading files, preprocessing, saving).")
    if total_errors == 0 and global_translation_failures == 0:
        print("Completed successfully.")

if __name__ == "__main__":
    main()