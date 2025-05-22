import os
from pathlib import Path
from dotenv import load_dotenv
import json
import time
import re
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from tqdm import tqdm
import argparse
from typing import Optional, List, Dict, Any  # Added List, Dict, Any

env_path = Path('.env.local')
load_dotenv(dotenv_path=env_path)

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)

MODEL_NAME = "gpt-4.1"
INPUT_DIR = "data/dialogues"
OUTPUT_DIR = "data/dialogues/nat"

# Define languages to process and their full names for the prompt
LANGUAGES_TO_PROCESS = {
    "fr": "French",
    "ja": "Japanese",
    "ko": "Korean"
}

# Retry mechanism configuration
MAX_RETRIES = 10
RETRY_DELAY_SECONDS = 5

# --- Prompt Loading ---
try:
    with open('data/prompts/prompts.json', 'r', encoding='utf-8') as f:
        prompts = json.load(f)
        if 'dialogue_translation' not in prompts or 'user_prompt' not in prompts['dialogue_translation']:
            raise KeyError("Prompt structure 'dialogue_translation.user_prompt' not found in prompts.json")
        BASE_PROMPT_TEMPLATE = prompts['dialogue_translation']['user_prompt']
        if 'retranslation_prompt' not in prompts['dialogue_translation']:
            raise KeyError("Prompt structure 'dialogue_translation.retranslation_prompt' not found in prompts.json")
        RETRANSLATION_PROMPT_BRACKETS = prompts['dialogue_translation']['retranslation_prompt']
except FileNotFoundError:
    raise FileNotFoundError("Error: prompts.json not found. Make sure the path 'data/prompts/prompts.json' is correct.")
except json.JSONDecodeError as e:
    raise json.JSONDecodeError(f"Error decoding prompts.json: {e.msg}", e.doc, e.pos)
except KeyError as e:
    raise KeyError(f"Error accessing prompt in prompts.json: {e}")

# --- Translation Function (Handles API call and basic parsing) ---
def translate_dialogue_api_call(
    dialogue_obj,
    source_language_name,
    additional_instruction: Optional[str] = None  # Added parameter
    ):
    """
    Makes the API call for translation and handles basic response cleaning/parsing.
    Separated to facilitate retries on specific errors. Includes optional additional instructions.

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

    # Append additional instruction if provided
    if additional_instruction:
        prompt += f"\n\nIMPORTANT ADDITIONAL INSTRUCTION: {additional_instruction}"
        print(f"      [API Call with additional instruction: {additional_instruction}]")  # Log that instruction is added

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
        raise ValueError("Received empty JSON string from API.")

    translated_dialogue_obj = json.loads(translated_json_string)
    return translated_dialogue_obj

# --- Consistency Check Functions ---
def check_keys(original_utterances, translated_utterances):
    """Checks if keys in translated utterances match the original."""
    if not original_utterances or not translated_utterances:
        return False
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
    bracket_pattern = re.compile(r"\[.*?\]")
    found_bracket_in_target = False
    found_bracket_elsewhere = False

    if not translated_utterances:
        print("    Consistency Check Info: Cannot check brackets, translated utterances list is empty.")
        return False

    for utterance in translated_utterances:
        text = utterance.get('text', '')
        has_brackets = bool(bracket_pattern.search(text))
        utterance_idx = utterance.get('index')

        if utterance_idx == ss_idx:
            if has_brackets:
                found_bracket_in_target = True
            else:
                print(f"    Consistency Check Failed: Brackets missing in target utterance (index {ss_idx}). Text: '{text}'")
                return False
        elif has_brackets:
            found_bracket_elsewhere = True
            print(f"    Consistency Check Failed: Unexpected brackets found in non-target utterance (index {utterance_idx}). Text: '{text}'")
            return False

    if not found_bracket_in_target:
        print(f"    Consistency Check Failed: Brackets missing in target utterance (index {ss_idx}) after checking all.")
        return False

    return True

# --- Manual Bracket Cleaning Function ---
def manually_clean_brackets(utterances: List[Dict[str, Any]], ss_idx: int) -> List[Dict[str, Any]]:
    """Removes brackets from utterances where index is not ss_idx."""
    cleaned_utterances = []
    for utt in utterances:
        utt_copy = utt.copy()
        if utt_copy.get('index') != ss_idx:
            text = utt_copy.get('text', '')
            cleaned_text = text.replace('[', '').replace(']', '')
            if text != cleaned_text:
                print(f"      Manually removed brackets from utterance index {utt_copy.get('index')}")
            utt_copy['text'] = cleaned_text
        cleaned_utterances.append(utt_copy)
    return cleaned_utterances

# --- Refactored Translation and Validation Function ---
def translate_and_validate_dialogue(
    original_dialogue_entry: dict,
    subject_word: str,
    source_language_name: str,
    original_utterances_for_check: list,
    is_retranslation: bool = False,
    bracket_check_failed: bool = False
    ):
    """
    Preprocesses, translates (with retries), and validates a single dialogue entry.
    Includes logic to add specific instructions during re-translation attempts.

    Returns:
        The translated and validated dialogue list (value for the 'dialogue' key),
        or None if translation/validation fails after retries.
    """
    if 'dialogue' not in original_dialogue_entry or not isinstance(original_dialogue_entry['dialogue'], list):
        print(f"    Skipping dialogue: 'dialogue' key missing or not a list.")
        return None
    if 'meta_data' not in original_dialogue_entry or 'ss_idx' not in original_dialogue_entry['meta_data']:
        print(f"    Skipping dialogue: 'meta_data' or 'ss_idx' missing.")
        return None

    ss_idx = original_dialogue_entry['meta_data']['ss_idx']
    preprocessed_dialogue_list = []

    # --- Preprocessing Step (Only needed once, even for re-translation) ---
    try:
        original_utterances = json.loads(json.dumps(original_dialogue_entry['dialogue']))
        for utterance in original_utterances:
            if utterance.get('index') == ss_idx:
                original_text = utterance.get('text', '')
                pattern = r'\b' + re.escape(subject_word) + r'\b'
                replacement = '[' + subject_word + ']'
                replaced_text, num_replacements = re.subn(pattern, replacement, original_text, flags=re.IGNORECASE)
                if num_replacements == 0 and subject_word.lower() in original_text.lower():
                    print(f"      Preprocessing Fallback: Regex failed for '{subject_word}' in '{original_text}'. Using simple replace.")
                    replaced_text = original_text.replace(subject_word, replacement)
                utterance['text'] = replaced_text
            preprocessed_dialogue_list.append(utterance)
    except Exception as preproc_e:
        print(f"    Error during preprocessing dialogue: {preproc_e}. Skipping.")
        return None

    # --- Determine Additional Instruction for Re-translation ---
    additional_instruction_for_api = None
    if is_retranslation and bracket_check_failed:
        additional_instruction_for_api = RETRANSLATION_PROMPT_BRACKETS.format(ss_idx=ss_idx)

    # --- Translation with Retry Logic ---
    translated_utterances = None
    last_error = None
    for attempt in range(MAX_RETRIES):
        dialogue_to_translate = {"dialogue": preprocessed_dialogue_list}
        try:
            translated_dialogue_part = translate_dialogue_api_call(
                dialogue_to_translate,
                source_language_name,
                additional_instruction=additional_instruction_for_api
            )

            if not translated_dialogue_part or 'dialogue' not in translated_dialogue_part:
                last_error = ValueError("Invalid structure in translated response")
                print(f"    Consistency Check Failed (Attempt {attempt + 1}): API response missing 'dialogue' key.")
                raise last_error

            current_translated_utterances = translated_dialogue_part['dialogue']

            if not check_keys(original_utterances_for_check, current_translated_utterances):
                last_error = ValueError("Consistency Check Failed: Keys mismatch.")
                print(f"    Consistency Check Failed (Attempt {attempt + 1}): Keys mismatch.")
                raise last_error

            if not check_brackets(current_translated_utterances, ss_idx):
                last_error = ValueError("Consistency Check Failed: Brackets mismatch.")
                print(f"    Consistency Check Failed (Attempt {attempt + 1}): Brackets mismatch.")
                raise last_error

            translated_utterances = current_translated_utterances
            last_error = None
            break

        except RateLimitError as e:
            last_error = e
            print(f"      Rate limit reached (Attempt {attempt + 1}/{MAX_RETRIES}). Waiting 60s. Error: {e}")
            time.sleep(60)
        except (APIError, APITimeoutError) as e:
            last_error = e
            print(f"      API Error/Timeout (Attempt {attempt + 1}/{MAX_RETRIES}). Waiting {RETRY_DELAY_SECONDS}s. Error: {e}")
            time.sleep(RETRY_DELAY_SECONDS)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            print(f"      Parsing/Consistency/Empty Response Error (Attempt {attempt + 1}/{MAX_RETRIES}). Waiting {RETRY_DELAY_SECONDS}s. Error: {e}")
            time.sleep(RETRY_DELAY_SECONDS)
        except Exception as e:
            last_error = e
            print(f"      An unexpected error occurred (Attempt {attempt + 1}/{MAX_RETRIES}). Waiting {RETRY_DELAY_SECONDS}s. Error: {e}")
            time.sleep(RETRY_DELAY_SECONDS)

    if translated_utterances is not None:
        return translated_utterances
    else:
        print(f"    Translation/Validation failed after {MAX_RETRIES} attempts. Last error: {last_error}")
        return None

# --- Main Processing Function ---
def main(skip_phase_1: bool):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_errors = 0
    global_translation_failures = 0
    global_retranslation_failures = 0
    manual_bracket_fixes = 0

    # === Phase 1: Initial Translation ===
    if not skip_phase_1:
        print("--- Phase 1: Initial Translation ---")
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

            for i, item in enumerate(tqdm(source_data, desc=f"Translating {lang_name}", unit="item")):
                subject_word = item.get('word')
                if not subject_word:
                    continue

                if 'dialogues' not in item or not isinstance(item['dialogues'], list):
                    continue

                translated_dialogues_for_item = []
                item_translation_failed = False

                for j, dialogue_entry in enumerate(item['dialogues']):
                    try:
                        original_dialogue_list = dialogue_entry.get('dialogue', [])
                        if not isinstance(original_dialogue_list, list):
                            raise TypeError("Original dialogue is not a list")
                        original_utterances_for_check = json.loads(json.dumps(original_dialogue_list))
                    except (json.JSONDecodeError, TypeError, Exception) as copy_e:
                        print(f"    Error copying/validating original utterances for dialogue {j+1} in item {i+1} ('{subject_word}'): {copy_e}. Skipping dialogue.")
                        item_translation_failed = True
                        language_had_errors = True
                        total_errors += 1
                        continue

                    translated_utterances = translate_and_validate_dialogue(
                        dialogue_entry, subject_word, lang_name, original_utterances_for_check,
                        is_retranslation=False,
                        bracket_check_failed=False
                    )

                    if translated_utterances is not None:
                        new_dialogue_entry = dialogue_entry.copy()
                        new_dialogue_entry['dialogue'] = translated_utterances
                        translated_dialogues_for_item.append(new_dialogue_entry)
                    else:
                        print(f"    Initial translation FAILED for word '{subject_word}', dialogue {j+1}.")
                        item_translation_failed = True
                        language_had_errors = True
                        global_translation_failures += 1

                if not item_translation_failed:
                    translated_item = item.copy()
                    translated_item['dialogues'] = translated_dialogues_for_item
                    current_lang_translated_data.append(translated_item)

            if current_lang_translated_data:
                print(f"\n  Saving initial translated data for {lang_name} to {output_filepath}...")
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(current_lang_translated_data, f, ensure_ascii=False, indent=4)
                    print(f"  Successfully saved {len(current_lang_translated_data)} initially translated items to {output_filename}.")
                except Exception as e:
                    print(f"  Error writing initial output file {output_filepath}: {e}")
                    total_errors += 1
            elif not language_had_errors:
                print(f"  No data was translated successfully for {lang_name} in Phase 1. Output file not created.")
            else:
                print(f"  No initial data saved for {lang_name} due to processing errors.")
    else:
        print("--- Skipping Phase 1: Initial Translation ---")

    # === Phase 2: Validation and Re-translation ===
    print("\n--- Phase 2: Validation and Re-translation ---")

    for lang_code, lang_name in LANGUAGES_TO_PROCESS.items():
        output_filename = f"{lang_code}2en.json"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        input_filepath = os.path.join(INPUT_DIR, f"{lang_code}.json")

        print(f"\n--- Validating and Re-translating: {lang_name} ({output_filename}) ---")

        if not os.path.exists(output_filepath):
            print(f"  Translated file not found: {output_filepath}. Skipping validation.")
            continue
        if not os.path.exists(input_filepath):
            print(f"  Original source file not found: {input_filepath}. Cannot validate. Skipping.")
            continue

        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                translated_data = json.load(f)
            with open(input_filepath, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            if not isinstance(translated_data, list) or not isinstance(source_data, list):
                print(f"  Error: Expected lists in {output_filepath} or {input_filepath}. Skipping validation.")
                continue
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Error reading or parsing JSON files for validation ({output_filepath} or {input_filepath}): {e}. Skipping language.")
            total_errors += 1
            continue

        original_dialogue_map = {}
        for item in source_data:
            word = item.get('word')
            if word and 'dialogues' in item:
                original_dialogue_map[word] = item.get('dialogues', [])

        needs_resave = False
        dialogues_to_retranslate_count = 0

        for i, translated_item in enumerate(tqdm(translated_data, desc=f"Validating {lang_name}", unit="item")):
            subject_word = translated_item.get('word')
            if not subject_word or subject_word not in original_dialogue_map:
                print(f"    Warning: Cannot find original data for word '{subject_word}' in item {i+1}. Skipping validation for this item.")
                continue

            original_dialogues_list = original_dialogue_map[subject_word]
            translated_dialogues_list = translated_item.get('dialogues', [])

            if len(original_dialogues_list) != len(translated_dialogues_list):
                print(f"    Warning: Dialogue count mismatch for word '{subject_word}'. Original: {len(original_dialogues_list)}, Translated: {len(translated_dialogues_list)}. Skipping detailed validation for this item.")
                continue

            for j, translated_dialogue_entry in enumerate(translated_dialogues_list):
                if j >= len(original_dialogues_list):
                    print(f"    Warning: Index out of bounds when accessing original dialogue for word '{subject_word}', dialogue index {j+1}. Skipping validation.")
                    continue
                original_dialogue_entry = original_dialogues_list[j]
                original_utterances_for_check = original_dialogue_entry.get('dialogue', [])
                translated_utterances = translated_dialogue_entry.get('dialogue', [])
                ss_idx = original_dialogue_entry.get('meta_data', {}).get('ss_idx')

                if ss_idx is None or not original_utterances_for_check:
                    continue

                keys_ok = check_keys(original_utterances_for_check, translated_utterances)
                brackets_ok = check_brackets(translated_utterances, ss_idx)

                if not keys_ok or not brackets_ok:
                    dialogues_to_retranslate_count += 1
                    print(f"  ! Consistency failed for word '{subject_word}', dialogue {j+1} (Keys OK: {keys_ok}, Brackets OK: {brackets_ok}). Attempting re-translation...")

                    retranslated_utterances = translate_and_validate_dialogue(
                        original_dialogue_entry,
                        subject_word,
                        lang_name,
                        original_utterances_for_check,
                        is_retranslation=True,
                        bracket_check_failed=(not brackets_ok)
                    )

                    if retranslated_utterances is not None:
                        print(f"    Re-translation successful for word '{subject_word}', dialogue {j+1}.")
                        translated_data[i]['dialogues'][j]['dialogue'] = retranslated_utterances
                        needs_resave = True
                    else:
                        print(f"    Re-translation FAILED for word '{subject_word}', dialogue {j+1} after retries.")
                        global_retranslation_failures += 1

                        if not brackets_ok:
                            print(f"      Attempting manual bracket cleaning for word '{subject_word}', dialogue {j+1}...")
                            cleaned_utterances = manually_clean_brackets(translated_utterances, ss_idx)

                            if check_brackets(cleaned_utterances, ss_idx):
                                print(f"      Manual bracket cleaning successful for word '{subject_word}', dialogue {j+1}.")
                                translated_data[i]['dialogues'][j]['dialogue'] = cleaned_utterances
                                needs_resave = True
                                manual_bracket_fixes += 1
                            else:
                                print(f"      Manual bracket cleaning FAILED for word '{subject_word}', dialogue {j+1}. Brackets still incorrect.")

        if needs_resave:
            print(f"\n  Re-saving updated translated data for {lang_name} to {output_filepath}...")
            try:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(translated_data, f, ensure_ascii=False, indent=4)
                print(f"  Successfully re-saved updated data for {lang_name}.")
                print(f"  ({dialogues_to_retranslate_count} dialogues flagged for re-translation attempt)")
                if manual_bracket_fixes > 0:
                    print(f"  ({manual_bracket_fixes} dialogues had brackets manually cleaned after re-translation failure)")
            except Exception as e:
                print(f"  Error re-writing updated output file {output_filepath}: {e}")
                total_errors += 1
        else:
            print(f"  No changes needed for {lang_name} after validation.")

    print("\n--- Translation and Validation process finished ---")
    if global_translation_failures > 0:
        print(f"Phase 1: Completed with {global_translation_failures} dialogues failing initial translation attempts.")
    if global_retranslation_failures > 0 or manual_bracket_fixes > 0:
        print(f"Phase 2: Completed with {global_retranslation_failures} dialogues failing re-translation attempts.")
        if manual_bracket_fixes > 0:
            print(f"         ({manual_bracket_fixes} of these failures had brackets manually cleaned as a fallback).")
    if total_errors > 0:
        print(f"Encountered {total_errors} other errors during processing (reading files, preprocessing, saving, etc.).")

    final_failures = global_retranslation_failures
    if total_errors == 0 and global_translation_failures == 0 and final_failures == 0:
        print("Completed successfully with no persistent translation failures.")
    elif total_errors == 0 and global_translation_failures == 0 and final_failures > 0 and final_failures == manual_bracket_fixes:
        print("Completed. All re-translation failures were resolved by manual bracket cleaning.")
    else:
        print("Completed with some persistent errors or failures.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate dialogues and validate consistency.")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip the initial translation phase (Phase 1) and only run validation/re-translation (Phase 2).")
    args = parser.parse_args()

    main(skip_phase_1=args.skip_phase1)