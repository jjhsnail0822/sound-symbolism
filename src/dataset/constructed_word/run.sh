DATA_ROOT_DIR='/home/sunahan/workspace/sound-symbolism/data/processed/art'
RESOURCE_PATH="${DATA_ROOT_DIR}/resources"
IPA_DICT_PATH="/home/sunahan/workspace/sound-symbolism/data/raw/ipa_dict"


# python3 process_ipa_word_dict.py \
#     --data_dir=$IPA_DICT_PATH \
#     --output_dir=$RESOURCE_PATH


python3 generate_word.py \
    --data_dir=$DATA_ROOT_DIR \
    --ensure_nonword

python3 annotate_score.py \
    --data_dir=$DATA_ROOT_DIR \
    --json_path="${DATA_ROOT_DIR}/constructed_words.json"

python3 ../audio/tts_generation_google.py