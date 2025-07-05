ROOT="/home/sunahan/workspace/sound-symbolism"
IPA_DICT_DIR="${ROOT}/data/raw/ipa_dict"

DATA_DIR="${ROOT}/data/processed/art"

RESOURCE_DIR="${DATA_DIR}/resources"
SEMANTIC_DIM_PATH="${DATA_DIR}/semantic_dimension/semantic_dimension_binary_gt.json"

cd "${ROOT}/src/dataset/constructed_word"

python3 process_ipa_word_dict.py \
    --data_dir=$IPA_DICT_DIR \
    --output_dir=$RESOURCE_DIR


python3 generate_word.py \
    --data_dir=$RESOURCE_DIR \
    --output_path=$SEMANTIC_DIM_PATH \
    --ensure_nonword

python3 annotate_score.py \
    --data_dir=$RESOURCE_DIR \
    --json_path=$SEMANTIC_DIM_PATH

python3 ../audio/tts_generation_google.py \
    --word_type='art' \
    --base_dir="${ROOT}/data/processed"