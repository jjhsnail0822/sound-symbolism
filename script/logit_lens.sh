for input_type in original ipa audio original_and_audio ipa_and_audio
do
  python src/experiments/semantic_dimension/qwen_omni_inference_logit_lens.py -w common -i "${input_type}"
done