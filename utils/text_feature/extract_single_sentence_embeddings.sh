#!/usr/bin/env bash
# Usage:
# bash utils/text_feature/extract_single_sentence_embeddings.sh OUTPUT_ROOT FINETUNE_MODE EXTRACTION_MODE SAVE_FILEPATH
# Examples:
# bash utils/text_feature/extract_single_sentence_embeddings.sh data/tvr_feature_release/bert_feature/ sub_query visual_pseudo tvr_visual_pseudo_query_pretrained_w_sub_query.h5
# bash utils/text_feature/extract_single_sentence_embeddings.sh data/tvr_feature_release/bert_feature/ sub_query textual_pseudo tvr_textual_pseudo_query_pretrained_w_sub_query.h5

output_root=$1
finetune_mode=$2  # sub_query or query_only
extraction_mode=$3  # sub or query
extracted_file_name=$4  # tvr_query_pretrained_w_sub_query.h5, will be saved at output_dir

data_root="data"
train_data_file="${data_root}/tvr_train_release.jsonl"
val_data_file="${data_root}/tvr_val_release.jsonl"
test_data_file1="${data_root}/tvr_test_public_release.jsonl"
sub_data_file="${data_root}/tvqa_preprocessed_subtitles.jsonl"
pseudo_query_data_path="${data_root}/tvr_pseudo_query_info_by_subtitle_based_2_5.jsonl" # path to the pseudo-supervision

="/net/bvisionserver14/playpen-ssd/jielei/data/tvr/bert_feature"
output_dir="${output_root}/${finetune_mode}"
model_type="roberta"
model_name_or_path="${output_dir}/roberta-base_tuned_model"


if [[ ${extraction_mode} == query ]]; then
    max_length=30
    extra_args=(--train_data_file)
    extra_args+=(${train_data_file})
    extra_args+=(${val_data_file})
    extra_args+=(${test_data_file1})
elif [[ ${extraction_mode} == sub ]]; then
    max_length=256
    extra_args=(--use_sub)
    extra_args+=(--sub_data_file)
    extra_args+=(${sub_data_file})
elif [[ ${extraction_mode} == textual_pseudo ]]; then
    max_length=30
    extra_args=(--use_textual_pseudo_query)
    extra_args+=(--pseudo_query_data_path)
    extra_args+=(${pseudo_query_data_path})

elif [[ ${extraction_mode} == visual_pseudo ]]; then
    max_length=30
    extra_args+=(--use_visual_pseudo_query)
    extra_args+=(--pseudo_query_data_path)
    extra_args+=(${pseudo_query_data_path})
fi

python utils/text_feature/lm_finetuning_on_single_sentences.py \
--output_dir ${output_dir} \
--model_type ${model_type} \
--model_name_or_path ${model_name_or_path} \
--do_extract \
--extracted_file_name ${extracted_file_name} \
--block_size ${max_length} \
${extra_args[@]} \
${@:5}
