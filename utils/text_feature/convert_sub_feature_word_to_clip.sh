#!/usr/bin/env bash
# Usage:
# bash utils/text_feature/convert_sub_feature_word_to_clip.sh POOL_TYPE CLIP_LENGTH [--debug]

#pool_type=$1  # [max, avg]
#clip_length=$2
#sub_token_h5_file=$3
#sub_clip_h5_file=$4 # tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5
#vid_clip_h5_file=$5  # .h5 file stores the clip-level video features, to make sure subtitle clip-level features have the same length as the video features.
#sub_meta_path=data/tvqa_preprocessed_subtitles.jsonl

pool_type='max'  # [max, avg]
clip_length=1.5
sub_token_h5_file='data/tvr_feature_release/bert_feature/sub_only2/tvr_sub_pretrained_w_sub.h5'
sub_clip_h5_file='tvr_sub_pretrained_w_sub_max_cl-1.5.h5'
vid_clip_h5_file='data/tvr_feature_release/video_feature/tvr_resnet152_rgb_max_i3d_rgb600_avg_cat_cl-1.5.h5'  # .h5 file stores the clip-level video features, to make sure subtitle clip-level features have the same length as the video features.
sub_meta_path=data/tvqa_preprocessed_subtitles.jsonl


python utils/text_feature/convert_sub_feature_word_to_clip.py \
--pool_type ${pool_type} \
--clip_length ${clip_length} \
--src_h5_file ${sub_token_h5_file} \
--tgt_h5_file ${sub_clip_h5_file} \
--sub_meta_path ${sub_meta_path} \
--vid_clip_h5_file ${vid_clip_h5_file} \
${@:3}
