train_path="data/tvr_train_release.jsonl"
subtitle_data_path="data/tvqa_preprocessed_subtitles.jsonl"
frame_data_root_path="/data3/mjjung/frames_hq/" # path to the raw frames.
video_feature_root="data/tvr_feature_release/video_feature"
bert_feature_root="data/tvr_feature_release/bert_feature/sub_query"
video_feat_file="${video_feature_root}/tvr_resnet152_rgb_max_i3d_rgb600_avg_cat_cl-1.5.h5"
sub_info_json_file="${bert_feature_root}/tvr_sub_pretrained_w_sub_query_max_cl-1.5_sub_info.json"
tvr_vid_path='data/tvr_vid_path.json'

method=$1

if [[ ${method} == "sub" ]]; then
    echo "Using subtitle-based random sampling method."
    output_path="tvr_pseudo_query_info_by_subtitle_based.jsonl"

elif [[ ${method} == "rand" ]]; then
    echo "Using random sampling method."
    output_path="tvr_pseudo_query_info_by_rand.jsonl"

elif [[ ${method} == "fixed" ]]; then
    echo "Using fix size sampling method."
    output_path="tvr_pseudo_query_info_by_fix.jsonl"
fi

echo "Generate pseudo queries for VCMR"

PYTHONPATH=$PYTHONPATH:. python pseudo_query_generation/generate_pseudo_query.py \
--train_path ${train_path} \
--output_path ${output_path} \
--frame_data_root_path ${frame_data_root_path} \
--subtitle_data_path ${subtitle_data_path} \
--sampling_mode ${method} \
--sub_info_json_file ${sub_info_json_file} \
--video_feat_file ${video_feat_file} \
--tvr_vid_path ${tvr_vid_path}