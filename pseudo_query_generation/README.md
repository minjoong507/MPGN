# Pseudo Query Generation

### Pre-requisites

The code requires installing packages to leverage [BLIP](https://github.com/salesforce/BLIP) and [BART](https://huggingface.co/facebook/bart-large) models.

```angular2html
git clone https://github.com/salesforce/BLIP.git
```

Also, you should download the [video frames](https://tvqa.cs.unc.edu/download_tvqa.html#tvqa-download-4).

Please make sure the path in `generate_pseudo_query.sh` is correct.

### Pseudo Query Generation
```angular2html
bash pseudo_query_generation/generate_pseudo_query.sh SAMPLING_METHOD
```

`SAMPLING_METHOD` could be `sub` where defines the temporal moment based on timestamp of subtitles.

The length of temporal moments depends on `min_length` and `max_length` in `generate_pseudo_query.py`.

We also provide `rand` and `fixed` options for `SAMPLING_METHOD`.

After pseudo queries are generated, you will get `tvr_pseudo_query_info_by_{SAMPLING_METHOD}.jsonl`.

The sample of generated annotation:
```
{'vid_name': 's06e03_seg01_clip_00', 'desc_id': 3003, 'character': ['Howard'], 
'dialogue': " Howard : Hey, Bernie. Hey. How's my little astronauttie hottie?  Howard : Okay, I guess.  Howard : It's just being cooped up in this tin can for weeks on end is starting to get to me. ", 
'pseudo_sub_query': 'Howard has been cooped up in a tin can for weeks on end.', 
'pseudo_vid_query': 'Howard is speaking. four balls on a spirally circular object in the sky.', 
'ts': [3.47, 14.433], 'length': 10.963}
```

We have provided `tvr_pseudo_query_info_by_subtitle_based_2_3.jsonl`, `tvr_pseudo_query_info_by_subtitle_based_2_5.jsonl`, and `tvr_pseudo_query_info_by_rand.jsonl`.

To extract features from the pseudo-queries, we follow the same progress provided by [TVR](https://github.com/jayleicn/TVRetrieval/tree/master/utils/text_feature).