import random
from utils.basic_utils import load_json, load_jsonl, read_and_save_jsonl, \
    save_jsonl, load_pickle, save_json, uniform_feature_sampling, AverageMeter
import json
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import h5py
import argparse
from tqdm import tqdm
import torch
import os
import numpy as np
import time
from pseudo_query_generation.BLIP.models.blip import blip_decoder
from PIL import Image
from collections import Counter
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn.functional as F

"""
https://github.com/salesforce/BLIP
https://colab.research.google.com/drive/1dul0Sg-TTMy9xZCJzmDRajXbyzDwtYx6?usp=sharing
"""

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


class Generate_Pseudo_Query:
    def __init__(self, opt):
        self.train_path = opt.train_path
        self.subtitles_path = opt.subtitle_data_path
        self.tvr_sub = load_json(opt.sub_info_json_file)
        self.frame_data_root_path = opt.frame_data_root_path
        self.video_feat_h5 = h5py.File(config.video_feat_file, 'r')
        self.min_length = opt.min_length
        self.max_length = opt.max_length
        self.output_path = opt.output_path
        self.device = opt.device
        self.fixed_size = opt.fixed_size
        self.image_size = opt.image_size
        self.sampling_mode = opt.sampling_mode
        self.query_generation_time = AverageMeter()
        self.debug = opt.debug

        if os.path.exists(opt.tvr_vid_path):
            self.vid_list = load_json(opt.tvr_vid_path)
        else:
            self.vid_list = self.set_vid_list(opt.subtitle_data_path, opt.tvr_vid_path)
        self.tv_show_list = ['bbt_frames', 'castle_frames', 'friends_frames', 'grey_frames', 'house_frames',
                             'met_frames']
        self.clip_path_dict = self.set_clip_path_info()

        """
            Load dialog summarization and image captioning models.
            Please check the details of model in the following links.
        """

        summariztion_model_url = 'lidiya/bart-large-xsum-samsum'
        self.tokenizer = AutoTokenizer.from_pretrained(summariztion_model_url, download=True)
        self.text_model = AutoModelForSeq2SeqLM.from_pretrained(summariztion_model_url)

        captioning_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
        self.image_model = blip_decoder(pretrained=captioning_model_url,
                                        image_size=opt.image_size,
                                        vit='base')
        self.pseudo_data = []
        self.flush = False

        if opt.debug:
            logger.info("Debug Mode, original vid list : {}".format(len(self.vid_list)))
            self.vid_list = self.vid_list[:2]

    def flush_output(self):
        del self.pseudo_data
        return []

    def load_image(self, image_path, image_size):
        raw_image = Image.open(image_path).convert('RGB')
        w, h = raw_image.size

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        image = transform(raw_image).unsqueeze(0).cuda()
        return image

    def image_captioning(self, image):
        with torch.no_grad():
            caption = self.image_model.generate(image, sample=True, num_beams=3, max_length=20, min_length=5)
        return caption[0]

    def dialogue_summarization(self, conversation):
        inputs = self.tokenizer("summarize: " + conversation, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.text_model.generate(inputs["input_ids"].cuda(), num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def load_vid_feat(self, vid):
        video_feat = self.video_feat_h5[vid]
        video_feat = uniform_feature_sampling(video_feat, 128)

        return torch.from_numpy(np.array(video_feat))

    def set_clip_path_info(self):
        dict = {}
        for tv_show in self.tv_show_list:
            clip_dir_list = os.listdir(os.path.join(self.frame_data_root_path, tv_show))
            dict[tv_show] = clip_dir_list
        return dict

    def set_vid_list(self, tvr_path, vid_output_dir):
        """
        Input : tvr dataset file.
        Output : tvr video id list and duration of video.
        """
        tvr_data = load_jsonl(tvr_path)
        vid_list = [data['vid_name'] for data in tvr_data]

        logger.info("list of vid json file saved.")
        save_json(vid_list, vid_output_dir)

        return vid_list

    def set_tv_show_for_unidentified_vid(self, name):
        for tv_show in self.clip_path_dict:
            if name in self.clip_path_dict[tv_show]:
                return tv_show

    def extract_character_name_from_subtitle(self, text):
        if ':' in text:
            person = text.split(':')[0].strip()
            if person.lower() not in stopwords.words('english') and 2 < len(person) < 10:
                return person
        else:
            return None

    def generate_vid_pseudo_query(self, character_list, image_root_path, frame_path):
        ch_length = len(character_list)
        if ch_length == 1:
            character_prompt = '{} is speaking.'.format(character_list[0])
        elif ch_length > 1:
            pre_character = ', '.join(character_list[:-1])
            character_prompt = '{} and {} are talking together.'.format(pre_character, character_list[-1])
        else:
            character_prompt = 'Someone is speaking.'

        if len(frame_path) < 2:
            frame_idx = int(len(frame_path) / 2)
        else:
            frame_idx = 0

        image_path = os.path.join(image_root_path, frame_path[frame_idx])
        if not os.path.isfile(image_path):
            raise FileNotFoundError

        image = self.load_image(image_path, self.image_size)
        caption = self.image_captioning(image)

        return character_prompt + ' ' + caption + '.'

    def load_image_path(self, vid_name):
        tv_name = vid_name.split('_')[0]
        tv_name += '_frames'
        if tv_name not in self.tv_show_list:
            tv_name = self.set_tv_show_for_unidentified_vid(vid_name)

        if tv_name is None:
            return None

        image_path = os.path.join(self.frame_data_root_path, tv_name, vid_name)
        return image_path

    def define_temporal_region_by_fixed_subtitles(self, vid, subtitles, size=1):
        temporal_region = []
        total_sub = len(subtitles)
        start_idx = 0

        while start_idx < total_sub:
            conversation = ""
            st_ed_indices = [999, 0]
            character = set()
            end_idx = min(random.randint(start_idx + self.min_length, start_idx + self.max_length), total_sub)

            for txt in subtitles[start_idx:start_idx + size]:
                conversation += txt['text'] + " "
                st_ed_indices[0] = min(txt['start'], st_ed_indices[0])
                st_ed_indices[1] = max(txt['end'], st_ed_indices[1])

                person = self.extract_character_name_from_subtitle(txt['text'])
                if person is not None:
                    character.add(person)

            temporal_region.append(
                {'vid_name': vid, 'st': st_ed_indices[0], 'ed': st_ed_indices[1], 'dialogue': conversation,
                 'length': st_ed_indices[1] - st_ed_indices[0], 'character_name': list(character)})
            start_idx = end_idx + 1

        return temporal_region

    def define_temporal_region_by_subtitles(self, vid, subtitles):
        temporal_region = []
        total_sub = len(subtitles)
        start_idx = 0

        while start_idx < total_sub:
            conversation = ""
            st_ed_indices = [999, 0]
            character = set()
            end_idx = min(random.randint(start_idx + self.min_length, start_idx + self.max_length), total_sub)

            for txt in subtitles[start_idx:end_idx + 1]:
                conversation += txt['text'] + " "
                st_ed_indices[0], st_ed_indices[1] = min(txt['start'], st_ed_indices[0]), max(txt['end'],
                                                                                              st_ed_indices[1])
                person = self.extract_character_name_from_subtitle(txt['text'])
                if person is not None:
                    character.add(person)

            temporal_region.append(
                {'vid_name': vid, 'st': st_ed_indices[0], 'ed': st_ed_indices[1], 'dialogue': conversation,
                 'length': st_ed_indices[1] - st_ed_indices[0], 'character_name': list(character)})
            start_idx = end_idx + 1

        return temporal_region

    def define_temporal_region_by_random(self, vid, subtitles, sub_meta, vid_length):
        segment = []
        for i in range(5):
            temp = random.sample(list(np.arange(0, vid_length)), 2)
            temp.sort()
            segment.append(list(np.arange(temp[0], temp[1])))

        temp = []
        for clip_segment in segment:
            meta = {}
            subtitle_idx = []
            conversation = ""
            st_ed_indices = [999, 0]
            character = set()

            for clip_idx in clip_segment:
                if str(clip_idx) in list(sub_meta['clip2sen'].keys()):
                    subtitle_idx.extend(sub_meta['clip2sen'][str(clip_idx)])

            subtitle_idx = list(set(subtitle_idx))
            if len(subtitle_idx) < 2:
                continue

            for sub_idx in subtitle_idx:
                conversation += subtitles[sub_idx]['text'] + " "
                st_ed_indices[0] = min(subtitles[sub_idx]['start'], st_ed_indices[0])
                st_ed_indices[1] = max(subtitles[sub_idx]['end'], st_ed_indices[1])
                person = self.extract_character_name_from_subtitle(subtitles[sub_idx]['text'])
                if person is not None:
                    character.add(person)

            meta['vid_name'] = vid
            meta['dialogue'] = conversation
            meta['st'] = st_ed_indices[0]
            meta['ed'] = st_ed_indices[1]
            meta['length'] = st_ed_indices[1] - st_ed_indices[0]
            meta['character_name'] = list(character)
            temp.append(meta)
        return temp

    def define_temporal_regions(self):
        temporal_regions = []
        define_temporal_region = time.time()

        if self.sampling_mode == 'sub':
            self.output_path = "{}_{}_{}.jsonl".format(self.output_path.split('.')[0], self.min_length, self.max_length)
            for vid in tqdm(self.vid_list, desc='Extract temporal regions by subtitle'.format(self.sampling_mode),
                            total=len(self.vid_list)):
                sub_meta = self.tvr_sub[vid]
                temporal_regions.extend(self.define_temporal_region_by_subtitles(vid=vid, subtitles=sub_meta['sub']))

        if self.sampling_mode == 'fixed':
            self.output_path = "{}_{}.jsonl".format(self.output_path.split('.')[0], self.fixed_size)
            for vid in tqdm(self.vid_list, desc='Extract temporal regions {}'.format(self.sampling_mode),
                            total=len(self.vid_list)):
                sub_meta = self.tvr_sub[vid]
                temporal_regions.extend(
                    self.define_temporal_region_by_fixed_subtitles(vid=vid, subtitles=sub_meta['sub'], size=self.fixed_size))

        if self.sampling_mode == 'rand':
            for vid in tqdm(self.vid_list, desc='Extract temporal regions by random'.format(self.sampling_mode),
                            total=len(self.vid_list)):
                sub_meta = self.tvr_sub[vid]
                video_feat = self.load_vid_feat(vid)
                vid_length = video_feat.shape[0]

                temporal_regions.extend(
                    self.define_temporal_region_by_random(vid=vid, subtitles=sub_meta['sub'], sub_meta=sub_meta,
                                                          vid_length=vid_length))

        if self.debug:
            print(temporal_regions)

        define_temporal_region_time = time.time() - define_temporal_region
        print(define_temporal_region_time)
        print('Number of generated temporal regions : {}, avg length : {}'.
              format(len(temporal_regions), sum([data['length'] for data in temporal_regions]) / len(temporal_regions)))

        return temporal_regions

    def generate_modal_specific_pseudo_query(self, temporal_regions):
        desc_id = 0
        skip_cnt = 0
        self.image_model.eval()
        self.text_model.eval()
        self.image_model.cuda()
        self.text_model.cuda()

        print('limit')
        temporal_regions = temporal_regions[:87175]

        logger.info("Now generating pseudo queries... expected num of pseudo query : {}".format(len(temporal_regions)))

        for segment in tqdm(temporal_regions, desc='Generating pseudo queries', total=len(temporal_regions)):
            generate_timer = time.time()
            vid = segment['vid_name']
            image_root_path = self.load_image_path(vid)

            if image_root_path is None or int(segment['st']) == int(segment['ed']):
                logger.info('Path is unavailable. Skip')
                skip_cnt += 1
                continue

            frame_path = os.listdir(image_root_path)
            if len(frame_path) < 1:
                logger.info('Num of frame is 0. Skip')
                skip_cnt += 1
                continue

            frame_path.sort()
            frame_path = frame_path[int(segment['st']) * 3: int(segment['ed']) * 3]

            pseudo_vid_query = self.generate_vid_pseudo_query(segment['character_name'], image_root_path, frame_path)
            pseudo_sub_query = self.dialogue_summarization(segment['dialogue'])

            dict = {}
            dict['vid_name'] = vid
            dict['desc_id'] = desc_id
            dict['character'] = segment['character_name']
            dict['dialogue'] = segment['dialogue']
            dict['pseudo_sub_query'] = pseudo_sub_query
            dict['pseudo_vid_query'] = pseudo_vid_query
            dict['ts'] = [segment['st'], segment['ed']]
            dict['length'] = dict['ts'][1] - dict['ts'][0]

            self.pseudo_data.append(dict)
            desc_id += 1
            self.query_generation_time.update(time.time() - generate_timer)

            if len(self.pseudo_data) > 1000:
                logger.info("Flushing list memory, len : {}".format(len(self.pseudo_data)))
                print("[Sample of pseudo query] {}".format(self.pseudo_data[0]))
                if self.flush:
                    read_and_save_jsonl(self.pseudo_data, self.output_path)
                else:
                    save_jsonl(self.pseudo_data, self.output_path)
                    self.flush = True
                self.pseudo_data = self.flush_output()

        if self.flush:
            read_and_save_jsonl(self.pseudo_data, self.output_path)
        else:
            save_jsonl(self.pseudo_data, self.output_path)

        print("{} saved. skip count : {},  {} pseudo-query generated".format(self.output_path, skip_cnt, len(load_jsonl(self.output_path))))
        print("Query generation time: max {query_generation_time.max} min {query_generation_time.min} avg {query_generation_time.avg}"
              .format(query_generation_time=self.query_generation_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--subtitle_summary_path", type=str, default=None, help="path to summary data")
    parser.add_argument("--subtitle_data_path", type=str, default=None, help="path to subtitle data")
    parser.add_argument("--frame_data_root_path", type=str, default=None, help="path to subtitle data")
    parser.add_argument("--video_feat_file", type=str, default=None, help="")
    parser.add_argument("--sub_info_json_file", type=str, default=None, help="")
    parser.add_argument("--tvr_vid_path", type=str, default=None, help="")
    parser.add_argument("--output_path", type=str, default=None, help="")
    parser.add_argument("--fixed_size", type=int, default="4", help="num of object")
    parser.add_argument("--min_length", type=int, default="2", help="minimum number of subtitle")
    parser.add_argument("--max_length", type=int, default="5", help="maximum number of subtitle")
    parser.add_argument("--image_size", type=int, default="384", help="size of images")
    parser.add_argument("--sampling_mode", type=str, default="sub", help="How to sample moments in videos",
                        choices=['sub', 'rand', 'fixed'])
    parser.add_argument("--device", type=int, default=0, help="")
    parser.add_argument("--debug", '-d', action="store_true",
                        help="debug (fast) mode, break all loops, do not load all data into memory.")

    config = parser.parse_args()
    Model = Generate_Pseudo_Query(config)
    temporal_regions = Model.define_temporal_regions()
    Model.generate_modal_specific_pseudo_query(temporal_regions)

