import os
import time
import torch
import argparse

from utils.basic_utils import mkdirp, load_json, save_json, make_zipfile
from baselines.clip_alignment_with_language.local_utils.proposal import ProposalConfigs


class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        self.parser.add_argument("--dset_name", type=str, choices=["tvr"])
        self.parser.add_argument("--eval_split_name", type=str, default="val",
                                 help="should match keys in video_duration_idx_path, must set for VCMR")
        self.parser.add_argument("--debug", action="store_true",
                                 help="debug (fast) mode, break all loops, do not load all data into memory.")
        self.parser.add_argument("--data_ratio", type=float, default=1.0,
                                 help="how many training and eval data to use. 1.0: use all, 0.1: use 10%."
                                      "Use small portion for debug purposes. Note this is different from --debug, "
                                      "which works by breaking the loops, typically they are not used together.")
        self.parser.add_argument("--results_root", type=str, default="results")
        self.parser.add_argument("--exp_id", type=str, default="res", help="id of the current run")
        self.parser.add_argument("--seed", type=int, default=2018, help="random seed")
        self.parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        self.parser.add_argument("--device_ids", type=int, nargs="+", default=[0], help="GPU ids to run the job")
        self.parser.add_argument("--num_workers", type=int, default=8,
                                 help="num subprocesses used to load the data, 0: use main process")
        self.parser.add_argument("--no_core_driver", action="store_true",
                                 help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")
        self.parser.add_argument("--no_pin_memory", action="store_true",
                                 help="Don't use pin_memory=True for dataloader. "
                                      "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")

        # training config
        self.parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
        self.parser.add_argument("--wd", type=float, default=0, help="weight decay")
        self.parser.add_argument("--momentum", type=float, default=0.95, help="momentum for SGD")
        self.parser.add_argument("--n_epoch", type=int, default=108, help="number of epochs to run")
        self.parser.add_argument("--max_es_cnt", type=int, default=108, help="number of epochs to early stop")
        self.parser.add_argument("--bsz", type=int, default=128, help="mini-batch size")
        self.parser.add_argument("--eval_query_bsz", type=int, default=1000,
                                 help="mini-batch size at inference, for query")
        self.parser.add_argument("--eval_proposal_bsz", type=int, default=200,
                                 help="mini-batch size at inference, for proposals")
        self.parser.add_argument("--eval_untrained", action="store_true", help="Evaluate on un-trained model")
        self.parser.add_argument("--grad_clip", type=float, default=-1, help="perform gradient clip, -1: disable")
        self.parser.add_argument("--margin", type=float, default=0.1, help="margin for hinge loss")
        self.parser.add_argument("--inter_loss_weight", type=float, default=0.4, help="margin for ranking loss")
        self.parser.add_argument("--loss_type", type=str, default="hinge", choices=["hinge", "lse"],
                                 help="att loss type, can be hinge loss or its smooth approximation LogSumExp")

        # Model and Data config
        self.parser.add_argument("--max_sub_l", type=int, default=50,
                                 help="max length of all sub sentence 97.71 under 50 for 3 sentences")
        self.parser.add_argument("--max_desc_l", type=int, default=30, help="max length of descriptions")
        self.parser.add_argument("--pos_iou_thd", type=float, default=0.7, help="moments with IoU >= as positive")
        self.parser.add_argument("--neg_iou_thd", type=float, default=0.35, help="moments with IoU < as negative")

        self.parser.add_argument("--train_path", type=str, default=None)
        self.parser.add_argument("--eval_path", type=str, default=None,
                                 help="Evaluating during training, for Dev set. If None, will only do training, "
                                      "anet_cap and charades_sta has no dev set, so None")
        self.parser.add_argument("--external_train_vr_res_path", type=str, default=None,
                                 help="if set, use external video retrieval results to guide "
                                      "inter-nvideo negative sampling. ")
        self.parser.add_argument("--init_ckpt_path", type=str, default=None,
                                 help="init model parameters from checkpoint. Use absolute path")
        self.parser.add_argument("--external_inference_vr_res_path", type=str, default=None,
                                 help="if set, use external video retrieval results to guide evaluation. ")
        self.parser.add_argument("--use_glove", action="store_true", help="Use GloVe instead of BERT features")
        self.parser.add_argument("--word2idx_path", type=str,
                                 help="a dict, {word: word_idx, ...}, "
                                      "special tokens are {<pad>: 0, <unk>: 1, <eos>: 2}")
        self.parser.add_argument("--vocab_size", type=int, default=-1,
                                 help="Set automatically to len(word2idx)")
        self.parser.add_argument("--glove_path", type=str,
                                 help="path to file containing the GloVe embeddings for words in word2idx")
        self.parser.add_argument("--desc_bert_path", type=str, default=None)
        self.parser.add_argument("--sub_bert_path", type=str, default=None)
        self.parser.add_argument("--sub_feat_size", type=int, default=768, help="feature dim for sub feature")
        self.parser.add_argument("--desc_feat_size", type=int, default=768)
        self.parser.add_argument("--ctx_mode", type=str,
                                 choices=["video", "sub", "tef", "video_sub", "video_tef", "sub_tef", "video_sub_tef"],
                                 help="which context to use. a combination of [video, sub, tef]")
        self.parser.add_argument("--video_duration_idx_path", type=str, default=None)
        self.parser.add_argument("--vid_feat_path", type=str, default="")
        self.parser.add_argument("--no_norm_vfeat", action="store_true",
                                 help="Do not do normalization on video feat, use it when using i3d_resnet concat feat")
        self.parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not do normalization on text feat")
        self.parser.add_argument("--clip_length", type=float, default=None,
                                 help="each video will be uniformly segmented into small clips, "
                                      "will automatically loaded from ProposalConfigs if None")
        self.parser.add_argument("--vid_feat_size", type=int, help="feature dim for video feature")

        self.parser.add_argument("--model_type", default="cal", choices=["cal", "mcn"])
        self.parser.add_argument("--embedding_size", type=int, default=768)
        self.parser.add_argument("--lstm_hidden_size", type=int, default=256)
        self.parser.add_argument("--visual_hidden_size", type=int, default=256)
        self.parser.add_argument("--output_size", type=int, default=256)

        # post processing
        self.parser.add_argument("--nms_thd", type=float, default=-1,
                                 help="additionally use non-maximum suppression "
                                      "(or non-minimum suppression for distance)"
                                      "to post-processing the predictions. "
                                      "-1: do not use nms. 0.6 for charades_sta, 0.5 for anet_cap,")
        self.parser.add_argument("--max_after_nms", type=int, default=100, help="Stores at max_after_nms for eval")
        self.parser.add_argument("--max_before_nms", type=int, default=300, help="Max before nms")
        self.parser.add_argument("--use_intermediate", action="store_true",
                                 help="Whether to use/save intermediate results to results directory."
                                      "Might want use this if we are going to ")

    def save_args(self, opt):
        args = vars(opt)
        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)  # not yaml file indeed
            save_json(args, option_file_path, save_pretty=True)

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        if opt.debug:
            opt.results_root = os.path.sep.join(opt.results_root.split(os.path.sep)[:-1] + ["debug_results", ])
            opt.no_core_driver = True
            opt.num_workers = 0

        if isinstance(self, TestOptions):
            # modify model_dir to absolute path
            opt.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", opt.model_dir)
            saved_options = load_json(os.path.join(opt.model_dir, self.saved_option_filename))
            for arg in saved_options:  # use saved options to overwrite all BaseOptions args.
                if arg not in ["results_root", "num_workers", "nms_thd", "debug", "eval_split_name", "eval_path",
                               "use_intermediate", "external_inference_vr_res_path"]:
                    setattr(opt, arg, saved_options[arg])
            # opt.no_core_driver = True
        else:
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for at a training option!")

            if opt.clip_length is None:
                opt.clip_length = ProposalConfigs[opt.dset_name]["clip_length"]
            opt.results_dir = os.path.join(opt.results_root,
                                           "-".join([opt.dset_name, opt.model_type, opt.ctx_mode, opt.exp_id,
                                                     time.strftime("%Y_%m_%d_%H_%M_%S")]))
            mkdirp(opt.results_dir)
            # save a copy of current code
            code_dir = os.path.dirname(os.path.realpath(__file__))
            code_zip_filename = os.path.join(opt.results_dir, "code.zip")
            make_zipfile(code_dir, code_zip_filename,
                         enclosing_dir="code",
                         exclude_dirs_substring="results",
                         exclude_dirs=["results", "debug_results", "__pycache__"],
                         exclude_extensions=[".pyc", ".ipynb", ".swap"])

        self.save_args(opt)

        if "sub" in opt.ctx_mode:
            assert opt.dset_name == "tvr", "sub is only supported for tvr dataset"

        if "video" in opt.ctx_mode and opt.vid_feat_size > 3000:  # 3072, the normalized concatenation of resnet+i3d
            assert opt.no_norm_vfeat

        opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
        opt.device = torch.device("cuda:%d" % opt.device_ids[0] if opt.device >= 0 else "cpu")
        opt.h5driver = None if opt.no_core_driver else "core"
        # num_workers > 1 will only work with "core" mode, i.e., memory-mapped hdf5
        opt.pin_memory = not opt.no_pin_memory
        opt.num_workers = 1 if opt.no_core_driver else opt.num_workers

        # Display settings
        print("------------ Options -------------\n{}\n-------------------"
              .format({str(k): str(v) for k, v in sorted(vars(opt).items())}))
        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""
    def initialize(self):
        BaseOptions.initialize(self)
        # also need to specify --eval_split_name
        self.parser.add_argument("--eval_id", type=str, help="evaluation id")
        self.parser.add_argument("--model_dir", type=str,
                                 help="dir contains the model file, will be converted to absolute path afterwards")
        self.parser.add_argument("--tasks", type=str, nargs="+", choices=["VCMR", "SVMR", "VR"], default="SVMR",
                                 help="Which tasks to run."
                                      "VCMR: Video Corpus Moment Retrieval;"
                                      "SVMR: Single Video Moment Retrieval;"
                                      "VR: regular Video Retrieval.")
