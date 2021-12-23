# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,5,6,7'

import logging
import random
from pprint import pformat
from argparse import ArgumentParser
import numpy as np
import torch
from work.trainer import Trainer
from od.inputters.inputter import build_dataloaders, build_dist_loaders
from transformers import (OpenAIGPTLMHeadModel, OpenAIGPTConfig, GPT2LMHeadModel, GPT2Config,
                          WEIGHTS_NAME, CONFIG_NAME, AdamW, BertTokenizer)
from torch.nn.parallel import DistributedDataParallel
from od.inputters.dataset_wb import WBDataset, WBdistDataset
from od.inputters.inputter import get_data


logger = logging.getLogger(__file__)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2019)


def train():
    parser = ArgumentParser()
    parser.add_argument('--gpt2', action='store_true', help="use gpt2")
    parser.add_argument("--model_checkpoint", type=str, default="config/cgpt/", help="Path or URL of the model")
    parser.add_argument("--from_step", type=int, default=-1, help="Init learning rate from this step")
    parser.add_argument('--pretrained', action='store_true', help="If False train from scratch")
    parser.add_argument("--data_path", type=str, default="",
                        help="Path or url of the dataset. ")
    parser.add_argument("--train_path", type=str, default="data/toy_train.txt",
                        help="Path of the train dataset for dist dataset. ")
    parser.add_argument("--valid_path", type=str, default="data/toy_valid.txt",
                        help="Path of the valid dataset for dist dataset. ")
    parser.add_argument("--dataset_cache", type=str, default="dataset_cache",
                        help="Path or url of the dataset cache")
    parser.add_argument('--log_file', '-log_file', type=str, default="", help="Output logs to a file under this path")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of subprocesses for data loading")
    parser.add_argument("--n_epochs", type=int, default=70, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--max_history", type=int, default=15, help="Number of previous exchanges to keep in history")
    parser.add_argument("--scheduler", type=str, default="noam", choices=['noam', 'linear'], help="method of optim")
    parser.add_argument("--n_emd", type=int, default=768, help="Number of n_emd in config file (for noam)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm up steps")
    parser.add_argument("--valid_steps", type=int, default=5000, help="Perfom validation every X steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process.
    # logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
    config_class = OpenAIGPTConfig if not args.gpt2 else GPT2Config
    tokenizer_class = BertTokenizer
    if args.pretrained:
        tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, do_lower_case=True,
                                                    never_split=["[speaker1]", "[speaker2]"])
        model = model_class.from_pretrained(args.model_checkpoint)
    else:
        tokenizer = tokenizer_class(os.path.join(args.model_checkpoint, "vocab.txt"), do_lower_case=True,
                                    never_split=["[speaker1]", "[speaker2]"])
        config = config_class.from_json_file(os.path.join(args.model_checkpoint, CONFIG_NAME))
        model = model_class(config)

    logger.info("Prepare datasets")
    if not args.data_path:
        train_dataset = WBdistDataset(tokenizer, data_path=args.train_path)
        valid_dataset = WBdistDataset(tokenizer, data_path=args.valid_path)
    else:
        datasets, _ = get_data(tokenizer, args.data_path, args.dataset_cache, logger)
        train_dataset = WBDataset(datasets["train"], tokenizer)
        valid_dataset = WBDataset(datasets["valid"], tokenizer)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    trainer = Trainer(args, logger, "./save", model, train_dataset, valid_dataset, args.device, args.distributed)
    trainer.train()


if __name__ == "__main__":
    train()
