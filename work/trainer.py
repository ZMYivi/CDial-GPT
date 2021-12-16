import torch
import os
import random
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import math
import torch.tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import tqdm
import json
from transformers import (OpenAIGPTLMHeadModel, OpenAIGPTConfig, GPT2LMHeadModel, GPT2Config,
                          WEIGHTS_NAME, CONFIG_NAME, AdamW, BertTokenizer)

from od.inputters.inputter import build_dataloaders, build_dist_loaders
from evaluator import Evaluator


class Trainer(object):
    def __init__(self, logger, log_dir, conf, conf_path, distributed, device=torch.device('cuda')):
        self.logger = logger
        self.log_dir = log_dir
        self.conf = conf
        self.conf_path = conf_path
        self.device = device

        model_class = OpenAIGPTLMHeadModel if not conf.gpt2 else GPT2LMHeadModel
        config_class = OpenAIGPTConfig if not conf.gpt2 else GPT2Config
        tokenizer_class = BertTokenizer
        if conf.pretrained:
            tokenizer = tokenizer_class.from_pretrained(conf.model_checkpoint, do_lower_case=True,
                                                        never_split=["[speaker1]", "[speaker2]"])
            self.model = model_class.from_pretrained(conf.model_checkpoint)
        else:
            tokenizer = tokenizer_class(os.path.join(conf.model_checkpoint, "vocab.txt"), do_lower_case=True,
                                        never_split=["[speaker1]", "[speaker2]"])
            config = config_class.from_json_file(os.path.join(conf.model_checkpoint, CONFIG_NAME))
            model = model_class(config)
        self.model.to(device)

        self.rank = torch.distributed.get_rank() if distributed else -1
        self.optimizer = AdamW([{'params': self.model.parameters(), 'initial_lr': self.conf.lr}], lr=self.conf.lr, correct_bias=True)

        loader_class = build_dist_loaders if not conf.data_path else build_dataloaders
        self.train_loader, self.val_loader, self.train_sampler, self.valid_sampler = loader_class(conf, tokenizer, logger)

        self.evaluator = Evaluator(args=self.conf, valid_sampler=self.valid_sampler, data=self.train_loader, model=self.model, logdir=self.log_dir)

    def _eval_train(self, epoch, lr_scheduler):
        if not (self.rank == -1 or self.rank == 0):
            iter = self.train_loader
        else:
            iter = tqdm.tqdm(self.train_loader, total=len(self.train_loader), dynamic_ncols=True)

        loss_ans = None
        lr_vaue = None
        for i, data in iter:
            if i % self.conf.valid_steps == 0:
                self.evaluator.run()

            lr_scheduler.last_epoch = epoch
            lr_list = lr_scheduler.get_lr()
            value = lr_list[0]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = value

            input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(self.device) for input_tensor in data)
            self.model.train()
            (lm_loss), *_ = self.model(input_ids, labels=lm_labels, token_type_ids=token_type_ids)
            loss = lm_loss / self.conf.gradient_accumulation_steps
            if self.conf.fp16:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.conf.max_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.max_norm)
            if i % self.conf.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # todo checkpoint
            loss_ans = loss.item()
            lr_vaue = self.optimizer.param_groups[0]['lr']

        return loss_ans, lr_vaue

    def train(self, start_epoch, epochs, after_epoch_funcs, after_step_funcs):
        self.evaluator.run(self.val_loader)
        epoch = start_epoch + 1
        while epoch < epochs:
            # todo 考虑下优化器是否需要再封装
            self.logger.info('Training on process {}, epoch {}, step {}'.format(
                self.rank, epoch, self.optimizer.step()))
            self.train_sampler.set_epoch(epoch)

            model_size = self.conf.n_emd
            noam_lambda = lambda step: (
                    model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * self.conf.warmup_steps ** (-1.5)))
            noam_scheduler = LambdaLR(self.optimizer, lr_lambda=noam_lambda, last_epoch=self.conf.from_step)

            loss, lr_value = self._eval_train(epoch, noam_scheduler)
            self.evaluator.run(self.val_loader)
            # todo checkpoint
            epoch += 1

