import torch
import os
import torch.distributed
import torch.tensor
import math
from pprint import pformat
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from transformers import (OpenAIGPTLMHeadModel, OpenAIGPTConfig, GPT2LMHeadModel, GPT2Config,
                          WEIGHTS_NAME, CONFIG_NAME, AdamW, BertTokenizer)

from od.inputters.inputter import build_dataloaders, build_dist_loaders
from tqdm.autonotebook import tqdm


class Trainer(object):
    def __init__(self, args, logger):
        self.logger = logger

        # Initialize distributed training if needed
        args.distributed = (args.local_rank != -1)
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')

        self.logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
        model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
        config_class = OpenAIGPTConfig if not args.gpt2 else GPT2Config
        tokenizer_class = BertTokenizer
        if args.pretrained:
            tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, do_lower_case=True,
                                                        never_split=["[speaker1]", "[speaker2]"])
            self.model = model_class.from_pretrained(args.model_checkpoint)
        else:
            tokenizer = tokenizer_class(os.path.join(args.model_checkpoint, "vocab.txt"), do_lower_case=True,
                                        never_split=["[speaker1]", "[speaker2]"])
            config = config_class.from_json_file(os.path.join(args.model_checkpoint, CONFIG_NAME))
            self.model = model_class(config)
        self.model.to(args.device)

        self.optimizer = AdamW([{'params': self.model.parameters(), 'initial_lr': args.lr}], lr=args.lr, correct_bias=True)

        logger.info("Prepare datasets")
        loader_class = build_dist_loaders if not args.data_path else build_dataloaders
        self.train_loader, self.val_loader, self.train_sampler, self.valid_sampler = loader_class(args, tokenizer, logger)

        # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
        if args.fp16:
            from apex import amp  # Apex is only required if we use fp16 training
            self.model, optimizer = amp.initialize(self.model, self.optimizer, opt_level=args.fp16)
        if args.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[args.local_rank], output_device=args.local_rank)

        self.conf = args

        model_size = self.conf.n_emd
        noam_lambda = lambda step: (
                model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * self.conf.warmup_steps ** (-1.5)))
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=noam_lambda, last_epoch=self.conf.from_step)
        self.milestones = [0, args.n_epochs * len(self.train_loader)]
        self.values = [args.lr, 0.0]
        self.event_index = 0

        self.metrics = {}
        self.pbar = None

    def process(self, iteration, batch):
        input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(self.conf.device) for input_tensor in batch)
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
        if iteration % self.conf.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.metrics['loss'] = loss.item()
        self.metrics['lr'] = self.optimizer.param_groups[0]['lr']

    def train(self):
        self.pbar = tqdm(total=len(self.train_loader), leave=True,
                         bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]',
                         mininterval=2)
        if self.conf.eval_before_start:
            self.evaluator(self.val_loader, 0)
        epoch = 0
        while epoch < self.conf.n_epochs:
            epoch += 1
            if self.conf.distributed:
                self.train_sampler.set_epoch(epoch)

            iteration = 0
            for batch in self.train_loader:
                iteration += 1

                if iteration % self.conf.valid_steps == 0:
                    self.evaluator(self.val_loader, epoch)

                if self.conf.scheduler != "linear":
                    self.lr_scheduler.last_epoch = epoch
                    lr_list = self.lr_scheduler.get_lr()
                    value = lr_list[0]
                else:
                    if self.milestones[0] > self.event_index:
                        start_index, end_index, start_value, end_value = self.event_index - 1, self.event_index, self.values[0], self.values[0]
                    elif self.milestones[1] <= self.event_index:
                        start_index, end_index, start_value, end_value = self.event_index, self.event_index + 1, self.values[1], self.values[1]
                    else:
                        start_index, end_index, start_value, end_value = self.milestones[0], self.milestones[1], self.values[0], self.values[1]

                    value = start_value + (end_value - start_value) * (self.event_index - start_index) / (
                                end_index - start_index)
                    self.event_index += 1

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = value
                self.process(iteration, batch)

                self.pbar.set_description("Epoch [{}/{}]".format(epoch, self.conf.n_epochs))
                self.pbar.set_postfix(**{'loss': self.metrics['loss'], 'lr': self.metrics['lr']})
                self.pbar.update()
                # todo 打印日志

            self.evaluator(self.val_loader, epoch)
            self.pbar.close()
            # todo 打印日志
            # todo 保存模型
        if self.conf.n_epochs < 1:
            self.evaluator(self.val_loader, self.conf.n_epochs)



    def average_distributed_scalar(self, scalar, args):
        """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
        if self.conf.local_rank == -1:
            return scalar
        scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
        torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
        return scalar_t.item()

    def evaluator(self, data, epoch):
        if self.conf.distributed:
            self.valid_sampler.set_epoch(1)
        metrics = {}
        _sum = 0
        num_examples = 0
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)

        for batch in data:

            self.model.eval()
            with torch.no_grad():
                input_ids, token_type_ids, lm_labels = tuple(
                    input_tensor.to(self.conf.device) for input_tensor in batch)
                # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
                lm_logits, *_ = self.model(input_ids, token_type_ids=token_type_ids)
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

            average_loss = loss_fn(lm_logits_flat_shifted, lm_labels_flat_shifted)
            _sum += average_loss.item() * len(lm_labels_flat_shifted)
            num_examples += len(lm_labels_flat_shifted)

        metrics['nll'] = _sum / num_examples
        metrics['average_nll'] = self.average_distributed_scalar([metrics['nll']], self.conf)
        metrics['average_ppl'] = math.exp(*metrics['average_nll'])
        self.pbar.write("Validation: %s" % pformat(metrics))
        # todo print log
        # todo checkpoint
