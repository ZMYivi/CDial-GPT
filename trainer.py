import torch
import os
import tempfile
import torch.distributed
import torch.tensor
import math
from pprint import pformat
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from od.inputters.dataset_wb import WBdistDataset
from transformers import (WEIGHTS_NAME, AdamW, CONFIG_NAME)


class Trainer(object):
    def __init__(self, args, logger, tokenizer, model, train_dataset, valid_dataset, device,
                 distributed=False, fp16=False):
        """
        Init
        :param args: User-defined parameters
        :param logger: Configured log
        :param save_dir: Path to save the file
        :param model: Algorithm model
        :param train_dataset: train dataset
        :param valid_dataset: validation dataset
        :param device: cpu or gpu
        :param distributed: does it need to be distributed
        :param fp16: does it need to be fp16
        """
        self.logger = logger
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
        self.model.to(device)

        self.train_writer = SummaryWriter()
        self.vaild_writer = SummaryWriter()
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if distributed else None
        self.train_loader = DataLoader(train_dataset,
                                       collate_fn=train_dataset.collate,
                                       num_workers=args.num_workers,
                                       sampler=self.train_sampler,
                                       batch_size=args.train_batch_size,
                                       shuffle=(not args.distributed))
        self.val_loader = DataLoader(valid_dataset,
                                     collate_fn=valid_dataset.collate,
                                     num_workers=args.num_workers,
                                     sampler=self.valid_sampler,
                                     batch_size=args.valid_batch_size,
                                     shuffle=False)
        if isinstance(train_dataset, WBdistDataset):
            self.train_loader.pin_memory = (device == 'cuda')
        if isinstance(valid_dataset, WBdistDataset):
            self.val_loader.pin_memory = (device == 'cuda')

        self.optimizer = AdamW([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, correct_bias=True)

        self.conf = args

        model_size = self.conf.n_emd
        noam_lambda = lambda step: (
                model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * self.conf.warmup_steps ** (-1.5)))
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=noam_lambda, last_epoch=self.conf.from_step)
        self.milestones = [0, args.n_epochs * len(self.train_loader)]
        self.values = [args.lr, 0.0]
        self.event_index = 0

        torch.save(args, self.train_writer.log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(self.train_writer.log_dir, CONFIG_NAME))
        self.tokenizer.save_vocabulary(self.train_writer.log_dir)

        self.metrics = {}
        self.pbar = None

        self.train_saved = []
        self.eval_saved = []
        self.eval_saved_iter = 0

        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        self.fp16 = fp16

    def process(self, iteration, batch):
        """
        The process of training the model
        :param iteration: Number of iterations
        :param batch: Data to be trained
        """

        # used pytorch's half-precision
        with torch.cuda.amp.autocast(enabled=self.fp16):
            input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(self.conf.device) for input_tensor in batch)
            self.model.train()
            (lm_loss), *_ = self.model(input_ids, labels=lm_labels, token_type_ids=token_type_ids)
            loss = lm_loss / self.conf.gradient_accumulation_steps
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.max_norm)
        if iteration % self.conf.gradient_accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        self.metrics['loss'] = self.metrics['loss'] * 0.98 + 0.02 * loss.item() if 'loss' in self.metrics.keys() else loss.item()
        self.metrics['lr'] = self.metrics['lr'] * 0.98 + 0.02 * self.optimizer.param_groups[0]['lr'] if 'lr' in self.metrics.keys() else self.optimizer.param_groups[0]['lr']

    def train(self):
        if self.conf.eval_before_start:
            self.evaluate(self.val_loader, 0)
        epoch = 0
        # start loop
        while epoch < self.conf.n_epochs:
            epoch += 1
            self.metrics = {}
            if self.conf.distributed:
                self.train_sampler.set_epoch(epoch)
            iteration = 0
            for batch in self.train_loader:
                iteration += 1

                if iteration % self.conf.valid_steps == 0:
                    self.evaluate(self.val_loader, epoch)

                # According to the logic of ignite, the learning rate needs to be calculated in two cases
                if self.conf.scheduler != "linear":
                    self.lr_scheduler.last_epoch = epoch
                    lr_list = self.lr_scheduler.get_lr()
                    value = lr_list[0]
                    self.lr_scheduler.step()
                else:
                    if self.milestones[0] > self.event_index:
                        start_index, end_index, start_value, end_value = self.event_index - 1, self.event_index, self.values[0], self.values[0]
                    elif self.milestones[1] <= self.event_index:
                        start_index, end_index, start_value, end_value = self.event_index, self.event_index + 1, self.values[1], self.values[1]
                    else:
                        start_index, end_index, start_value, end_value = self.milestones[0], self.milestones[1], self.values[0], self.values[1]

                    value = start_value + (end_value - start_value) * (self.event_index - start_index) / (end_index - start_index)
                    self.event_index += 1

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = value
                self.process(iteration, batch)

                # update tqdm
                if self.pbar is None:
                    self.pbar = tqdm(total=len(self.train_loader), leave=True,
                                     bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]',
                                     mininterval=2)
                self.pbar.set_description("Epoch [{}/{}]".format(epoch, self.conf.n_epochs))
                self.pbar.set_postfix(loss=self.metrics['loss'], lr=self.metrics['lr'])
                self.pbar.update()
                self.train_writer.add_scalar("training/loss", self.metrics['loss'], iteration)
                params = {"lr/group_{}".format(i): float(param_group['lr'])
                          for i, param_group in enumerate(self.optimizer.param_groups)}

                for k, v in params.items():
                    self.train_writer.add_scalar(k, v, iteration)

            # evaluate
            self.evaluate(self.val_loader, epoch)
            self.pbar.close()

            # Save the model, only the latest three
            if len(self.train_saved) < 3 or self.train_saved[0][0] < epoch:
                saved_objs = []
                fname = 'checkpoint_{}_{}.pth'.format('mymodel', epoch)
                path = os.path.join(self.train_writer.log_dir, fname)
                tmp = tempfile.NamedTemporaryFile(delete=False, dir=os.path.expanduser(self.train_writer.log_dir))
                torch.save(self.model.state_dict(), path)
                tmp.close()
                os.rename(tmp.name, path)
                saved_objs.append(path)
                self.train_saved.append((epoch, saved_objs))
                self.train_saved.sort(key=lambda item: item[0])
            if len(self.train_saved) > 3:
                _, paths = self.train_saved.pop(0)
                for p in paths:
                    os.remove(p)

        if self.conf.n_epochs < 1:
            self.evaluate(self.val_loader, self.conf.n_epochs)

        if self.conf.local_rank in [-1, 0] and self.conf.n_epochs > 0:
            os.rename(self.train_saved[-1][1][-1],
                      os.path.join(self.train_writer.log_dir,
                                   WEIGHTS_NAME))
        self.train_writer.close()
        self.vaild_writer.close()

    def average_distributed_scalar(self, scalar):
        """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
        if self.conf.local_rank == -1:
            return scalar
        scalar_t = torch.tensor(scalar, dtype=torch.float, device=self.device) / torch.distributed.get_world_size()
        torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
        return scalar_t.item()

    def evaluate(self, data, epoch):
        """
        evaluate
        :param data: data to be evaluate
        :param epoch: epoch of train
        """
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
        metrics['average_nll'] = self.average_distributed_scalar(metrics['nll'])
        metrics['average_ppl'] = math.exp(metrics['average_nll'])
        # self.pbar.write("Validation: %s" % pformat(metrics))
        tqdm.write("Validation: %s" % pformat(metrics))
        for k, v in metrics.items():
            self.vaild_writer.add_scalar("validation/{}".format(k), v, epoch)

        # Save the model, only the latest three
        self.eval_saved_iter += 1
        if len(self.eval_saved) < 3 or self.eval_saved[0][0] < self.eval_saved_iter:
            saved_objs = []
            fname = 'checkpoint_mymodel_{}.pth'.format(epoch)
            path = os.path.join(self.vaild_writer.log_dir, fname)
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=os.path.expanduser(self.vaild_writer.log_dir))
            torch.save(self.model.state_dict(), path)
            tmp.close()
            os.rename(tmp.name, path)
            saved_objs.append(path)
            self.eval_saved.append((epoch, saved_objs))
            self.eval_saved.sort(key=lambda item: item[0])
        if len(self.eval_saved) > 3:
            _, paths = self.eval_saved.pop(0)
            for p in paths:
                os.remove(p)
