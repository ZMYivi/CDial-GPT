import os
import tqdm
import logging
import time
import tempfile

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertConfig, BertForMultipleChoice


class Evaluator(object):
    def __init__(self, device, args, valid_sampler, data, model, logdir):
        self.should_terminate = False
        self.args = args
        self.valid_sampler = valid_sampler
        self.dataloader = data
        self.batch = None
        self.iteration = 0
        self.checkpoint_iteration = 0
        self.model = model
        self.output = None
        self.should_terminate_single_epoch = False
        self.saved = []
        self.logdir = logdir

    def checkpoint(self, engine, to_save):
        if len(to_save) == 0:
            raise RuntimeError("No objects to checkpoint found.")

        self.checkpoint_iteration += 1
        priority = self.checkpoint_iteration
        if self.checkpoint_iteration != 0:
            return

        if (len(self.saved) < 3) or (self.saved[0][0] < priority):
            saved_objs = []
            suffix = ""
            for name, obj in to_save.items():
                fname = '{}_{}_{}{}.pth'.format('checkpoint', name, self.checkpoint_iteration, suffix)
                path = os.path.join(self._dirname, fname)

                tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.logdir)
                try:
                    if not hasattr(obj, "state_dict") or not callable(obj.state_dict):
                        raise ValueError("Object should have `state_dict` method.")
                    torch.save(obj.state_dict(), path)
                except BaseException:
                    tmp.close()
                    os.remove(tmp.name)
                    raise
                else:
                    tmp.close()
                    os.rename(tmp.name, path)
                self._save(obj=obj, path=path)
                saved_objs.append(path)

            self._saved.append((priority, saved_objs))
            self._saved.sort(key=lambda item: item[0])

        if len(self._saved) > self._n_saved:
            _, paths = self._saved.pop(0)
            for p in paths:
                os.remove(p)

    def to_hours_mins_secs(self, time_taken):
        """Convert seconds to hours, mins, and seconds."""
        mins, secs = divmod(time_taken, 60)
        hours, mins = divmod(mins, 60)
        return hours, mins, secs

    def core_work(self, batch):
        self.model.eval()
        with torch.no_grad():
            input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(self.args.device) for input_tensor in batch)
            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            lm_logits, *_ = self.model(input_ids, token_type_ids=token_type_ids)
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return lm_logits_flat_shifted, lm_labels_flat_shifted

    def run_once_on_dataset(self):
        start_time = time.time()
        try:
            for batch in self.dataloader:
                self.batch = batch
                self.iteration += 1
                self.output = self.core_work(batch)

        except BaseException as e:
            # self._logger.error("Current run is terminating due to exception: %s.", str(e))
            # self._handle_exception(e)
            print('error')

    def run(self, data, max_epochs=1):
        self.dataloader = data
        epoch = 0
        while epoch < max_epochs:
            epoch += 1
            if self.args.distributed:
                self.valid_sampler.set_epoch(epoch)
            self.run_once_on_dataset()
            if self.args.distributed:
                self.valid_sampler.set_epoch(epoch)
            self.run_once_on_dataset()
            # self._logger.info("Epoch[%s] Complete. Time taken: %02d:%02d:%02d", self.state.epoch, hours, mins, secs)

            # todo checkpoint
            # checkpoint_handler = ModelCheckpoint(tb_logger.writer.logdir, 'checkpoint', save_interval=1, n_saved=3)