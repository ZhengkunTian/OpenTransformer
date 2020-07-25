"""
@Author: Zhengkun Tian
@Email: zhengkun.tian@outlook.com
@Date: 2020-04-23 15:14:28
@LastEditTime: 2020-04-23 15:16:49
@FilePath: \OpenTransformer\otrans\train.py
"""

import os
import torch
import math
import time
import torch.distributed as dist
from otrans.utils import MeanLoss, init_logger, Visulizer, AverageMeter, Summary, map_to_cuda


class Trainer(object):
    def __init__(self, params, model, optimizer, scheduler=None, is_visual=True, expdir='./',
                 ngpu=1, parallel_mode='dp', local_rank=0, mixed_precision=False, opt_level='O1'):

        self.params = params
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.expdir = expdir
        self.is_visual = is_visual

        self.ngpu = ngpu
        self.parallel_mode = parallel_mode
        self.local_rank = local_rank

        self.shuffle = params['train']['shuffle']
        self.accum_steps = params['train']['accum_steps']
        self.grad_noise = params['train']['grad_noise']
        self.grad_clip = params['train']['clip_grad']
        self.global_step = 0
        self.log_interval = 10
        self.mean_loss = MeanLoss()

        self.mixed_precision = mixed_precision
        self.opt_level = opt_level

        self.logger = init_logger(log_file=os.path.join(expdir, 'train.log'))
        if self.is_visual and local_rank == 0:
            self.visulizer = Visulizer(log_dir=os.path.join(expdir, 'visual'))

        if self.params['train']['load_model']:
            self.load_model(self.params['train']['load_model'])
            self.logger.info('Load the checkpoint from %s' % self.params['train']['load_model'])

        if self.mixed_precision:
            import apex.amp as amp
            self.model, self.optimizer.optimizer = amp.initialize(self.model, self.optimizer.optimizer, opt_level=self.opt_level)

        if self.ngpu > 1:
#             if self.parallel_mode == 'hvd':
#                 import horovod.torch as hvd
#                 hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
#                 self.logger.info('[Horovod] Use %d gpus for training!' % self.ngpu)

            if self.parallel_mode == 'ddp':
                import torch.distributed as dist
                dist.init_process_group(backend="nccl", init_method='env://',
                                        rank=local_rank, world_size=self.ngpu)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank)
                self.logger.info('[DDP] Use %d gpus for training!' % self.ngpu)

            elif self.parallel_mode == 'dp':
                self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(self.ngpu)])
                self.logger.info('[DP] Use %d gpus for training!' % self.ngpu)

            else:
                self.logger.warning('Please chose one of dp, ddp and hvd for parallel computing!')
        elif self.ngpu == 1:
            self.logger.info('Use only 1 gpu for training!')
        else:
            self.logger.info('Train the model in CPU!')

    def train(self, train_loader, dev_loader=None):

        epochs = self.params['train']['epochs']
        TrainLossNote = Summary()
        DevLossNote = Summary()
        for epoch in range(epochs):

            self.optimizer.epoch()
            if self.parallel_mode == 'ddp':
                train_loader.set_epoch(epoch)
                self.logger.info('Set the epoch of train sampler as %d' % epoch)

            train_loss = self.train_one_epoch(epoch, train_loader.loader)
            TrainLossNote.update(epoch, train_loss)

            if self.local_rank == 0:
                self.logger.info('-*Train-Epoch-%d/%d*-, AvgLoss:%.5f' % (epoch, epochs, train_loss))

                self.save_model(epoch)
                self.logger.info('Save the model!')

            if self.is_visual and self.local_rank == 0:
                self.visulizer.add_scalar('train_epoch_loss', train_loss, epoch)

            if dev_loader is not None:
                dev_loss = self.eval(dev_loader.loader)
                DevLossNote.update(epoch, dev_loss)
                if self.local_rank == 0:
                    self.logger.info('-*Eval-Epoch-%d/%d*-, AvgLoss:%.5f' % (epoch, epochs, dev_loss))

                if dev_loss < DevLossNote.best()[1] and self.local_rank == 0:
                    self.save_model('model.best.pt')
                    self.logger.info('Update the best checkpoint!')

        if self.local_rank == 0:
            self.logger.info('Training Summary:')
            BEST_T_EPOCH, BEST_T_LOSS = TrainLossNote.best()
            self.logger.info('At the %d-st epoch of training, the model performs best (Loss:%.5f)!' % (BEST_T_EPOCH, BEST_T_LOSS))
            if dev_loader is not None:
                BEST_E_EPOCH, BEST_E_LOSS = DevLossNote.best()
                self.logger.info('At the %d-st epoch of validation, the model performs best (Loss:%.5f)!' % (BEST_E_EPOCH, BEST_E_LOSS))

            if self.is_visual:
                self.visulizer.close()

    def train_one_epoch(self, epoch, train_loader):

        self.model.train()
        batch_steps = len(train_loader)

        step_loss = AverageMeter()
        span = 0
        for step, (_, batch) in enumerate(train_loader):

            if self.ngpu > 0:
                batch = map_to_cuda(batch)

            start = time.process_time()
            loss = self.model(**batch)
            loss = torch.mean(loss) / self.accum_steps

            if self.mixed_precision:
                import apex.amp as amp
                with amp.scale_loss(loss, self.optimizer.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            end = time.process_time()
            span += (end - start)
            if self.grad_noise:
                raise NotImplementedError

            if self.get_rank() == 0:
                step_loss.update(loss.item() * self.accum_steps, batch['inputs'].size(0))

            if step % self.accum_steps == 0:
                if self.local_rank == 0:
                    self.mean_loss.update(step_loss.avg)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if math.isnan(grad_norm):
                    self.logger.warning('Grad norm is NAN. DO NOT UPDATE MODEL!')
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

                if self.is_visual and self.local_rank == 0:
                    self.visulizer.add_scalar('train_loss', loss.item(), self.global_step)
                    self.visulizer.add_scalar('lr', self.optimizer.lr, self.global_step)

                if self.global_step % self.log_interval == 0 and self.local_rank == 0:
                    
                    # process = step * self.world_size / batch_steps * 100
                    process = step / batch_steps * 100
                    self.logger.info('-Training-Epoch-%d(%.5f%%), Global Step:%d, lr:%.8f, Loss:%.5f, AvgLoss: %.5f, '
                                     'Run Time:%.3f' % (epoch, process, self.global_step, self.optimizer.lr,
                                                        step_loss.avg, self.mean_loss.mean(), span))
                    span = 0

                self.global_step += 1
                step_loss.reset()

        return self.mean_loss.mean()

    def eval(self, dev_loader):
        self.model.eval()
        eval_loss = 0
        for step, (_, batch) in enumerate(dev_loader):

            if self.ngpu > 0:
                batch = map_to_cuda(batch)

            loss = self.model(**batch)

            eval_loss += loss.item()

        return eval_loss / (step+1)

    def save_model(self, epoch=None, save_name=None):
        if save_name is None:
            save_name = 'model.epoch.%d.pt' % epoch

        if self.mixed_precision:
            import apex.amp as amp
            amp_state_dict = amp.state_dict()
        else:
            amp_state_dict = None

        checkpoint = {
            'epoch': epoch,
            'params': self.params,
            'model': self.model.module.state_dict() if self.ngpu > 1 else self.model.state_dict(),
             #'optimizer': self.optimizer.state_dict(),
            'amp': amp_state_dict
        }

        torch.save(checkpoint, os.path.join(self.expdir, save_name))

    def load_model(self, checkpoint):

        state_dict = torch.load(checkpoint)
        self.model.load_state_dict(state_dict['model'])
        if self.mixed_precision:
            import apex.amp as amp
            amp.load_state_dict(state_dict['amp'])

    def get_rank(self):

        if self.parallel_mode == 'ddp':
            return dist.get_rank()
#         elif self.parallel_mode == 'hvd':
#             return hvd.rank()
        else:
            return 0

    @property
    def world_size(self):
        if self.ngpu > 1:
            return self.ngpu
        else:
            return 1
        

