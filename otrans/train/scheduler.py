# File   : scheduler.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import torch
import math
import numpy as np


BuildOptimizer = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}


class BaseScheduler(object):
    def __init__(self, optimizer, stepwise=False):

        # Attach optimizer
        self.optimizer = optimizer
        self.global_step = 0
        self.global_epoch = 0
        self.stepwise = stepwise
        self.lr = 0

        self.initial_lr()

    def get_epoch_lr(self, epoch=None):
        # Compute learning rate epoch by epoch
        raise NotImplementedError

    def get_step_lr(self, step=None):
        # Compute learning rate step by step
        raise NotImplementedError

    def set_lr(self, lr=None):
        new_lr = self.lr if lr is None else lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def initial_lr(self):
        if self.stepwise:
            self.lr = self.get_step_lr(step=self.global_step)
            self.lr = self.step()
        else:
            self.lr = self.get_epoch_lr(epoch=self.global_epoch)
            self.set_lr()

    def step(self):
        self.global_step += 1
        if self.stepwise:
            self.lr = self.get_step_lr(step=self.global_step)
            self.set_lr()

    def epoch(self):
        self.global_epoch += 1
        if not self.stepwise:
            self.lr = self.get_epoch_lr(epoch=self.global_epoch)
            self.set_lr()


class ConstantValueScheduler(BaseScheduler):
    def __init__(self, optimizer, lr):

        self.fixed_lr = lr
        super(ConstantValueScheduler, self).__init__(optimizer, stepwise=False)

    def get_epoch_lr(self, epoch):
        return self.fixed_lr


def get_linear_lr(i, start, end, start_lr, end_lr):
    assert end > start
    lr = start_lr + (i - start) * (end_lr - start_lr) / (end - start)
    return float(np.where(i < start, start_lr, np.where(i > end, end_lr, lr)))



class LinearStepScheduler(BaseScheduler):
    def __init__(self, optimizer, final_step, start_lr, final_lr):

        self.final_step = final_step
        self.start_lr = start_lr
        self.final_lr = final_lr
        super(LinearStepScheduler, self).__init__(optimizer, stepwise=True)

    def get_step_lr(self, step):
        return get_linear_lr(step, 0, self.final_step, self.start_lr, self.final_lr)


class LinearEpochScheduler(BaseScheduler):
    def __init__(self, optimizer, final_epoch, start_lr, final_lr):
        
        self.final_epoch = final_epoch
        self.start_lr = start_lr
        self.final_lr = final_lr
        super(LinearEpochScheduler, self).__init__(optimizer, stepwise=False)

    def get_epoch_lr(self, epoch):
        return get_linear_lr(epoch, 0, self.final_epoch, self.start_lr, self.final_lr)


class ExponentialScheduler(BaseScheduler):
    def __init__(self, optimizer, final_step, start_lr, final_lr):

        self.final_step = final_step
        self.start_lr = start_lr
        self.final_lr = final_lr
        super(ExponentialScheduler, self).__init__(optimizer, stepwise=True)

    def get_step_lr(self, step):
        linear_lr = get_linear_lr(step, 0, self.final_step, self.start_lr, self.final_lr)
        return math.exp(linear_lr)


class  StepwiseExponentialScheduler(BaseScheduler):
    def __init__(self, optimizer, init_lr, decay_factor, min_lr=1e-6):
        
        self.init_lr = init_lr
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.lr = self.init_lr
        super( StepwiseExponentialScheduler, self).__init__(optimizer, stepwise=True)

    def get_step_lr(self, step):
        return max(math.pow(self.lr, self.decay_factor), self.min_lr)


class TransformerScheduler(BaseScheduler):
    def __init__(self, optimizer, model_size, warmup_steps, factor=1.0):
        
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        super(TransformerScheduler, self).__init__(optimizer, stepwise=True)  

    def get_step_lr(self, step):
        return self.factor * self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))


class LinearWarmUpAndExpDecayScheduler(BaseScheduler):
    def __init__(self, optimizer, warmup_steps, decay_start, peak_lr, final_lr, decay_factor):

        self.warmup_steps = warmup_steps
        self.decay_start = decay_start
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.decay_factor = decay_factor

        assert self.decay_start > self.warmup_steps and self.decay_factor < 1.0
        super(LinearWarmUpAndExpDecayScheduler, self).__init__(optimizer, stepwise=True)
    
    def get_step_lr(self, step):

        return float(np.where(
            step < self.warmup_steps,
            get_linear_lr(step, 0, self.warmup_steps, 0.0, self.peak_lr),
            np.where(
                step > self.decay_start,
                max(self.get_expdecay_lr(step), self.final_lr),
                self.peak_lr
            )
        ))
    
    def get_expdecay_lr(self, step):
        return math.pow(self.lr, self.decay_factor)


BuildScheduler = {
    'constant': ConstantValueScheduler,
    'step-linear': LinearStepScheduler,
    'epoch-linear': LinearEpochScheduler,
    'exp': ExponentialScheduler,
    'step-exp': StepwiseExponentialScheduler,
    'transformer': TransformerScheduler,
    'linear-warmup-exp-decay': LinearWarmUpAndExpDecayScheduler
}
