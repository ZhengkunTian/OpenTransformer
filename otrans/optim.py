import torch
import math
from torch.optim.lr_scheduler import LambdaLR


class Optimizer(object):
    def __init__(self, model, params, update_lr_stepwise=False, parallel_mode='dp'):

        self.params = params
        self.model = model
        self.update_lr_stepwise = update_lr_stepwise
        self.parralle_mode = parallel_mode

        self.lr = self.params['lr']
        self.global_step = 1
        self.global_epoch = 0

        if params['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr,
                                              betas=(0.9, 0.98), eps=1e-9)
        elif params['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr, momentum=0.9)
        elif params['optimizer'] == 'adadelate':
            self.optimizer = torch.optim.Adadelta(
                filter(lambda p: p.requires_grad, model.parameters()))
        else:
            raise NotImplementedError

        if self.parralle_mode == 'hvd':
            import horovod.torch as hvd
            self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=model.named_parameters())

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def set_lr(self, lr=None):
        new_lr = self.lr if lr is None else lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def step(self):
        self.optimizer.step()
        if self.update_lr_stepwise:
            self.lr = self.get_lr()
            self.set_lr()
        self.global_step += 1

    def epoch(self):
        if not self.update_lr_stepwise:
            self.lr = self.get_lr()
            self.set_lr()
        self.global_epoch += 1


class TransformerOptimizer(Optimizer):

    def __init__(self, model, params, model_size, parallel_mode='dp'):
        super(TransformerOptimizer, self).__init__(model, params, True, parallel_mode)

        self.model_size = model_size
        self.factor = params['lr']
        self.warmup_steps = params['warmup_steps']
        self.lr = self.get_lr()
        self.set_lr()

    def get_lr(self):
        return self.factor * self.model_size ** (-0.5) * min(self.global_step ** (-0.5), self.global_step * self.warmup_steps ** (-1.5))


class FixedLROptimizer(Optimizer):
    def __init__(self, model, params, parallel_mode='dp'):
        super(FixedLROptimizer, self).__init__(model, params, update_lr_stepwise=False, parallel_mode=parallel_mode)

    def get_lr(self):
        return self.lr


# class AdaptiveOptimizer(Optimizer):
#     def __init__(self, model, params, parallel_mode='dp'):
#         super(AdaptiveOptimizer, self).__init__(model, params,
#                                                 update_lr_stepwise=False, parallel_mode=parallel_mode)
#         self.gamma = params['gamma']

#     def get_lr(self):
#         return self.lr * self.gamma


class LinearOptimizer(Optimizer):
    def __init__(self, model, params, parallel_mode='dp'):
        super(LinearOptimizer, self).__init__(model, params,
              update_lr_stepwise=False, parallel_mode=parallel_mode)

        self.init_lr = params['lr']
        self.final_lr = params['final_lr']

        self.epochs = params['epochs'] - 1
        self.decay_factor = (self.final_lr - self.init_lr) / self.epochs

    def get_lr(self):
        return self.decay_factor * self.global_epoch + self.init_lr



KindsOfOptimizer = {
    'fixed': FixedLROptimizer,
    'linear': LinearOptimizer
}
