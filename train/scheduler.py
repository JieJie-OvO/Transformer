import torch
import math
import numpy as np

class BaseScheduler(object):
    def __init__(self, optimizer, stepwise=False):

        # Attach optimizer
        self.optimizer = optimizer
        self.global_step = 1
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

class TransformerScheduler(BaseScheduler):
    def __init__(self, optimizer, model_size, warmup_steps, factor=1.0):
        
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        super(TransformerScheduler, self).__init__(optimizer, stepwise=True)  

    def get_step_lr(self, step):
        return self.factor * self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
    
class DecayScheduler(BaseScheduler):
    def __init__(self, optimizer, base=0.0001, decay=0.5):
        
        self.base = base
        self.decay = decay
        super(DecayScheduler, self).__init__(optimizer, stepwise=False)  

    def get_epoch_lr(self, epoch):
        if epoch % 2 == 0:
            self.base = self.base * self.decay
        return self.base