import os
import torch
import math
import time
import logging
from .utils import MeanLoss, AverageMeter, map_to_cuda, Visulizer
import numpy as np

class Trainer(object):
    def __init__(self, model, optimizer, scheduler, epochs=60, accum_steps=4, 
                 grad_clip=5, keep_last_n_chkpt=20, ngpu=1,from_epoch=0,):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_epochs = epochs
        self.accum_steps = accum_steps
        self.grad_clip = grad_clip
        self.global_training_step = 1
        self.log_interval = 10
        self.mean_loss = MeanLoss()
        self.keep_last_n_chkpt = keep_last_n_chkpt
        self.ngpu = ngpu

        self.from_epoch = from_epoch
        
        self.visulizer = Visulizer(log_dir='./visual/')

    def train(self, train_loader):
        for epoch in range(self.from_epoch, self.total_epochs):
            train_loss = self.train_one_epoch(epoch, train_loader.loader)
            self.scheduler.epoch()

            print('-*Train-Epoch-%d/%d*-, AvgLoss:%.5f' % (epoch, self.total_epochs, train_loss))
            self.visulizer.add_scalar('train_epoch_loss', train_loss, epoch)

            self.save_model(epoch)
            self.save_optimizer_state_dict()
            self.clear_checkpoint(epoch)

        self.optimizer.zero_grad()

    def train_one_epoch(self, epoch, train_loader):

        self.model.train()
        batch_steps = len(train_loader)

        step_loss = AverageMeter()
        span = 0
        for step, (_, inputs, targets) in enumerate(train_loader):

            if self.ngpu > 0:
                inputs = map_to_cuda(inputs)
                targets = map_to_cuda(targets)

            start = time.time()

            loss = self.model(inputs, targets)

            loss = torch.mean(loss) / self.accum_steps

            # 积累loss，之后step进行更新
            loss.backward()
            end = time.time()
            span += (end - start)

            step_loss.update(loss.item() * self.accum_steps, inputs['inputs'].size(0))

            if self.global_training_step % self.accum_steps == 0:
                self.mean_loss.update(step_loss.avg)

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                if math.isnan(grad_norm):
                    print('Grad norm is NAN. DO NOT UPDATE MODEL!')
                else:
                    self.scheduler.step()
                    self.optimizer.step()

                    self.visulizer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'] , self.scheduler.global_step)

                self.optimizer.zero_grad()

                if self.scheduler.global_step % self.log_interval == 0:
                    process = (step + 1) / batch_steps * 100
                    print_info = "-Training-Epoch-%d(%.5f%%), Global Step:%d, lr:%.8f, Loss:%.5f, AvgLoss: %.5f, Run Time:%.3f" \
                        % (epoch, process, self.scheduler.global_step, self.scheduler.lr, step_loss.avg, self.mean_loss.mean(), span)
                    print(print_info)
                    
                    span = 0

                step_loss.reset()

            self.global_training_step += 1

        return self.mean_loss.mean()

    def save_model(self, epoch, path="./pth/"):

        save_name = 'model.epoch.%d.pth' % epoch
        path = path + save_name

        self.model.save_model(path)

        print('Save the model checkpoint!')

    def save_optimizer_state_dict(self, path="./pth/"):
        save_name = 'latest_optimizer.pth'
        path = path + save_name
        checkpoint = {
            'global_step': self.scheduler.global_step,
            'optim': self.optimizer.state_dict()
        }

        torch.save(checkpoint, path)
        print('Save the optimizer checkpoint!')

    def clear_checkpoint(self, epoch, path="./pth/"):
        if epoch + 1 > self.keep_last_n_chkpt:
            save_name = 'model.epoch.%d.pth' % (epoch - self.keep_last_n_chkpt)
            path = path + save_name
            if os.path.isfile(path):
                os.remove(path)
        else:
            print('There are no any checkpoints to be cleaned!')

    def load_model(self, checkpoint):
        chkpt = torch.load(checkpoint)
        self.model.load_model(chkpt)
