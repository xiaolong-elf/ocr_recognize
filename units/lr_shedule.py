import numpy as np


class LRSchedule(object):
    def __init__(self, lr_init=1e-3, lr_min=1e-4, total_step=None, lr_type=None):
        # store parameters
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.lr = lr_init
        self.total_step = total_step
        self.lr_type = lr_type

    def update(self, batch_no=None):
        """
        Update the learning rate:
            - decay by self.decay rate if score is lower than previous best
            - decay by self.decay_rate
        """
        if self.lr_type == 'pow':
            base_lr = self.lr_init
            if batch_no > self.total_step / 2:
                base_lr = self.lr_init / 100
            self.lr = np.power(1 - batch_no / self.total_step, 0.7) * (base_lr - self.lr_min) + self.lr_min

        elif self.lr_type == 'uniform':
            if batch_no < self.total_step:
                self.lr = self.lr_init - ((self.lr_init - self.lr_min) / self.total_step) * batch_no
            else:
                self.lr = self.lr_min
        else:
            self.lr = self.lr
