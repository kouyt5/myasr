from visdom import Visdom
import numpy as np
import torch

# assert viz.check_connection(timeout_seconds=3), \
#         'No connection could be formed quickly'

class VisdomLogger(object):
    def __init__(self, num_epochs):
        self.viz = Visdom(port=8097, server="http://chenc.icu")
        self.opts = dict(title='visdom', ylabel='', xlabel='Epoch', legend=['Loss', 'WER', 'CER'])
        self.viz_window = None
        self.epochs = torch.arange(1, num_epochs + 1)
        self.visdom_plotter = True

    def update(self, epoch, values):
        x_axis = self.epochs[0:epoch]
        loss = torch.Tensor(values.loss[0:epoch])
        y_axis = torch.stack((torch.Tensor(values.loss[:epoch]),
                              torch.Tensor(values.wers[:epoch]),
                              torch.Tensor(values.cers[:epoch])),
                             dim=1)
        self.viz_window = self.viz.line(
            X=x_axis,
            Y=y_axis,
            opts=self.opts,
            win=self.viz_window,
            update='replace' if self.viz_window else None
        )

    def load_previous_values(self, start_epoch, results_state):
        self.update(start_epoch - 1, results_state)  # Add all values except the iteration we're starting from
class State():
    def __init__(self):
        self.wers = []
        self.cers = []
        self.loss = []
    def append(self, epoch, loss, wer, cer):
        self.wers.append(wer)
        self.cers.append(cer)
        self.loss.append(loss)
    def get_len(self):
        return len(self.wers)