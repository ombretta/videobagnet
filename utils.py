import csv
import random
from functools import partialmethod

import torch
from torchmetrics.classification import Accuracy, AveragePrecision
#from torchmetrics.functional import Accuracy, AveragePrecision
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_RF_size(model):
    named_layers = dict(model.named_modules())
    downsampling = []
    l = []  # [[1, 1, 1]]
    i = 0

    for layer in named_layers.keys():
        curr_layer = named_layers[layer]
        # n.b.:
        if 'kernel_size' in curr_layer.__dict__ and 'downsample' not in layer:
            k = curr_layer.kernel_size
            s = curr_layer.stride
            if type(k) == int: k, s = (k, k, k), (s, s, s)
            downsampling.append({'f': k, 's': s})

            dims = len(downsampling[i]['f'])
            if l == []: l = [[1] * dims]

            si, li = [], []
            for dim in range(min(dims, len(l[i]))):
                si_d = np.prod([x['s'][dim] for x in downsampling[:i]])
                li_d = l[i][dim] + (downsampling[i]['f'][dim] - 1) * si_d
                si += [si_d]
                li += [li_d]
            l.append(li)
            print("%d) The receptive field after %s is %s"
                  % (i + 1, layer, li))
            i += 1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets, multilabel=False):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        #print("outputs", outputs, "pred", pred)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size

def calculate_accuracy_pytorch(outputs, targets, multilabel=False, n_classes=10):
    with torch.no_grad():
        if not multilabel:
            pred = torch.nn.Softmax(dim=1)(outputs).cpu()
            #print(pred, targets)
            accuracy = Accuracy(task="multiclass", num_classes=n_classes)
            return accuracy(pred, targets.to('cpu', dtype=torch.int))
        else:
            pred = torch.nn.Sigmoid()(outputs).cpu()
            #print(pred, targets.to('cpu', dtype=torch.int))
            average_precision = AveragePrecision(task="multiclass", num_classes=n_classes, average='macro')
            AP = average_precision(pred, targets.to('cpu', dtype=torch.int))
            return AP

def calculate_precision_and_recall(outputs, targets, pos_label=1):
    with torch.no_grad():
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        precision, recall, _, _ = precision_recall_fscore_support(
            targets.view(-1, 1).cpu().numpy(),
            pred.cpu().numpy())

        return precision[pos_label], recall[pos_label]


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)


def partialclass(cls, *args, **kwargs):

    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass
