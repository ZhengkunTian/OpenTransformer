import torch
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from otrans.data import PAD
from torch.utils.tensorboard import SummaryWriter


class MeanLoss(object):
    def __init__(self, num=100):
        self.num = num
        self.losses = []

    def update(self, loss):
        if len(self.losses) >= self.num:
            self.losses = self.losses[1:]

        self.losses.append(loss)
        assert len(self.losses) <= self.num, 'Please only keep the 100 latest losses!'

    def tensor2numpy(self, tensor):
        return tensor.cpu().numpy()

    def mean(self):
        return np.mean(self.losses) if len(self.losses) > 0 else 0.0


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


class Visulizer(object):
    def __init__(self, log_dir=None):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)

    def add_graph(self, model):
        self.writer.add_graph(model)

    def add_image(self, tag, img, data_formats):
        self.writer.add_image(tag, img, 0, dataformats=data_formats)

    def add_img_figure(self, tag, img, step=None):
        fig, axes = plt.subplots(1, 1)
        axes.imshow(img)
        self.writer.add_figure(tag, fig, global_step=step)

    def close(self):
        self.writer.close()


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


class AuxiliaryLossAverageMeter(object):
    def __init__(self):
        self.aux_loss = {}

    def update(self, value_dict, accu_steps, n=1):
        if value_dict is not None:
            for key, value in value_dict.items():
                if key not in self.aux_loss:
                    self.aux_loss[key] = AverageMeter()
                    self.aux_loss[key].update(value / accu_steps, n)
                else:
                    self.aux_loss[key].update(value / accu_steps, n)

    def reset(self):
        if len(self.aux_loss) > 0:
            for key in self.aux_loss.keys():
                self.aux_loss[key].reset()

    @property
    def avg_infos(self):
        if len(self.aux_loss) == 0:
            return ""
        else:
            infos = []
            for key in self.aux_loss.keys():
                infos.append('%s: %.3f' % (key, self.aux_loss[key].avg))
            return ", "+ ", ".join(infos)


def dict_to_print(dict_info):

    strings = ""
    for key, value in dict_info.items():
        strings+= '%s: %.3f ' % (key, value)

    return strings 


class Summary(object):
    def __init__(self):
        self.note = {}

    def best(self):
        sorted_list = sorted(self.note.items(), key=lambda x: x[1], reverse=False)
        return sorted_list[0]

    def update(self, epoch, loss):
        self.note[epoch] = loss


def map_to_cuda(tensor_dict):
    cuda_tensor_dict = {}
    for key, value in tensor_dict.items():
        cuda_tensor_dict[key] = value.cuda()
    
    return cuda_tensor_dict
