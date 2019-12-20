import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from otrans.data import PAD
from otrans.module import LayerNorm


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
        return np.mean(self.losses)


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


def get_enc_padding_mask(tensor, tensor_length):
    return torch.sum(tensor, dim=-1).ne(0).unsqueeze(-2)


def get_seq_mask(targets):
    batch_size, steps = targets.size()
    seq_mask = torch.ones([batch_size, steps, steps], device=targets.device)
    seq_mask = torch.tril(seq_mask).bool()
    return seq_mask


def get_dec_seq_mask(targets, targets_length=None):
    steps = targets.size(-1)
    padding_mask = targets.ne(PAD).unsqueeze(-2)
    seq_mask = torch.ones([steps, steps], device=targets.device)
    seq_mask = torch.tril(seq_mask).bool()
    seq_mask = seq_mask.unsqueeze(0)
    return seq_mask & padding_mask


def get_length_mask(tensor, tensor_length):
    b, t, _ = tensor.size()  
    mask = tensor.new_ones([b, t], dtype=torch.uint8)
    for i, length in enumerate(tensor_length):
        length = length.item()
        mask[i].narrow(0, 0, length).fill_(0)
    return mask.bool()


def initialize(model, init_type="pytorch"):
    """Initialize Transformer module

    :param torch.nn.Module model: transformer instance
    :param str init_type: initialization type
    """
    # weight init
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(p.data)
            elif init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(p.data)
            elif init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError("Unknown initialization: " + init_type)
    # bias init
    for p in model.parameters():
        if p.dim() == 1:
            p.data.zero_()

    # reset some modules with default init
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, LayerNorm)):
            m.reset_parameters()


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

# class Reportor(object):
#     def __init__():

#     def 
