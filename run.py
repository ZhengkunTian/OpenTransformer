# File   : run.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import re
import os
import yaml
import random
import logging
import shutil
import numpy as np
import torch
import argparse
from otrans.model import End2EndModel, LanguageModel
from otrans.train.scheduler import BuildOptimizer, BuildScheduler
from otrans.train.trainer import Trainer
from otrans.utils import count_parameters
from otrans.data.loader import FeatureLoader


def main(args, params, expdir):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    model_type = params['model']['type']
    if model_type[-2:] == 'lm':
        model = LanguageModel[model_type](params['model'])
    else:
        model = End2EndModel[model_type](params['model'])

    # Count total parameters
    count_parameters(model.named_parameters())

    if args.ngpu >= 1:
        model.cuda()
    logging.info(model)

    optimizer = BuildOptimizer[params['train']['optimizer_type']](
        filter(lambda p: p.requires_grad, model.parameters()), **params['train']['optimizer']
    )
    logger.info('[Optimizer] Build a %s optimizer!' % params['train']['optimizer_type'])
    scheduler = BuildScheduler[params['train']['scheduler_type']](optimizer, **params['train']['scheduler'])
    logger.info('[Scheduler] Build a %s scheduler!' % params['train']['scheduler_type'])

    if args.continue_training and args.init_model:
        chkpt = torch.load(args.init_model)
        model.load_model(chkpt)
        logger.info('[Continue Training] Load saved model %s' % args.init_model)

    if args.continue_training and args.init_optim_state:
        ochkpt = torch.load(args.init_optim_state)
        optimizer.load_state_dict(ochkpt['optim'])
        logger.info('[Continue Training] Load saved optimizer state dict!')

        global_step = ochkpt['global_step'] if 'global_step' in ochkpt else args.from_step
        scheduler.global_step = global_step
        scheduler.set_lr()
        logger.info('Set the global step to %d and init lr to %.6f' % (scheduler.global_step, scheduler.lr))

    trainer = Trainer(params, model=model, optimizer=optimizer, scheduler=scheduler, expdir=expdir, ngpu=args.ngpu,
                      parallel_mode=args.parallel_mode, local_rank=args.local_rank, is_debug=args.debug,
                      keep_last_n_chkpt=args.keep_last_n_chkpt, from_epoch=args.from_epoch, mixspeech=args.mixspeech)

    train_loader = FeatureLoader(params, 'train', ngpu=args.ngpu, mode=args.parallel_mode)

    trainer.train(train_loader=train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='egs/aishell/conf/transformer_base.yaml')
    parser.add_argument('-n', '--ngpu', type=int, default=1)
    parser.add_argument('-g', '--gpus', type=str, default='0')
    parser.add_argument('-se', '--seed', type=int, default=1234)
    parser.add_argument('-p', '--parallel_mode', type=str, default='dp')
    parser.add_argument('-r', '--local_rank', type=int, default=0)
    parser.add_argument('-l', '--logging_level', type=str, default='info', choices=['info','debug'])
    parser.add_argument('-lg', '--log_file', type=str, default=None)
    parser.add_argument('-mp', '--mixed_precision', action='store_true', default=False)
    parser.add_argument('-ct', '--continue_training', action='store_true', default=False)
    parser.add_argument('-dir', '--expdir', type=str, default=None)
    parser.add_argument('-im', '--init_model', type=str, default=None)
    parser.add_argument('-ios', '--init_optim_state', type=str, default=None)
    parser.add_argument('-debug', '--debug', action='store_true', default=False)
    parser.add_argument('-knpt', '--keep_last_n_chkpt', type=int, default=30)
    parser.add_argument('-tfs', '--from_step', type=int, default=0)
    parser.add_argument('-tfe', '--from_epoch', type=int, default=0)
    parser.add_argument('-vb', '--verbose', type=int, default=0)
    parser.add_argument('-ol', '--opt_level', type=str, choices=['O0', 'O1', 'O2', 'O3'], default='O1')
    parser.add_argument('-ms', '--mixspeech', type=int, default=0)
    cmd_args = parser.parse_args()

    with open(cmd_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if cmd_args.expdir is not None:
        expdir = os.path.join(cmd_args.expdir, params['train']['save_name'])
    else:
        expdir = os.path.join('egs', params['data']['name'], 'exp', params['train']['save_name'])
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    shutil.copy(cmd_args.config, os.path.join(expdir, 'config.yaml'))

    logging_level = {
        'info': logging.INFO,
        'debug': logging.DEBUG
    }

    if cmd_args.log_file is not None:
        log_file = cmd_args.log_file
    else:
        log_file = cmd_args.config.split('/')[-1][:-5] + '.log'
    
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level[cmd_args.logging_level], format=LOG_FORMAT)
    logger = logging.getLogger(__name__)

    if cmd_args.ngpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cmd_args.gpus)
        logger.info('Set CUDA_VISIBLE_DEVICES as %s' % cmd_args.gpus)

    if cmd_args.parallel_mode == 'ddp':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ["OMP_NUM_THREADS"] = '1'
        torch.cuda.set_device(cmd_args.local_rank)

    main(cmd_args, params, expdir)


