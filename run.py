import os
import yaml
import torch
import argparse
from otrans.model import Transformer
from otrans.optim import *
from otrans.train import Trainer
from otrans.data import AudioDataset


def main(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    with open(args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    expdir = os.path.join('egs', params['data']['name'], 'exp', params['train']['save_name'])
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    model = Transformer(params['model'])

    if args.ngpu >= 1:
        model.cuda()
    print(model)

    # build optimizer
    optimizer = TransformerOptimizer(model, params['train'], model_size=params['model']['d_model'],
                                     parallel_mode=args.parallel_mode)

    trainer = Trainer(params, model=model, optimizer=optimizer, is_visual=True, expdir=expdir, ngpu=args.ngpu,
                      parallel_mode=args.parallel_mode, local_rank=args.local_rank)

    train_dataset = AudioDataset(params['data'], 'train')
    trainer.train(train_dataset=train_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('-n', '--ngpu', type=int, default=1)
    parser.add_argument('-s', '--seed', type=int, default=1234)
    parser.add_argument('-p', '--parallel_mode', type=str, default='dp')
    parser.add_argument('-r', '--local_rank', type=int, default=0)
    cmd_args = parser.parse_args()

    if cmd_args.parallel_mode == 'ddp':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ["OMP_NUM_THREADS"] = '1'
        torch.cuda.set_device(cmd_args.local_rank)

    main(cmd_args)
