import os
import yaml
import torch
import argparse
from otrans.model import Transformer
from otrans.recognizer import TransformerRecognizer
from otrans.data import load_vocab, AudioDataset, FeatureLoader


def main(args):

    checkpoint = torch.load(args.load_model)
    if 'params' in checkpoint:
        params = checkpoint['params']
    else:
        assert os.path.isfile(args.config), 'please specify a configure file.'
        with open(args.config, 'r') as f:
            params = yaml.load(f)

    params['data']['shuffle'] = False
    params['data']['spec_argument'] = False
    params['data']['short_first'] = False
    params['data']['batch_size'] = args.batch_size

    expdir = os.path.join('egs', params['data']['name'], 'exp', params['train']['save_name'])
    if args.suffix is None:
        decode_dir = os.path.join(expdir, 'decode_%s' % args.decode_set)
    else:
        decode_dir = os.path.join(expdir, 'decode_%s_%s' % (args.decode_set, args.suffix))

    if not os.path.exists(decode_dir):
        os.makedirs(decode_dir)

    model = Transformer(params['model'])

    model.load_state_dict(checkpoint['model'])
    print('Load pre-trained model from %s' % args.load_model)

    model.eval()
    if args.ngpu > 0:
        model.cuda()

    char2unit = load_vocab(params['data']['vocab'])
    unit2char = {i:c for c, i in char2unit.items()}

    dataset = AudioDataset(params['data'], args.decode_set)
    data_loader = FeatureLoader(dataset)
    
    recognizer = TransformerRecognizer(model, unit2char=unit2char, beam_width=args.beam_width,
                                       max_len=args.max_len, penalty=args.penalty, lamda=args.lamda, ngpu=args.ngpu)

    totals = len(dataset)
    batch_size = params['data']['batch_size']
    writer = open(os.path.join(decode_dir, 'predict.txt'), 'w')
    for step, (utt_id, batch) in enumerate(data_loader.loader):

        if args.ngpu > 0:
            inputs = batch['inputs'].cuda()
            inputs_length = batch['inputs_length'].cuda()

        preds = recognizer.recognize(inputs, inputs_length)

        targets = batch['targets']
        targets_length = batch['targets_length']

        for b in range(len(preds)):
            n = step * batch_size + b
            truth = ' '.join([unit2char[i.item()] for i in targets[b][1:targets_length[b]]])
            print('[%d / %d ] %s - pred : %s' % (n, totals, utt_id[b], preds[b]))
            print('[%d / %d ] %s - truth: %s' % (n, totals, utt_id[b], truth))
            writer.write(utt_id[b] + ' ' + preds[b] + '\n')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('-n', '--ngpu', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-bw', '--beam_width', type=int, default=5)
    parser.add_argument('-p', '--penalty', type=float, default=0.6)
    parser.add_argument('-ld', '--lamda', type=float, default=5)
    parser.add_argument('-m','--load_model', type=str, default=None)
    parser.add_argument('-d', '--decode_set', type=str, default='test')
    parser.add_argument('-ml', '--max_len', type=int, default=100)
    parser.add_argument('-s', '--suffix', type=str, default=None)
    cmd_args = parser.parse_args()

    main(cmd_args)
