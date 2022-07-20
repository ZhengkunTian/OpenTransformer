import re
import os
import yaml
import time
import torch
import logging
import argparse
import editdistance
from otrans.model import End2EndModel, LanguageModel
from otrans.recognize import build_recognizer
from otrans.data.loader import FeatureLoader
from otrans.train.utils import map_to_cuda


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def main(args):

    checkpoint = torch.load(args.load_model)

    if args.config is not None:
        with open(args.config, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    else:
        params = checkpoint['params']

    params['data']['batch_size'] = args.batch_size
    model_type = params['model']['type']
    model = End2EndModel[model_type](params['model'])

    if 'frontend' in checkpoint:
        model.frontend.load_state_dict(checkpoint['frontend'])
        logger.info('[FrontEnd] Load the frontend checkpoint!')

    model.encoder.load_state_dict(checkpoint['encoder'])
    logger.info('[Encoder] Load the encoder checkpoint!')

    if 'decoder' in checkpoint:
        model.decoder.load_state_dict(checkpoint['decoder'])
        logger.info('[Decoder] Load the decoder checkpoint!')

    if 'joint' in checkpoint:
        model.joint.load_state_dict(checkpoint['joint'])
        logger.info('[JointNet] Load the joint net of transducer checkpoint!')

    if 'look_ahead_conv' in checkpoint:
        model.lookahead_conv.load_state_dict(checkpoint['look_ahead_conv'])
        logger.info('[LookAheadConvLayer] Load the external lookaheadconvlayer checkpoint!')

    if 'ctc' in checkpoint:
        model.assistor.load_state_dict(checkpoint['ctc'])
        logger.info('[CTC Assistor] Load the ctc assistor checkpoint!')
    logger.info('Finished! Loaded pre-trained model from %s' % args.load_model)

    model.eval()
    if args.ngpu > 0:
        model.cuda()

    if args.load_language_model is not None:
        lm_chkpt = torch.load(args.load_language_model)
        lm_parms = lm_chkpt['params']
        lm_type = lm_parms['model']['type']
        lm = LanguageModel[lm_type](lm_parms['model'])
        lm.load_state_dict(lm_chkpt['model'])
        logger.info('Load pre-trained language model from %s' % args.load_language_model)
        lm.eval()
        if args.ngpu > 0: lm.cuda()
    else:
        lm = None
        lm_type = None   

    data_loader = FeatureLoader(params, args.decode_set, is_eval=True)
    idx2unit = data_loader.dataset.idx2unit
    recognizer = build_recognizer(model_type, model, lm, args, idx2unit)
  
    totals = len(data_loader.dataset)

    expdir = os.path.join('egs', params['data']['name'], 'exp', params['train']['save_name'])
    decoder_folder_name = ['decode']
    decoder_folder_name.append(args.decode_set)
    decoder_folder_name.append(args.mode)
    if args.mode != 'greedy':
        decoder_folder_name.append('%d' % args.beam_width)
    if args.load_language_model is not None:
        decoder_folder_name.append('%s_%.2f' % (lm_type, args.lm_weight))
    if args.ctc_weight > 0.0:
        decoder_folder_name.append('ctc_weight_%.3f' % args.ctc_weight)
    if args.ngram_lm is not None:
        decoder_folder_name.append('ngram_alpha%.2f_beta%.2f' % (args.alpha, args.beta))
    if args.apply_rescoring:
        decoder_folder_name.append('rescore')
        decoder_folder_name.append('rw_%.2f' % args.rescore_weight)
    if args.apply_lm_rescoring:
        decoder_folder_name.append('lm_rescore')
        decoder_folder_name.append('rw_%.2f' % args.rescore_weight)
    try:
        ep = re.search(r'from(\d{1,3})to(\d{1,3})', args.load_model).groups()
        decoder_folder_name.append('_'.join(list(ep)))
    except:
        ep = re.search(r'epoch.(\d{1,3}).pt', args.load_model).groups()[0]
        decoder_folder_name.append('epoch_%s' % ep)

    if args.debug:
        decoder_folder_name.append('debug_%d_samples' % args.num_sample)

    if args.suffix is not None:
        decoder_folder_name.append(args.suffix)
    decode_dir = os.path.join(expdir, '_'.join(decoder_folder_name))

    if not os.path.exists(decode_dir):
        os.makedirs(decode_dir)
    
    writer = open(os.path.join(decode_dir, 'predict.txt'), 'w')
    detail_writer = open(os.path.join(decode_dir, 'predict.log'), 'w')

    top_n_false_tokens = 0
    false_tokens = 0
    total_tokens = 0
    accu_time = 0
    total_frames = 0
    for step, (utt_id, inputs, targets) in enumerate(data_loader.loader):

        if args.ngpu > 0:
            inputs = map_to_cuda(inputs)
            
        enc_inputs = inputs['inputs']
        enc_mask = inputs['mask']

        if args.batch_size == 1:
            total_frames += enc_inputs.size(1)

        st = time.time()
        preds, scores = recognizer.recognize(enc_inputs, enc_mask)
        et = time.time()
        span = et - st
        accu_time += span

        truths = targets['targets']
        truths_length = targets['targets_length']

        for b in range(len(preds)):
            n = step * args.batch_size + b

            truth = [idx2unit[i.item()] for i in truths[b][1:truths_length[b]]]
            if args.piece2word:
                truth = ''.join(truth).replace('▁', ' ')
            else:
                truth = ' '.join(truth)

            print_info = '[%d / %d ] %s - truth             : %s' % (n, totals, utt_id[b], truth)
            logger.info(print_info)
            detail_writer.write(print_info+'\n')
            total_tokens += len(truth.split())  

            nbest_min_false_tokens = 1e10
            for i in range(len(preds[b])):
                
                pred = preds[b][i]
                if args.piece2word:
                    pred = ''.join(preds[b][i].split()).replace('▁', ' ')
                _truth = truth.replace("<PESN> ", "").replace("<VIET> ", "").replace("<SWAH> ", "")
                _pred = pred.replace("<PESN> ", "").replace("<VIET> ", "").replace("<SWAH> ", "")
                n_diff = editdistance.eval(_truth.split(), _pred.split())
                if i == 0:
                    false_tokens += n_diff
                nbest_min_false_tokens = min(nbest_min_false_tokens, n_diff)
                              
                print_info = '[%d / %d ] %s - pred-%2d (%3.4f) : %s' % (n, totals, utt_id[b], i, float(scores.cpu()[b, i]), pred)
                logger.info(print_info)
                detail_writer.write(print_info+'\n')
                
            writer.write(utt_id[b] + ' ' + preds[b][0] + '\n')
            top_n_false_tokens += nbest_min_false_tokens

            detail_writer.write('\n')

        if args.debug and (step+1) * args.batch_size >= args.num_sample:
            break

    writer.close()
    detail_writer.close()

    with open(os.path.join(decode_dir, 'RESULT'), 'w') as w:

        wer = false_tokens / total_tokens * 100
        logger.info('The WER is %.3f.' % wer)
        topn_wer = top_n_false_tokens / total_tokens * 100
        logger.info('The top %d WER is %.3f' % (args.nbest, topn_wer))
        w.write('The Model Chkpt: %s \n' % args.load_model)
        if model_type == 'ctc':
            w.write('Decode Mode: %s \n' % args.mode)
        w.write('The WER is %.3f. \n' % wer)

        if args.batch_size == 1:
            rtf = accu_time / total_frames * 100
            logger.info('The RTF is %.6f' % rtf)
            w.write('The RTF is %.6f' % rtf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('-n', '--ngpu', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-nb', '--nbest', type=int, default=1)
    parser.add_argument('-bw', '--beam_width', type=int, default=5)
    parser.add_argument('-pn', '--penalty', type=float, default=0.6)
    parser.add_argument('-ld', '--lamda', type=float, default=5)
    parser.add_argument('-m', '--load_model', type=str, default=None)
    parser.add_argument('-lm', '--load_language_model', type=str, default=None)
    parser.add_argument('-ngram', '--ngram_lm', type=str, default=None)
    parser.add_argument('-alpha', '--alpha', type=float, default=0.1)
    parser.add_argument('-beta', '--beta', type=float, default=0.0)
    parser.add_argument('-lmw', '--lm_weight', type=float, default=0.1)
    parser.add_argument('-cw', '--ctc_weight', type=float, default=0.0)
    parser.add_argument('-d', '--decode_set', type=str, default='test')
    parser.add_argument('-ml', '--max_len', type=int, default=60)
    parser.add_argument('-md', '--mode', type=str, default='beam')
    # transducer related
    parser.add_argument('-mt', '--max_tokens_per_chunk', type=int, default=5)
    parser.add_argument('-pf', '--path_fusion', action='store_true', default=False)
    parser.add_argument('-s', '--suffix', type=str, default=None)
    parser.add_argument('-p2w', '--piece2word', action='store_true', default=False)
    parser.add_argument('-resc', '--apply_rescoring', action='store_true', default=False)
    parser.add_argument('-lm_resc', '--apply_lm_rescoring', action='store_true', default=False)
    parser.add_argument('-rw', '--rescore_weight', type=float, default=1.0)
    parser.add_argument('-debug', '--debug', action='store_true', default=False)
    parser.add_argument('-sba', '--sort_by_avg_score', action='store_true', default=False)
    parser.add_argument('-ns', '--num_sample', type=int, default=1)
    cmd_args = parser.parse_args()

    main(cmd_args)
