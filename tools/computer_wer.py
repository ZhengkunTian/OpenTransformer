# Author: Zhengkun Tian
import sys
import editdistance


# Usage: python compute_wer.py target_file predict_file

tgt_file = sys.argv[1]
dec_file = sys.argv[2]

total_words = 0
total_false = 0

target_dict = {}
with open(tgt_file, 'r', encoding='utf-8') as tf:
    for line in tf:
        parts = line.strip().split()
        idx = parts[0]
        words = parts[1:]
        target_dict[idx] = words

with open(dec_file, 'r', encoding='utf-8') as df:
    for line in df:
        parts = line.strip().split()
        idx = parts[0]
        words = parts[1:]
        ref_words = target_dict[idx]
        total_words += len(ref_words)
        diff = editdistance.eval(ref_words, words)
        total_false += diff

print('The WER/CER is %.2f' % 100 * total_false / total_words)


