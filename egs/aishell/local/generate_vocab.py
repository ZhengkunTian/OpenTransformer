'''
@Author: Zhengkun Tian
@Email: zhengkun.tian@outlook.com
@Date: 2020-06-12 12:06:19
@LastEditTime: 2020-06-12 14:56:44
@FilePath: \OpenTransducer\egs\aishell\local\generate_vocab.py
'''
import os
import sys


if __name__ == '__main__':

    text_in = sys.argv[1]
    vocab_out = sys.argv[2]

    lexicon = {}
    with open(text_in, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            idx = parts[0]
            phones = parts[1:]

            for p in phones:
                if p not in lexicon:
                    lexicon[p] = 1
                else:
                    lexicon[p] += 1

    print('There are %d label in lexicon!' % len(lexicon))

    vocab = sorted(lexicon.items(), key=lambda x: x[1], reverse=True)

    index = 3
    with open(vocab_out, 'w') as w:
        w.write('<PAD> 0\n')
        w.write('<S/E> 1\n')
        w.write('<UNK> 2\n')
        for (l, n) in vocab:
            w.write(l+' '+str(index)+'\n')
            index += 1

    print('Done!')
