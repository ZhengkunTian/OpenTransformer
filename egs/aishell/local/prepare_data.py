'''
@Author: Zhengkun Tian
@Email: zhengkun.tian@outlook.com
@Date: 2020-06-12 12:07:29
@LastEditTime: 2020-06-12 15:04:43
@FilePath: \OpenTransducer\egs\aishell\local\prepare_data.py
'''
import os
import sys
import glob


def text_norm(seq):
    new_seq = ''
    for s in seq:
        inside_code = ord(s)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        new_s= chr(inside_code)

        if new_s.encode('UTF-8').isalpha():
            new_s = new_s.upper()

        new_seq += new_s
        new_seq += ' '

    return ' '.join(new_seq.split())


datadir = sys.argv[1]   # /data2/tianzhengkun/data/aishell2/
outputdir = sys.argv[2]

trans = {}
with open(os.path.join(datadir, 'data_aishell/transcript/aishell_transcript_v0.8.txt'), 'r', encoding='utf-8') as r:
    for line in r:
        parts = line.strip().split()
        idx = parts[0]
        trans[idx] = text_norm(''.join(parts[1:]))


for name in ['train', 'dev', 'test']:

    wav_dir = os.path.join(datadir, 'data_aishell/wav', name)
    wav_list = glob.glob(wav_dir+'/*/*.wav')

    if not os.path.exists(os.path.join(outputdir, name)):
        os.makedirs(os.path.join(outputdir, name))

    with open(os.path.join(outputdir, name, 'wav.scp'), 'w') as w1:
        with open(os.path.join(outputdir, name, 'text'), 'w', encoding='utf-8') as w2:
            for wav in wav_list:
                idx = wav.split('/')[-1][:-4]
                w1.write(idx+' '+wav+'\n')
                if idx not in trans:
                    print('Skip %s in %s set!' % (idx, name))
                    continue
                context = trans[idx]
                w2.write(idx+' '+context+'\n')
    print('There are %d utterances in %s wav.scp' % (len(wav_list), name))              
    print('%s set is dealed!' % name)

