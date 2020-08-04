import sys

text_in = sys.argv[1]
text_out = sys.argv[2]

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


with open(text_in, 'r', encoding='utf-8') as f1:
    with open(text_out, 'w', encoding='utf-8') as f2:
        for line in f1:
            parts = line.strip().split()
            utt_id = parts[0]
            norm_text = text_norm(' '.join(parts[1:]))
            f2.write(utt_id+' '+norm_text+'\n')