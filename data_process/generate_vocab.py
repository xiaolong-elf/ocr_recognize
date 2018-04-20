import os
import sys
from setting import *
sys.path.append(PROJECT_ROOT)


class Vocab(object):
    def __init__(self):
        self.label_path = '../..//data/baidu_data/baidu.lst'
        self.output_file = '../../data/baidu_data/vocab.txt'
        self.unk_threshold = 1


def main():
    parameters = Vocab()

    label_path = parameters.label_path
    assert os.path.exists(label_path), label_path

    vocab = {}
    with open(label_path) as fin:
        for line in fin:
            line = line.strip().split('\t')
            if len(line) < 2:
                continue
            line_strip = line[1]
            line_strip = ' '.join(line_strip).split()
            # tokens = line_strip.split()
            print(line_strip)
            for token in line_strip:
                if token == '角d':
                    print(line)
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
    vocab_sort = sorted(list(vocab.keys()))
    vocab_out = []
    num_unknown = 0
    for word in vocab_sort:
        if vocab[word] > parameters.unk_threshold:
            vocab_out.append(word)
        else:
            num_unknown += 1
    vocab = [word for word in vocab_out]
    with open(parameters.output_file, 'w') as fout:
        fout.write('\n'.join(vocab))


if __name__ == '__main__':
    main()
