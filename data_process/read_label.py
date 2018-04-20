import os
import sys
from setting import *

sys.path.append(PROJECT_ROOT)


class Vocab(object):
    def __init__(self):
        self.label_path = '../../data/chinese_formula_data/label.txt'


def main():
    parameters = Vocab()

    label_path = parameters.label_path
    assert os.path.exists(label_path), label_path

    vocab = []
    count = 0
    with open(label_path) as fin:
        for line in fin:
            line = line.strip().split('\t')
            if len(line) < 2:
                continue
            line_strip = line[1]
            line_strip = ' '.join(line_strip).split()
            print(line_strip)
            vocab.append(len(line_strip))
            count += 1
    print(max(vocab))


if __name__ == '__main__':
    main()
