import os
import sys
from setting import *
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
sys.path.append(PROJECT_ROOT)
font = FontProperties(fname=r"/usr/share/fonts/truetype/simplified_chinese.ttf", size=14)

class Vocab(object):
    def __init__(self):
        self.label_path = '../../data/chinese_formula_data/label.txt'


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
            line_strip = line_strip.split()
            line_len = len(line_strip)
            if line_len not in vocab:
                vocab[line_len] = 0
            vocab[line_len] += 1
    x = np.array([i for i in vocab])
    y = np.array([vocab[i] for i in vocab])
    plt.bar(x, y, width=0.5, align="center", yerr=0.000001)
    plt.xlabel(u'the length of the true label 我')
    plt.ylabel(u'the number of same label 他', fontproperties=font)
    plt.show()


if __name__ == '__main__':
    main()
