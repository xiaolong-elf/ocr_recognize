import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mathtext import MathTextParser
import sys
from setting import *

sys.path.append(PROJECT_ROOT)

STA = "_STA"  # for start words
UNK = "_UNK"  # for unknown words
PAD = "_PAD"  # for padding
END = "_END"


def load_vocab(filename):
    """
    Args:
        filename: (string) path to vocab txt file one word per line
    Returns:
        dict: d[token] = id
    """
    vocab = dict()
    with open(filename) as f:
        for idx, token in enumerate(f):
            token = token.strip()
            vocab[idx + 4] = token

    # add pad and unk tokens
    vocab[0] = "_STA"
    vocab[1] = '_PAD'
    vocab[2] = '_UNK'
    vocab[3] = '_END'

    return vocab


# vocab = open('data/latex_vocab.txt').read().split('\n')
vocab_to_idx = load_vocab('../data/chinese_formula_data/vocab.txt')


# vocab_to_idx = dict([(vocab[i], i) for i in range(len(vocab))])


def convert_to_formula(labels):
    real_label = [vocab_to_idx[i] for i in labels]
    label_copy = real_label.copy()
    for k in label_copy:
        if k in ['_STA', '_PAD', '_UNK', '_END']:
            real_label.remove(k)
    real_label = ''.join(real_label)
    return real_label


def display_result(img, gt_labels, predicted_labels):
    # Usage: print_result(images[0],labels[0],predicted_labels[0])

    fig, axes = plt.subplots(3)
    for ax in axes:
        ax.set_xticks(())
        ax.set_yticks(())
    axes[0].set_title("Input image")
    axes[0].imshow(img[:, :, 0], cmap='gray')

    kwargs = {'x': 0.5, 'y': 0.5, 'size': 30, 'verticalalignment': 'center', 'horizontalalignment': 'center'}

    axes[1].set_title("Prediction")
    predicted_formula = convert_to_formula(predicted_labels)
    print('predict label:', predicted_formula)
    try:
        axes[1].text(s=predicted_formula, **kwargs)
    except ValueError:
        kwargs['size'] = 15
        axes[1].text(s=predicted_formula[1:-1], **kwargs)

    axes[2].set_title("[ Ground Truth ]")
    gt_formula = convert_to_formula(gt_labels)
    print('get formula:', gt_formula)
    try:
        axes[2].text(s=gt_formula, **kwargs)
    except ValueError:
        axes[2].text(s=gt_formula[1:-1], **kwargs)
    plt.show()


def plot_bbox(ax, bbox, score):
    """Plot bounding-box on matplotlib ax"""
    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='red', linewidth=2)
    )
    if score > 0.5:
        ax.text(bbox[0], bbox[1] - 2,
                '{:.3f}'.format(score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')


def plot_attention(image, predicted_labels, rf_coords, alignment_history, time=0):
    """Plot the highest attention region for image at time t in the decoding"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].set_title("Highest alignment score")
    axes[0].imshow(np.squeeze(image), cmap='gray')
    rf_coords_flat = np.reshape(rf_coords, (-1, 4))

    idx = alignment_history.argmax(axis=1)
    i, score = idx[time], alignment_history[np.arange(len(alignment_history)), idx][time]
    bbox = rf_coords_flat[i]
    plot_bbox(axes[0], bbox, score)

    check_parser = MathTextParser('MacOSX')

    kwargs = {'x': 0.0, 'y': 0.5, 'size': 30, 'verticalalignment': 'center', 'horizontalalignment': 'left'}

    axes[1].set_title("Prediction so far")
    predicted_formula = "$" + convert_to_formula(predicted_labels[0][:time]) + "$"
    try:
        check_parser.parse(predicted_formula)
        axes[1].text(s=predicted_formula, **kwargs)
    except ValueError:
        kwargs['size'] = 15
        axes[1].text(s=predicted_formula[1:-1].replace(' ', ''), **kwargs)
