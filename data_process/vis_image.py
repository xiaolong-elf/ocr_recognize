import matplotlib.pyplot as plt
import cv2
import sys

from setting import *

sys.path.append(PROJECT_ROOT)


def display_result(img, gt_labels):
    # Usage: print_result(images[0],labels[0],predicted_labels[0])

    fig, axes = plt.subplots(2)
    for ax in axes:
        ax.set_xticks(())
        ax.set_yticks(())
    axes[0].set_title("Input image")
    axes[0].imshow(img[:, :, 0], cmap='gray')

    kwargs = {'x': 0.5, 'y': 0.5, 'size': 30, 'verticalalignment': 'center', 'horizontalalignment': 'center'}

    axes[1].set_title("[ Ground Truth ]")
    gt_formula = gt_labels
    # print('get formula:', gt_formula)
    try:
        axes[1].text(s=gt_formula, **kwargs)
    except ValueError:
        axes[1].text(s=gt_formula[1:-1], **kwargs)
    plt.show()


path = '../../data/baidu_data/baidu.lst'
image_root = '../../data/baidu_data/process_image/'

fi = open(path).read().split('\n')
for line in fi:
    line = line.strip().split('\t')
    if len(line) != 2:
        continue
    image = cv2.imread(image_root + line[0])
    label = line[1]
    print(label)
    display_result(image, label)






