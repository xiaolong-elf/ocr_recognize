import cv2
from PIL import Image
import numpy as np
import os, sys
import glob
from setting import *

sys.path.append(PROJECT_ROOT)

# buckets_size = [[32, 40], [32, 50], [32, 75], [32, 100], [32, 125], [32, 150], [32, 200], [32, 250], [32, 300],
#                 [32, 350], [32, 400], [32, 450], [32, 500], [32, 550], [32, 600], [32, 700], [32, 800], [32, 900],
#                 [32, 1000], [32, 1100], [32, 1200], [32, 1300], [32, 1400], [32, 1500], [32, 1600]]

buckets_size = [[46, 200], [46, 300], [46, 350], [46, 400], [46, 450], [46, 500], [46, 550], [46, 600],
                [46, 700], [46, 800], [46, 900], [46, 1000], [46, 1100], [46, 1200], [46, 1300], [46, 1400],
                [46, 1500], [46, 1600]]
#
# buckets_size = [[32, 200], [32, 300], [32, 350], [32, 400], [32, 450], [32, 500], [32, 550], [32, 600],
#                 [32, 700], [32, 800], [32, 900], [32, 1000], [32, 1100], [32, 1200], [32, 1300], [32, 1400],
#                 [32, 1500], [32, 1600]]


def resize(im, max_height=42):
    height, weight = im.shape[:2]
    im_scale_h = max_height / height
    im = cv2.resize(im, None, None, fx=im_scale_h, fy=im_scale_h, interpolation=cv2.INTER_LINEAR)
    return im


def pad_group_image(old_im, output_path=None):
    buckets = buckets_size
    PAD_TOP, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT = [2, 2, 2, 2]
    # old_size = (old_im.shape[0] + PAD_TOP + PAD_BOTTOM, old_im.shape[1] + PAD_LEFT + PAD_RIGHT)
    j = -1
    for i in range(len(buckets)):
        if old_im.shape[0] <= buckets[i][0] and old_im.shape[1] <= buckets[i][1]:
            j = i
            break
    # if j < 0:
    #     new_im = cv2.copyMakeBorder(old_im, PAD_TOP,  PAD_BOTTOM, PAD_LEFT, PAD_RIGHT, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #     print('#######', new_im.shape)
    #     cv2.imwrite('%s' % output_path, new_im)
    # else:
    if j != -1:
        top = int((buckets[j][0] - old_im.shape[0]) / 2)
        bottom = buckets[j][0] - old_im.shape[0] - top
        left = int((buckets[j][1] - old_im.shape[1]) / 2)
        right = buckets[j][1] - old_im.shape[1] - left
        new_im = cv2.copyMakeBorder(old_im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        assert new_im.shape[0] == 46 and new_im.shape[1] <= 1600
        print('@@@@@', new_im.shape)
        # cv2.imwrite('%s' % output_path, new_im)
        return new_im

if __name__ == '__main__':
    path = '../../data/chinese_formula_data/processed_image'
    output_path = '../../data/chinese_formula_data/processed_image/'
    image_list = glob.glob(os.path.join(path, '*jpg'))
    image_list_png = glob.glob(os.path.join(path, '*png'))
    image_list = image_list + image_list_png

    for i in image_list:
        new_path = os.path.join(output_path, os.path.basename(i))
        image = cv2.imread(i)
        image = resize(image)
        pad_group_image(image, new_path)

