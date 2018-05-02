import glob
import os
import cv2
import numpy as np
import random
from multiprocessing import Pool
from functools import partial


buckets_size = [[32, 200], [32, 300], [32, 350], [32, 400], [32, 450], [32, 500], [32, 550], [32, 600],
                [32, 700], [32, 800], [32, 900], [32, 1000], [32, 1100], [32, 1200], [32, 1300], [32, 1400],
                [32, 1500], [32, 1600]]


def resize(im, max_height=28):
    height, weight = im.shape[:2]
    im_scale_h = max_height / height
    im = cv2.resize(im, None, None, fx=im_scale_h, fy=im_scale_h, interpolation=cv2.INTER_LINEAR)
    return im


def crop_image(img):
    img = cv2.imread(img, 0)
    img_data = np.asarray(img, dtype=np.uint8)  # height, width
    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0 or len(nnz_inds[1]) == 0:
        return 0
    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    img = img[y_min: y_max, x_min: x_max]
    # img = img.crop((x_min + 1, y_min + 1, x_max + 1, y_max + 1))
    # cv2.imshow(' ', img)
    # cv2.waitKey()
    img[img == 255] = 150

    return img


def merge_img(image1, image2):
    image1 = resize(image1)
    image2 = resize(image2)
    image = np.concatenate([image1, image2], axis=1)
    return image


def generate(row):
    row = row.strip().split(' ')
    latxt_image_path = os.path.join(latxt_image_root, row[1] + '.png')
    latxt_image_label = latxt_label_lst[int(row[0])]
    chinese_label = random.sample(right_chinese_label, 1)
    chinese_image_path = os.path.join(chinese_image_root, chinese_label[0][0])
    chinese_image_label = chinese_label[0][1]
    merge_label = ' '.join(chinese_image_label) + ' ' + latxt_image_label
    latxt_image = crop_image(latxt_image_path)
    if np.sum(latxt_image) > 0:
        image_name = row[1] + '_' + chinese_label[0][0]
        if len(merge_label) < 400:
            fo.write(image_name + '\t' + merge_label + '\n')
            chinese_image = cv2.imread(chinese_image_path, 0)
            merge_image = merge_img(chinese_image, latxt_image)
            cv2.imwrite('%s/%s' % (save_image_root, image_name), merge_image)
            print('ok!')


if __name__ == '__main__':
    chinese_label_root = '/home/yulongwu/d/ocr/data/baidu_data/baidu.lst'
    latxt_root = '/home/yulongwu/d/ocr/im2latex_my/data/im2latex_train.lst'
    latxt_label_root = '/home/yulongwu/d/ocr/im2latex_my/data/formulas_normal.lst'

    chinese_image_root = '/home/yulongwu/d/ocr/data/baidu_data/baidu'
    # latxt_image_root = '/home/yulongwu/d/ocr/im2latex_my/data/images_processed'
    latxt_image_root = '/home/yulongwu/d/data/ocr_data/formula_data/formula_images'

    save_image_root = '/home/yulongwu/d/ocr/data/chinese_formula_data/processed_image'
    save_label_root = '/home/yulongwu/d/ocr/data/chinese_formula_data/label.txt'
    right_chinese_label = []
    fo = open(save_label_root, 'w')
    chinese_fi = open(chinese_label_root).read().split('\n')
    for i in chinese_fi:
        label = i.split('\t')
        if len(label) < 2:
            continue
        if 10 < len(label[1]) < 30:
            right_chinese_label.append(label)

    latxt_label_lst = open(latxt_label_root).read().split('\n')
    lst = open(latxt_root, 'r').readlines()
    pool = Pool(15)
    task = pool.map(generate, lst)

