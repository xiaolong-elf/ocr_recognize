import cv2
import glob
import os
import numpy as np

path = '/home/yulongwu/d/data/ocr_data/formula_data/formula_images'
image_lst = glob.glob(os.path.join(path, '*png'))
for i in image_lst:
    image = cv2.imread(i, 0)
    print(np.unique(image))
    exit()