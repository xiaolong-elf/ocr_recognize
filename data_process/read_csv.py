import os
import cv2

show = False


def origin_image():
    path = '/home/yulongwu/d/BaiduNetdiskDownload/question_data/origin_image/FDQG-题目原图拍提图片对应表0410.csv'
    image_root = '/home/yulongwu/d/BaiduNetdiskDownload/question_data/origin_image/FDQG-Image/'

    fi = open(path, encoding='GBK')
    count = [0, 0]
    for line in fi:
        image_name = line.split(',')[1:]
        ori_im = image_root + "ori_image/" + image_name[0].strip()
        pic_im = image_root + "pic_image/" + image_name[1].strip()
        is_ori = os.path.isfile(ori_im)
        is_pic = os.path.isfile(pic_im)
        if is_ori:
            count[0] += 1
        if is_pic:
            count[1] += 1
        if show:
            if is_ori:
                print('is_ori', is_ori)
                ori_im = cv2.imread(ori_im)
                cv2.imshow('ori_im', ori_im)
            if is_pic:
                print('is_pic', is_pic)
                pic_im = cv2.imread(pic_im)
                cv2.imshow('pic_im', pic_im)
            if is_ori or is_pic:
                cv2.waitKey()
    print(count)


def picture_image():
    path = '/home/yulongwu/d/BaiduNetdiskDownload/question_data/picture_image/tt题目原图拍题图片对应表20180410.csv'
    ori_root = '/home/yulongwu/d/BaiduNetdiskDownload/question_data/origin_image/FDQG-Image/ori_image'
    pic_root = '/home/yulongwu/d/BaiduNetdiskDownload/question_data/picture_image'

    fi = open(path, encoding='GBK')
    count = [0, 0]
    for line in fi:
        image_name = line.split(',')[1:]
        ori_im = ori_root + image_name[0].strip()
        pic_im = pic_root + image_name[1].strip().split('search')[-1]
        is_ori = os.path.isfile(ori_im)
        is_pic = os.path.isfile(pic_im)
        if is_ori:
            count[0] += 1
        if is_pic:
            count[1] += 1
        if show:
            if is_ori:
                ori_im = cv2.imread(ori_im)
                cv2.imshow('ori_im', ori_im)
            if is_pic:
                pic_im = cv2.imread(pic_im)
                cv2.imshow('pic_im', pic_im)
            if is_ori or is_pic:
                cv2.waitKey()
    print(count)


if __name__ == '__main__':
    origin_image()
    # picture_image()
