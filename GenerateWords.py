# -*- coding:utf-8 -*-

from PIL import Image, ImageDraw, ImageFont
import pickle
import argparse
from argparse import RawTextHelpFormatter
import os
import cv2
import sys
import random
import numpy as np
import shutil
import traceback
import copy


# 数据增强,增加鲁棒性
class DataAugmentation(object):
    def __init__(self, noise=True, dilate=True, erode=True):
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    # 增加噪点
    @classmethod
    def add_noise(cls, img, num_noise=20):
        for i in range(num_noise):
            x_tem = np.random.randint(0, img.shape[0])
            y_tem = np.random.randint(0, img.shape[1])
            img[x_tem, y_tem] = 255
        return img

    # 膨胀图像
    @classmethod
    def add_dilate(cls, img, shape=cv2.MORPH_RECT, ksize=(3, 3)):
        # 返回指定类型和尺寸的结构元素
        structElement = cv2.getStructuringElement(shape, ksize)
        # 使用指定的结构元素来腐蚀图片
        img = cv2.dilate(img, structElement)
        return img

    # 对图片增加腐蚀效应
    @classmethod
    def add_erode(cls, img, shape=cv2.MORPH_RECT, ksize=(3, 3)):
        structElement = cv2.getStructuringElement(shape, ksize)
        img = cv2.erode(img, structElement)
        return img

    def do(self, img_list=[]):
        # 两个list独立，不会互相影响
        operated_list = copy.deepcopy(img_list)
        for i in range(len(img_list)):
            img = img_list[i]
            if self.noise and random.random < 0.5:
                img = self.add_noise(img)
            if self.dilate and random.random < 0.5:
                img = self.add_dilate(img)
            if self.add_erode(img):
                img = self.add_erode(img)
            # 在原图的基础上再添加增强后的数据
            operated_list.append(img)
        return operated_list


# 等比例缩放
class CenterRatioResize(object):
    def __init__(self, processed_width, processed_height):
        self.width = processed_width
        self.height = processed_height

    # 根据processed_width，processed_height对img进行等比例缩放，让其在不改变原图比例的基础上最符合给定的大小
    def do(self, img):
        processed_width = self.width
        processed_height = self.height
        cur_height, cur_width = img.shape[:2]
        ratio_w = float(processed_width) / float(cur_width)
        ratio_h = float(processed_height) / float(cur_height)
        # 确保等比例缩放后，w和h都不会超过最大的尺寸
        ratio = min(ratio_h, ratio_w)
        new_size = (int(cur_width * ratio), int(cur_height * ratio))
        resized_img = cv2.resize(img, new_size)
        return resized_img


# 寻找图像最小包含矩形
class FindValidBox(object):
    def __init__(self):
        pass

    def do(self, img):
        height = img.shape[0]
        width = img.shape[1]
        w_sum = np.sum(img, axis=0)
        h_sum = np.sum(img, axis=1)
        left = 0
        top = 0
        right = width - 1
        low = height - 1
        # 从左往右扫描，遇到非零像素点就以此为字体的左边界
        for i in range(width):
            if w_sum[i] > 0:
                left = i
                break
        # 从右往左扫描，遇到非零像素点就以此为字体的右边界
        for i in range(width - 1, -1, -1):
            if w_sum[i] > 0:
                right = i
                break
        # 从上往下扫描，遇到非零像素点就以此为字体的上边界
        for i in range(height):
            if h_sum[i] > 0:
                top = i
                break
        # 从下往上扫描，遇到非零像素点就以此为字体的下边界
        for i in range(height - 1, -1, -1):
            if h_sum[i] > 0:
                low = i
                break
        return (left, top, right, low)


# 改变比例，并且把字体图片放到背景图中央
class RatioResizeAndFilledWithBg(object):
    def __init__(self, width, height, fill_bg=False, auto_avoid_fill_bg=True, margin=None):
        self.width = width
        self.height = height
        self.fill_bg = fill_bg
        self.auto_avoid_fill_bg = auto_avoid_fill_bg
        self.margin = margin

    # 如果长宽比小于1/3或者大于3，则需要填充背景
    @classmethod
    def is_need_fill_bg(cls, cv2_img):
        height, width = cv2_img.shape
        if height * 3 < width:
            return True
        if width * 3 < height:
            return True
        return False

    @classmethod
    def put_img_into_center(cls, bg_img, font_img):
        width_bg = bg_img.shape[1]
        height_bg = bg_img.shape[0]
        width_font = font_img.shape[1]
        height_font = font_img.shape[0]
        if width_font > width_bg:
            raise ValueError('the width of bg should larger than the width of font')
        if height_font > height_bg:
            raise ValueError('the height of bg should larger than the height of font')

        # 把font_img放到bg_img中央
        start_height = int((height_bg - height_font) / 2)
        start_width = int((width_bg - width_font) / 2)
        bg_img[start_height:start_height + height_font, start_width: start_width + width_font] = font_img
        return bg_img

    def do(self, cv2_img):
        width_minus_margin = self.width
        height_minus_margin = self.height
        if self.margin is not None:
            width_minus_margin = max(2, self.width - self.margin)
            height_minus_margin = max(2, self.height - self.margin)
        cur_height, cur_width = cv2_img.shape[:2]
        if len(cv2_img.shape) > 2:
            pix_dim = cv2_img.shape[2]
        else:
            pix_dim = None

        centerRatioResize = CenterRatioResize(width_minus_margin, height_minus_margin)
        # 把字体图片等比例缩放到指定尺寸
        resized_img = centerRatioResize.do(cv2_img)
        if self.auto_avoid_fill_bg:
            need_fill_bg = self.is_need_fill_bg(cv2_img)
            if not need_fill_bg:
                self.fill_bg = False
            else:
                self.fill_bg = True

        if not self.fill_bg:
            # 如果不需要填充背景，那么直接缩放到指定尺寸
            processed_img = cv2.resize(resized_img, (width_minus_margin, height_minus_margin))
        else:
            # 如果需要填充背景，就把图片放到指定尺寸的黑色背景中去
            if pix_dim is None:
                bg_img = np.zeros((height_minus_margin, width_minus_margin), dtype=np.uint8)
            else:
                bg_img = np.zeros((height_minus_margin, width_minus_margin, pix_dim), dtype=np.uint8)
            # 把字体图片放到背景图中央
            processed_img = self.put_img_into_center(bg_img, resized_img)

        if self.margin is not None:
            if pix_dim is not None:
                # 生成黑色的图片作为背景
                bg_img = np.zeros((self.height,
                                   self.width,
                                   pix_dim),
                                  np.uint8)
            else:
                bg_img = np.zeros((self.height,
                                   self.width),
                                  np.uint8)
            processed_img = self.put_img_into_center(bg_img, resized_img)
        return processed_img


# 检查字体是否有效
class CheckFontValid(object):
    def __init__(self, lang_chars, width=32, height=32):
        self.lang_chars = lang_chars
        self.width = width
        self.height = height

    def do(self, font_path):
        width = self.width
        height = self.height
        try:
            for i, char in enumerate(self.lang_chars):
                # 新建一个指定模式和尺寸的image
                img = Image.new("RGB", (width, height), "black")
                draw = ImageDraw.Draw(img)
                # 加载一个TrueType或者OpenType字体文件，并且为指定大小的字体创建一个字体对象
                font = ImageFont.truetype(font_path, int(width * 0.9), )
                # 用指定的字体画白色字
                draw.text((0, 0), char, (255, 255, 255),
                          font=font)
                # RGB模式的图片data格式如下:[(255, 255, 255), (255, 255, 255), (216, 216, 216), (8, 8, 8), (191, 191, 191),...]
                data = list(img.getdata())
                sum_val = 0
                for i_data in data:
                    sum_val += sum(i_data)
                if sum_val < 2:
                    # 不能写字即不可用
                    return False
        except:
            print("fail to load:%s" % font_path)
            traceback.print_exc(file=sys.stdout)
            # 加载失败即不可用
            return False
        return True


# 根据字体生成图片
class Font2Image(object):
    def __init__(self, width, height, margin):
        self.width = width
        self.height = height
        self.margin = margin

    def do(self, font_path, char, rotate=0):
        find_valid_box = FindValidBox()
        img = Image.new('RGB', (self.width, self.height), 'black')
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, int(self.width * 0.7))
        draw.text((0, 0), char, (255, 255, 255), font=font)
        if rotate != 0:
            img = img.rotate(rotate)
        # 把ImagingCore(img.getdata())对象转换成list，格式为[(R,G,B), (R,G,B), ...]
        data = list(img.getdata())
        sum_val = 0
        for i in data:
            sum_val += sum(i)
        if sum_val > 2:
            np_img = np.asarray(data, dtype='uint8')
            # 只保留获取每个RGB中的R值
            np_img = np_img[:, 0]
            np_img = np_img.reshape(self.height, self.width)
            cropped_box = find_valid_box.do(np_img)
            left, top, right, low = cropped_box
            np_img = np_img[top:low + 1, left:right + 1]
            ratioResizeFillBg = RatioResizeAndFilledWithBg(self.width, self.height, margin=self.margin)
            np_img = ratioResizeFillBg.do(np_img)
            return np_img
        else:
            print ('font failed 2 image')


# 获取chinese_labels里面的数据，返回一个字典,映射关系是(ID: 文字)
def get_chinese_dict():
    label_path = './chinese_labels'
    label_file = open(label_path, mode='rb')
    label_dict = pickle.load(label_file)
    label_file.close()
    return label_dict


# 解析参数
def parse_argument():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--out_dir', dest='out_dir', default=None, required=True, help='the dir of generate dateset')
    parser.add_argument('--font_dir', dest='font_dir', default=None, required=True,
                        help='font dir to to produce images')
    parser.add_argument('--test_ratio', dest='test_ratio', default=0.2, required=False, help='')
    parser.add_argument('--width', dest='width', default=32, required=True, help='width of image')
    parser.add_argument('--height', dest='height', default=32, required=True, help='height of image')
    parser.add_argument('--margin', dest='margin', default=0, required=False, help='', )
    parser.add_argument('--rotate', dest='rotate', default=0, required=False, help='max rotate degree 0-45')
    parser.add_argument('--rotate_step', dest='rotate_step', default=0, required=False,
                        help='rotate step for the rotate angle')
    parser.add_argument('--need_aug', dest='need_aug', default=False, required=False, help='need data augmentation',
                        action='store_true')
    args = vars(parser.parse_args())
    return args


# 利用上面定义好的方法和类，生成自己想要的各种样式的汉子图片
if __name__ == '__main__':
    args = parse_argument()
    # os.path.expanduser把path中包含的"~"和"~user"转换成用户目录
    out_dir = os.path.expanduser(args['out_dir'])
    font_dir = os.path.expanduser(args['font_dir'])
    test_ratio = float(args['test_ratio'])
    width = int(args['width'])
    height = int(args['height'])
    margin = int(args['margin'])
    rotate = int(args['rotate'])
    need_aug = args['need_aug']
    rotate_step = int(args['rotate_step'])
    train_image_dir_name = "train"
    test_image_dir_name = "test"

    # 将dataset分为train和test两个文件夹分别存储
    train_image_dir = os.path.join(out_dir, train_image_dir_name)
    test_image_dir = os.path.join(out_dir, test_image_dir_name)
    if os.path.isdir(train_image_dir):
        # 递归删除train_images_dir目录
        shutil.rmtree(train_image_dir)
    os.makedirs(train_image_dir)
    if os.path.isdir(test_image_dir):
        shutil.rmtree(test_image_dir)
    os.makedirs(test_image_dir)

    label_dict = get_chinese_dict()
    char_list = []
    id_list = []
    for (id, char) in label_dict.items():
        print (id, char)
        char_list.append(char)
        id_list.append(id)
    # 合并成新的映射关系表：（汉字：ID）  zip把两个list转成对应序列元素组成元组元素的list,再经过dict得到的结果相当于之前字典key和value交换
    lang_chars = dict(zip(char_list, id_list))
    font_check = CheckFontValid(lang_chars)
    if rotate < 0:
        roate = - rotate
    if rotate > 0 and rotate <= 45:
        all_rotate_angles = []
        for i in range(0, rotate + 1, rotate_step):
            all_rotate_angles.append(i)
        for i in range(-rotate, 0, rotate_step):
            all_rotate_angles.append(i)

    # 通过验证的字体集合
    verified_font_paths = []
    ## 迭代所有字体进行验证
    for font_name in os.listdir(font_dir):
        font_path = os.path.join(font_dir, font_name)
        if (font_check.do(font_path)):
            verified_font_paths.append(font_path)

    font2image = Font2Image(width, height, margin)
    # 生成字体图片
    for char, id in lang_chars.items():
        # 同一次循环生成的所有图片都是同一个id(汉字),所以可以放在同一个文件夹下
        image_list = []
        print (char, id)
        for font_index, verified_font_path in enumerate(verified_font_paths):
            if rotate == 0:
                img = font2image.do(verified_font_path, char)
                image_list.append(img)
            else:
                for angle in all_rotate_angles:
                    img = font2image.do(verified_font_path, char, rotate=angle)
                    image_list.append(img)

        if need_aug:
            dataAug = DataAugmentation(True, True, True)
            image_list = dataAug.do(image_list)

        test_num = len(image_list) * test_ratio
        random.shuffle(image_list)  # 打乱顺序
        count = 0
        # 从当前汉字的图片列表中抽取test_num作为测试数据
        for i in range(len(image_list)):
            img = image_list[i]
            if count < test_num:
                char_dir = os.path.join(test_image_dir, "%0.5d" % id)
            else:
                char_dir = os.path.join(train_image_dir, "%0.5d" % id)
            if not os.path.isdir(char_dir):
                os.makedirs(char_dir)

            path_image = os.path.join(char_dir, "%d.png" % count)
            cv2.imwrite(path_image, img)
            count += 1
