# -*- coding:utf-8 -*-
import numpy as np
import os
import cv2
from PIL import Image


class Segmentation(object):
    def __init__(self, width, height):
        # 切割后单个文字的大小
        self.width = width
        self.height = height

    # 考虑到实际图片会有噪点影响，min_value，才算有效值
    # 考虑到噪点，end - start大于min_range才算有效大小
    # 不仅可以提取行范围，也可以提取每行中的字符宽度范围
    @classmethod
    def extract_valid_ranges_from_array(cls, array, min_value=10, min_range=2):
        start_index = None
        end_index = None
        valid_ranges = []
        for index, value in enumerate(array):
            if value >= min_value and start_index is None:
                start_index = index
            elif value < min_value and start_index is not None:
                end_index = index
                if end_index - start_index >= min_range:
                    valid_ranges.append((start_index, end_index))
                start_index = None
                end_index = None
        return valid_ranges

    # 因为存在一些字体书写连在一起，所以用各行字符宽度的中位值对各行进行再切割，提高切割准确率
    @classmethod
    def median_split_column(cls, column_text_per_raw_range):
        new_text_per_raw_range = []
        widths = []
        for text_per_raw_range in column_text_per_raw_range:
            start = text_per_raw_range[0]
            end = text_per_raw_range[1]
            width = end - start + 1
            widths.append(width)
        widths = np.array(widths)
        # 求行字符宽度的中位值
        width_median = np.median(widths)
        for i, text_range in enumerate(column_text_per_raw_range):
            # 假设中位数为单个字符宽度，某个字符宽度n>=2时候，就认为此字符可切分为n个字符
            num_char = int(round(widths[i] / width_median, 0))
            if num_char > 1:
                char_w = float(widths[i] / num_char)
                for j in range(num_char):
                    start = text_range[0] + int(j * char_w)
                    end = text_range[0] + int((j + 1) * char_w)
                    new_text_per_raw_range.append((start, end))
            else:
                new_text_per_raw_range.append(text_range)
        return new_text_per_raw_range

    def export_img_list_per_column(self, raw, min_value=40, min_range=1):
        vertical_sum = np.sum(raw, axis=0)
        column_text_range = self.extract_valid_ranges_from_array(vertical_sum, min_value, min_range)
        text_per_raw_range = self.median_split_column(column_text_range)
        text_per_raw_img_set = []
        for text_range in text_per_raw_range:
            start = text_range[0]
            end = text_range[1]
            chines_img = raw[:, start:end]
            chines_img = cv2.resize(chines_img, (self.height, self.width))
            text_per_raw_img_set.append(chines_img)
        return text_per_raw_img_set

    def cutImgToText(self, predict_dir='./origin', img_name='toPredict.png'):
        path_predict_img = os.path.join(predict_dir, img_name)
        image_color = cv2.imread(path_predict_img)
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        # 二值化，黑底白字，黑色值为0方便后面处理
        adaptive_threshold = cv2.adaptiveThreshold(
            image_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
            cv2.THRESH_BINARY_INV, 11, 2)
        horizontal_sum = np.sum(adaptive_threshold, axis=1)
        raw_ranges = self.extract_valid_ranges_from_array(horizontal_sum)

        predict_text_img_set = []
        # copy_threshold = np.copy(adaptive_threshold)
        for i, raw_range in enumerate(raw_ranges):
            start = raw_range[0]
            end = raw_range[1]
            raw = adaptive_threshold[start:end, :]
            # pt1 = (0, start)
            # pt2 = (copy_threshold.shape[1], end)
            # cv2.rectangle(copy_threshold, pt1, pt2, 255)
            text_img_per_raw = self.export_img_list_per_column(raw, min_value=40, min_range=1)
            predict_text_img_set.append(text_img_per_raw)
        # cv2.imshow('line', copy_threshold)
        # cv2.waitKey(0)
        return predict_text_img_set


# test
# segment = Segmentation(64, 64)
# predict_text_img_set = segment.cutImgToText(predict_dir='./test/origin')
# for i in range(len(predict_text_img_set)):
#     raw_text = predict_text_img_set[i]
#     for j in range(len(raw_text)):
#         text = raw_text[j]
#         if j == 0 and i == 0:
#             tem_image = Image.fromarray(text).convert('L')
#             tem_image = tem_image.resize((64, 64), Image.ANTIALIAS)
#             # print (np.asarray(tem_image).shape)
#             tem_image = np.asarray(tem_image) / 255.0
#             # tem_image = tem_image.reshape([-1, 64, 64, 1])
#             # print (tem_image.shape)
#             tem_image = Image.fromarray(tem_image)
#             tem_image.show()
