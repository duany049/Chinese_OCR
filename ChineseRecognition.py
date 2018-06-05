# -*- coding:utf-8 -*-
import os
# 指定使用哪块gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import cv2
from tensorflow.python.ops import control_flow_ops
from Segmentaion import *
import sys

# python2
# reload(sys)
# sys.setdefaultencoding('utf-8')

# 使用复杂的模型

logger = logging.getLogger('Training chinese recognition')
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(sh)

# 添加命令行的可选参数
tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 1, 'Number of epoches')

tf.app.flags.DEFINE_integer('charset_size', 3782, "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_integer('max_steps', 20002, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 500, "the steps to save")
tf.app.flags.DEFINE_integer('batch_size', 128, 'Validation batch size')

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './dataset/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './dataset/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('mode', 'validation', 'Running mode. One of {"train", "test", "predict"}')
tf.app.flags.DEFINE_string('predict_dir', './predict', "the origin and resule to store predict")
tf.app.flags.DEFINE_string('to_predict_img', 'toPredict.png', "the origin img to be predicted")
tf.app.flags.DEFINE_string('predict_result', 'predict.result', "the result to store predict result")

tf.app.flags.DEFINE_float('learning_rate', 0.05, "the learning_rate of model")
# 使用adam优化器会根据参数动态约束学习率,不需要指定衰减率
# tf.app.flags.DEFINE_float('decay_rate', 0.95, "the decay rate of learning rate")
# tf.app.flags.DEFINE_float('decay_steps', 500, "the steps of decay rate")

# 配置每个gpu占用内存的比例
# gup_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#  此变量可从命令行中取出指定参数的参数值
FLAGS = tf.app.flags.FLAGS


def data_augmentation(images):
    # 镜像变换
    if FLAGS.random_flip_up_down:
        images = tf.image.random_flip_up_down(images)
    # 图像亮度变化
    if FLAGS.random_brightness:
        images = tf.image.random_brightness(images, max_delta=0.3)
    # 对比度变化
    if FLAGS.random_contrast:
        images = tf.image.random_contrast(images, 0.8, 1.2)
    return images


# 获取数据
class DataGain(object):
    def __init__(self, data_dir):
        # 比数据集中id最大的汉字的id大1，也就是所有字符的path都会小于max_id_path
        max_path = data_dir + ("%05d" % FLAGS.charset_size)
        # 遍历训练集所有图像的路径，存储在image_names内
        self.images_path = []
        for root_path, sub_folder_name_list, file_name_list in os.walk(data_dir):
            if root_path < max_path:
                self.images_path += [os.path.join(root_path, file_name) for file_name in file_name_list]
        random.shuffle(self.images_path)
        # 例如images_path为./dataset/train/00001/2.png，提取00001就是其label
        self.labels = [int(image_path[len(data_dir):].split(os.sep)[0]) for image_path in self.images_path]

    @property
    def size(self):
        return len(self.labels)

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        images_tensor = tf.convert_to_tensor(self.images_path, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        # 将image_list ,label_list做一个slice处理,每次只生成一对数据，循环num_epochs次，如果为None就不限制循环次数
        image_path, label = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)
        # image_path = input[0]
        # label = input[1]
        image_content = tf.read_file(image_path)
        image = tf.image.convert_image_dtype(tf.image.decode_png(image_content, channels=1), tf.float32)
        if aug:
            image = data_augmentation(image)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        image = tf.image.resize_images(image, new_size)
        # 根据输入的元素，生成文件名队列，batch是每次出队的数目,capacity是总数,需要和启动文件名填充线程配合执行
        print ("image: %s label: %s" % (image, label))
        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        return image_batch, label_batch


def build_graph(top_k):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')
    with tf.device('/cpu:0'):
        # network: conv2d->max_pool2d->conv2d->max_pool2d->conv2d->max_pool2d->conv2d->conv2d->
        # max_pool2d->fully_connected->fully_connected
        # 给slim.conv2d和slim.fully_connected准备了默认参数：batch_norm
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
            conv3_1 = slim.conv2d(images, 64, (3, 3), 1, padding='SAME', scope='conv3_1')
            conv3_2 = slim.conv2d(conv3_1, 64, (3, 3), 1, padding='SAME', scope='conv3_2')
            max_pool_1 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='SAME', scope='pool1')
            conv3_3 = slim.conv2d(max_pool_1, 128, (3, 3), 1, padding='SAME', scope='conv3_3')
            conv3_4 = slim.conv2d(conv3_3, 128, (3, 3), 1, padding='SAME', scope='conv3_4')
            max_pool_2 = slim.max_pool2d(conv3_4, [2, 2], [2, 2], padding='SAME', scope='pool2')
            conv3_5 = slim.conv2d(max_pool_2, 256, (3, 3), 1, padding="SAME", scope='conv3_5')
            conv3_6 = slim.conv2d(conv3_5, 256, (3, 3), 1, padding="SAME", scope='conv3_6')
            conv3_7 = slim.conv2d(conv3_6, 256, (3, 3), 1, padding="SAME", scope='conv3_7')
            conv3_8 = slim.conv2d(conv3_7, 256, (3, 3), 1, padding="SAME", scope='conv3_8')
            max_pool_3 = slim.max_pool2d(conv3_8, [2, 2], [2, 2], padding="SAME", scope='pool3')
            conv3_9 = slim.conv2d(max_pool_3, 512, (3, 3), 1, padding="SAME", scope='conv3_9')
            conv3_10 = slim.conv2d(conv3_9, 512, (3, 3), 1, padding="SAME", scope='conv3_10')
            conv3_11 = slim.conv2d(conv3_10, 512, (3, 3), 1, padding="SAME", scope='conv3_11')
            conv3_12 = slim.conv2d(conv3_11, 512, (3, 3), 1, padding="SAME", scope='conv3_12')
            # max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], [2, 2], padding='SAME', scope='pool4')
            max_pool_4 = slim.max_pool2d(conv3_12, [2, 2], [2, 2], padding='VALID', scope='pool4')

            # 将输入扁平化，但是保持batch_size
            flatten = slim.flatten(max_pool_4)
            # 注意随机失活是作用于数据，是把输入的数据按概率变为0，并且把未失活的数据变大让总期望不变.
            fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob=keep_prob), 1024, activation_fn=tf.nn.relu,
                                       scope='fc1')
            fc2 = slim.fully_connected(slim.dropout(fc1, keep_prob=keep_prob), 1024, activation_fn=tf.nn.relu,
                                       scope='fc2')
            # 最终输出要分类的类的数目，得到的值便是所有样本在各个类上面的最终得分
            logits = slim.fully_connected(slim.dropout(fc2, keep_prob=keep_prob), FLAGS.charset_size, activation_fn=None, scope='fc3')

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        accuracy_top_1 = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))
        # 获取bn层的信息
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        # learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.start_learning_rate, global_step=global_step,
        #                                            decay_rate=FLAGS.decay_rate, decay_steps=FLAGS.decay_steps,
        #                                            staircase=True)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        # 创造一个操作，用来计算梯度，并且返回loss
        train_operation = slim.learning.create_train_op(loss, optimizer=optimizer, global_step=global_step)
        probabilities = tf.nn.softmax(logits)

        # 绘制loss accuracy曲线
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy_top_1', accuracy_top_1)
        merged_summary_op = tf.summary.merge_all()
        # 返回top k 个预测结果及其概率；
        predicted_prob_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

        return {'images': images,
                'labels': labels,
                'keep_prob': keep_prob,
                'top_k': top_k,
                'global_step': global_step,
                'train_operation': train_operation,
                'loss': loss,
                'is_training': is_training,
                'accuracy_top_1': accuracy_top_1,
                'accuracy_top_k': accuracy_in_top_k,
                'merged_summary_op': merged_summary_op,
                'predicted_distribution': probabilities,
                'predicted_index_top_k': predicted_index_top_k,
                'predicted_prob_top_k': predicted_prob_top_k}


def train():
    print('==========================Begin training==========================')
    train_feeder = DataGain(data_dir='./dataset/train/')
    test_feeder = DataGain(data_dir='./dataset/test/')
    model_name = 'chinese-orc-model'
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # 获取batch
        train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)
        graph = build_graph(1)  # 训练时top k = 1
        # Saver作用是保存训练后模型的参数到checkpoint，以及从checkpoint中恢复变量
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # 多线程协调器
        coord = tf.train.Coordinator()
        # 启动执行文件名队列填充的线程，之后计算单元才可以把数据读出来，否则文件名队列为空的，计算单元就会处于一直等待状态，导致系统阻塞
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
        start_step = 0
        # 可以从某个step下的模型继续训练
        if FLAGS.restore:
            # 返回最后一个保存的checkpoint文件名path，如果不存在就返回None
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print ('restore from {0} checkpoint'.format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        logger.info(':::Training Start:::')
        try:
            i = 0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                # 执行多线程操作，进行入队出队得到batch
                train_batch_images, train_batch_labels = sess.run([train_images, train_labels])
                feed_dict = {graph['images']: train_batch_images,
                             graph['labels']: train_batch_labels,
                             graph['keep_prob']: 0.8,
                             graph['is_training']: True}
                _, loss_val, train_summary, step = sess.run(
                    [graph['train_operation'], graph['loss'], graph['merged_summary_op'], graph['global_step']],
                    feed_dict=feed_dict)
                train_writer.add_summary(train_summary, step)
                end_time = time.time()
                logger.info(
                    "the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
                if step > FLAGS.max_steps:
                    break
                if step % FLAGS.eval_steps == 1:
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['keep_prob']: 1.0,
                                 graph['is_training']: False}
                    accuracy_test, test_summary = sess.run([graph['accuracy_top_1'], graph['merged_summary_op']],
                                                           feed_dict=feed_dict)
                    if step > 300:
                        test_writer.add_summary(test_summary, step)
                    logger.info('===============Eval a batch=======================')
                    logger.info('the step {0} test accuracy: {1}'.format(step, accuracy_test))
                if step % FLAGS.save_steps == 1:
                    logger.info('Save the ckpt of step {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name),
                               global_step=graph['global_step'])

        except tf.errors.OutOfRangeError:
            logger.info('==================Train Finished================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name), global_step=graph['global_step'])
        finally:
            # 达到最大训练迭代数的时候请求关闭线程
            coord.request_stop()
        coord.join(threads)


def test():
    print ('==========================Begin Test==========================')
    testDataFeed = DataGain('./dataset/test/')
    final_predict_prob_top_k = []
    final_predict_index_top_k = []
    groundtruth = []
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # num_epochs为1这个很关键，test模式和train模式不一样，不需要重复读取数据
        testImagesBatch, testLablesBatch = testDataFeed.input_pipeline(FLAGS.batch_size, num_epochs=1)
        graph = build_graph(5)
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess=sess, save_path=ckpt)
            print ('restore form checkpoint {0}'.format(ckpt))
        logger.info('==========================Start Test==========================')
        acc_top_1, acc_top_k = 0.0, 0.0
        try:
            i = 0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                test_images_batch, test_labels_batch = sess.run([testImagesBatch, testLablesBatch])
                feed_dict = {graph['images']: test_images_batch,
                             graph['labels']: test_labels_batch,
                             graph['keep_prob']: 1.0,
                             graph['is_training']: False}
                batch_labels, probs_top_k, indices_top_k, \
                accuracy_top_1, accuracy_top_k = sess.run([graph['labels'],
                                                           graph['predicted_prob_top_k'],
                                                           graph['predicted_index_top_k'],
                                                           graph['accuracy_top_1'],
                                                           graph['accuracy_top_k']],
                                                          feed_dict=feed_dict)
                final_predict_prob_top_k += probs_top_k.tolist()
                final_predict_index_top_k += indices_top_k.tolist()
                groundtruth += batch_labels.tolist()
                acc_top_1 += accuracy_top_1
                acc_top_k += accuracy_top_k
                end_time = time.time()
                logger.info('the batch: {0} takes {1} seconds and acc_top_1: {2} acc_top_k: {3}'
                            .format(i, end_time - start_time, acc_top_1, acc_top_k))

        except tf.errors.OutOfRangeError:
            logger.info('==================Validation Finished================')
            # 看成acc_top_1 * (FLAGS.batch_size / testDataFeed.size)就好理解多了
            acc_top_1 = acc_top_1 * FLAGS.batch_size / testDataFeed.size
            acc_top_k = acc_top_k * FLAGS.batch_size / testDataFeed.size
            logger.info('top 1 accuracy {0} top k accuracy {1}'.format(acc_top_1, acc_top_k))
        finally:
            coord.request_stop()
        coord.join(threads)
    return {'prob': final_predict_prob_top_k, 'indices': final_predict_index_top_k, 'groundtruth': groundtruth}


# 获待预测文件夹内的所有图像路径
def get_file_path_list(path):
    file_path_list = []
    files_name = os.listdir(path)
    files_name.sort()
    for file_name in files_name:
        file_path = os.path.join(path, file_name)
        file_path_list.append(file_path)
    return file_path_list


# 图像二值化，需注意待预测的汉字是黑底白字还是白底黑字
def binary_pic(path_list):
    for path in path_list:
        imag = cv2.imread(path_list)
        # cvtColor用于颜色空间转换，此处参数为COLOR_BGR2GRAY，即转换为灰色图片
        gray_image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
        # 灰图二值化
        ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        single_name = path.split('t/')[1]
        print (single_name)
        cv2.imwrite(single_name, thresh)


# 获取汉字label映射表
def get_label_dict():
    labels_file = open('./chinese_labels', mode='rb')
    labels_dict = pickle.load(labels_file)
    labels_file.close()
    return labels_dict


def predict(text_set):
    print ('===================Begin Predict===================')
    image_set = []
    # 存储每行行末的位置
    change_line_index = []
    cur_raws_size = 0
    for raw_index in range(len(text_set)):
        raw_text = text_set[raw_index]
        raw_text_size = len(raw_text)
        cur_raws_size += raw_text_size
        change_line_index.append(cur_raws_size)
        for text_per_raw_index in range(raw_text_size):
            text = raw_text[text_per_raw_index]
            # 转变成Image类，并且转换成灰图
            tem_image = Image.fromarray(text).convert('L')
            tem_image = tem_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
            tem_image = np.asarray(tem_image) / 255.0
            # shape变成(1,64,64,1)
            tem_image = tem_image.reshape([-1, FLAGS.image_size, FLAGS.image_size, 1])
            image_set.append(tem_image)

    # allow_soft_placement 如果你指定的设备不存在，允许TF自动分配设备
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        logger.info('========Start predict============')
        graph = build_graph(3)
        saver = tf.train.Saver()
        # 自动获取最后一次保存的模型
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print ('restore form checkpoint {0}'.format(ckpt))

        predict_prob_list = []
        predict_index_list = []
        for item in image_set:
            tem_image = item
            predicted_index_top_k, predicted_prob_top_k = sess.run([graph['predicted_index_top_k'],
                                                                    graph['predicted_prob_top_k']],
                                                                   feed_dict={graph['images']: tem_image,
                                                                              graph['keep_prob']: 1.0,
                                                                              graph['is_training']: False})
            predict_index_list.append(predicted_index_top_k)
            predict_prob_list.append(predicted_prob_top_k)
    return predict_prob_list, predict_index_list, change_line_index


def main(_):
    print ('===================cur run mode is: {0}========================='.format(FLAGS.mode))
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'test':
        test_result = test()
        result_file = 'test.resule'
        print ('start write test resule save on file{0}'.format(result_file))
        with open(result_file, 'wb') as f:
            pickle.dump(test_result, f)
        print ('write test result edn')
    elif FLAGS.mode == 'predict':
        label_dict = get_label_dict()
        segmentation = Segmentation(FLAGS.image_size, FLAGS.image_size)
        predict_text_img_set = segmentation.cutImgToText(FLAGS.predict_dir, FLAGS.to_predict_img)
        # 将待预测的图片名字列表送入predict()进行预测，得到top3预测的结果及其index(在这儿即类别)
        final_predict_prob, final_predict_id, change_line_index = predict(predict_text_img_set)
        final_reco_text = []  # 存储最后识别出来的文字串
        # 给出top 3预测，candidate1是概率最高的预测结果
        for i in range(len(final_predict_prob)):
            candidate1 = final_predict_id[i][0][0]
            candidate2 = final_predict_id[i][0][1]
            candidate3 = final_predict_id[i][0][2]
            # 存储识别结果最高的字
            final_reco_text.append(label_dict[int(candidate1)])
            # logger.info('image {0} predict top_3 result is {1} {2} {3}, and the probility is {4}'
            #             .format(path_list[i], label_dict[int(candidate1)], label_dict[int(candidate2)],
            # label_dict[int(candidate3)], final_predict_prob[i]))
        print ('=====================OCR RESULT=======================\n')
        print (change_line_index)
        with open(os.path.join(FLAGS.predict_dir, FLAGS.predict_result), 'w') as f:
            cur_line_inde = 0
            # 打印出所有识别出来的结果（取top 1）
            for i in range(len(final_reco_text)):
                cur_change_line_index = change_line_index[cur_line_inde]
                if i == cur_change_line_index:
                    f.write('\n')
                    cur_line_inde += 1
                f.write(final_reco_text[i].encode('utf-8'))


if __name__ == '__main__':
    # 看源码可知，这行代码意思是解析参数，并且调用main函数
    tf.app.run()
