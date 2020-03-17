import cv2
import numpy as np
import tensorflow as tf
import time
import os


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_tf_example(image, label, shape):
    example = tf.train.Example(features=tf.train.Features(feature={'img': bytes_feature(image),  # 这里面的一些关键词的拼写要特别注意，如果错了就会报错而且不容易检查出
                                                                   'label': int64_feature(label),
                                                                   'shape': int64_feature(shape)}))
    return example


def get_label_list(label):  # 只有大写字母加数字，每张图片只有一个字符，做成one_hot,26+10=36
    label_list = [0 for i in range(36)]
    idex = ord(label)
    if idex >= 48 and idex <= 58:
        idex = idex - 48
    else:
        idex = idex - 65 + 10
    label_list[idex] = 1
    return label_list


def create_dataset(img_file_path):
    start_time = time.time()
    all_names = os.listdir(img_file_path)
    writer = tf.python_io.TFRecordWriter('./dataset/train_tfr.tfrecords')

    for idex, name in enumerate(all_names):
        pic_path = os.path.join(os.getcwd(), os.path.join(img_file_path, name))
        img = cv2.imread(pic_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
        shape = img.shape
        shape = [shape[0], shape[1]]
        print(shape)
        img = np.array(img).flatten()
        print(img)
        img = img.tostring()
        label = name.split('_')[0]
        label = get_label_list(label)
        print(label)
        tf_example = get_tf_example(img, label, shape)
        writer.write(tf_example.SerializeToString())
    end_time = time.time()
    print('耗时：%g' % (end_time - start_time))


if __name__ == '__main__':
    img_file_path = './progress-train3-picture2/'
    create_dataset(img_file_path)
