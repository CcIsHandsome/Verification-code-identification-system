import tensorflow as tf
import os
import cv2
import numpy as np


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_tf_example(img, label, shape):
    example = tf.train.Example(features=tf.train.Features(feature={'img': bytes_feature(img),
                                                                   'label': int64_feature(label),
                                                                   'shape': int64_feature(shape)}))
    return example


def get_label_list(label):
    label_list = [0 for i in range(36)]
    idex = ord(label)
    if idex >= 48 and idex <= 57:
        idex = idex - 48
        label_list[idex] = 1
    elif idex >= 65 and idex <= 90:
        idex = idex - 65 + 10
        label_list[idex] = 1
    return label_list


def create_dataset(img_file_path):
    writer = tf.python_io.TFRecordWriter('./dataset/tfrecord_train_dataset.tfrecords')
    names = os.listdir(img_file_path)
    for idex, name in enumerate(names):
        img = cv2.imread(os.path.join('./', os.path.join(img_file_path, name)))  # cv2读取图片不能有中文，这里getcwd有中文
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = img.shape
        shape = list(shape)
        # print(shape)
        img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).flatten()
        img = img.tostring()
        # print(img)
        label = name.split('_')[0]
        label = get_label_list(label)
        # print(label)
        example = get_tf_example(img, label, shape)
        print("序号：%d" % (idex), label)
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    img_file_path = './progress-train2-picture2/'
    create_dataset(img_file_path)
