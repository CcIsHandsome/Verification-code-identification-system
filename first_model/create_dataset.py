import tensorflow as tf
import numpy as np
import os
import cv2
import time


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_tf_example(image, label, shape):
    tf_example = tf.train.Example(features=tf.train.Features(feature={'image': bytes_feature(image),
                                                                      'label': int64_feature(label),
                                                                      'shape': int64_feature(shape)}))
    return tf_example


def get_label_list(label):
    label_list = [0 for j in range(13)]  # 序号0-9为0-9,序号10位*，序号11为+，序号12为-
    # 这里的label_list是矩阵，需要处理成一维
    if label == '=':
        label_list[10] = 1
    elif (label == '+'):
        label_list[11] = 1
    elif (label == '-'):
        label_list[12] = 1
    else:
        label_list[int(label)] = 1

    return label_list


def create_dataset(img_file_path):
    start_time = time.time()
    all_names = os.listdir(img_file_path)
    writer = tf.python_io.TFRecordWriter('./dataset/train_tfr.tfrecords')
    for idex, name in enumerate(all_names):
        pic_path = os.path.join(os.getcwd(), os.path.join(img_file_path, name))
        img = cv2.imread(pic_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
        shape = img.shape
        img = np.array(img).flatten()
        print(img)
        img = img.tostring()
        label = name.split('_')[0]
        label = get_label_list(label)  # 因为这里转化的时候value必须是列表（向量），而不能是矩阵或者是张量，如果是必须reshape成向量
        print(label, type(label))
        shape = [shape[0], shape[1]]
        print(shape)
        tf_example = get_tf_example(img, label, shape)
        writer.write(tf_example.SerializeToString())
        print('已完成：%d' % idex)
    writer.close()
    end_time = time.time()
    print('共耗时：%g' % (end_time - start_time))


if __name__ == '__main__':
    img_file_path = './progress-train1-picture2/'
    create_dataset(img_file_path)
