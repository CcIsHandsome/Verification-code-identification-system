import tensorflow as tf
import forward
import backward
import cv2
import numpy as np
import os
import time

num_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '*', '+', '-']


def shadow_cut(img):  # 投影切割算法
    result = []
    h, w = img.shape[:2]
    count = [0 for i in range(w)]
    for x in range(0, w):
        for y in range(0, h):
            # print((x, y))
            if img[y][x] == 0:  # cv2中的图像第一个维度是高，而不是宽
                count[x] += 1
    start = 0
    end = 0
    while (end < w):
        if (count[end] == 0):
            start = end
            end = end + 1
        else:
            while (end < w and count[end] != 0):
                end = end + 1
            if (end < w and end - start > 25):  # 为了防止小的噪点影响切割
                result.append(img[0:80, start:end])
                start = end
    return result


def test(imgs_path, labels):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(dtype=tf.float32, shape=[None, 50 * 50])

        y_ = tf.placeholder(dtype=tf.float32, shape=[None, 13])

        y = forward.forward(x, None)

        saver = tf.train.Saver()

        count = 0
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.SAVE_MODEL_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            for idex, img_path in enumerate(imgs_path):
                label = labels[idex]
                ans = ''
                flag = False
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                imgs = shadow_cut(img)
                # print(imgs)
                for image in imgs:
                    new_image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_CUBIC)
                    new_image = np.reshape(new_image, (-1, 50 * 50))
                    predict = sess.run(y, feed_dict={x: new_image})
                    # print(predict)
                    predict = num_map[int(np.argmax(predict, 1))]
                    ans += predict
                if ans == label:
                    count += 1
                    flag = True
                print('预测值：%s,真实值：%s,是否正确:%s' % (ans, label, flag))
        print('正确率为%g' % (count / 1000))


if __name__ == '__main__':
    labels_path = "./mappings1-test.txt"
    imgs_path = './progress-train1-test-picture/'
    labels = []
    imgs = []
    f = open(labels_path)
    for i in f:
        labels.append(i.split(",")[1].split("=")[0].strip())
    f.close()
    names = os.listdir(imgs_path)
    for name in names:
        imgs.append(os.path.join(imgs_path, name))
    start = time.time()
    test(imgs, labels)
    end = time.time()
    print("总共用时：%g" % (end - start))
