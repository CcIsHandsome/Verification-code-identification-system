import tensorflow as tf
import forward
import backward
import os
import cv2
import numpy as np

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
map_table = number + ALPHABET


def cut_img(im):
    im_cut_1 = im[0:80, 5:55]
    im_cut_2 = im[0:80, 50:100]
    im_cut_3 = im[0:80, 100:150]
    im_cut_4 = im[0:80, 150:200]
    im_cut = [im_cut_1, im_cut_2, im_cut_3, im_cut_4]
    return im_cut


def test(img_file_path, labels):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(shape=[None, 50 * 50], dtype=tf.float32)
        y = forward.forward(x, None)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            img_names = os.listdir(img_file_path)
            count = 0
            for idex, name in enumerate(img_names):
                img_path = os.path.join(img_file_path, name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cut_imgs = cut_img(img)
                ans = ''
                for im in cut_imgs:
                    im = cv2.resize(im, (50, 50), interpolation=cv2.INTER_CUBIC)
                    im = np.reshape(im, (1, 2500))
                    predict = sess.run(y, feed_dict={x: im})
                    predict = map_table[int(np.argmax(predict, 1))]
                    ans += predict
                if ans == labels[idex]:
                    count += 1
                print("预测值：%s,真实值%s" % (ans, labels[idex]))
            print("正确率为：%g" % (count / 1000))


if __name__ == '__main__':
    img_file_path = './progress-train3-test-picture/'
    labels_path = './mappings3-test.txt'
    label = []
    f = open(labels_path)
    for i in f:
        label.append(i.split(",")[1].split("=")[0].strip())
    f.close()
    test(img_file_path, label)
