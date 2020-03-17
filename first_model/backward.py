import tensorflow as tf
import forward
import os
import time

BATCH_SIZE = 128
REUGLARIZER = 0.00005
LEARNING_RATE_BASE = 0.001
LEARING_RATE_STEPS = 9000 / BATCH_SIZE
LEARNING_RATE_DECAY = 0.9
STEPS = 10000
SAVE_MODEL_PATH = './model2/'
MODEL_NAME = 'class1_model'
TFR_file_path = './dataset/train_tfr.tfrecords'


def _parase_example(example_porot):
    features = {'image': tf.FixedLenFeature((), tf.string),
                'label': tf.FixedLenFeature([13], tf.int64),
                'shape': tf.FixedLenFeature([2], tf.int64)}
    parase_feuture = tf.parse_single_example(example_porot, features)
    img = tf.decode_raw(parase_feuture['image'], out_type=tf.uint8)  # 因为image转回来是string类型的，所以需要用tf转化成uint8
    label = parase_feuture['label']
    return img, label


def backward():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 50 * 50])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 13])

    y = forward.forward(x, regularizer=None)  # 暂时不用正则化

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARING_RATE_STEPS, LEARNING_RATE_DECAY,
                                               staircase=False)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))  # sigmoid和softmax有什么区别

    right_ans = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 这里不能用y_与y相等来计算正确率，因为y输出的是每个可能的概率，而y_只有01
    acc = tf.reduce_mean(tf.cast(right_ans, tf.float32))

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step)

    saver = tf.train.Saver()

    dataset = tf.data.TFRecordDataset(TFR_file_path)
    dataset = dataset.map(_parase_example)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(100000)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(SAVE_MODEL_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        try:
            while True:
                start = time.time()
                for i in range(STEPS):
                    imgs, labels = sess.run(fetches=next_batch)  # fetches这个参数就是需要计算的张量，
                    _, steps, loss_val = sess.run([train_op, global_step, loss],
                                                  feed_dict={x: imgs, y_: labels})
                    if (i % 10 == 0):
                        print("step:%d,loss:%g" % (steps, loss_val))
                    if (i % 50 == 0):
                        acc_val = sess.run(acc, feed_dict={x: imgs, y_: labels})
                        print("step:%d,acc:%g" % (steps, acc_val))
                        saver.save(sess, os.path.join(SAVE_MODEL_PATH, MODEL_NAME), global_step=steps)
                    if (i == 2000):
                        end = time.time()
                        print("2000 steps花费时间为：%g" % (end - start))
        except tf.errors.OutOfRangeError:
            print('example is end')


if __name__ == '__main__':
    backward()
