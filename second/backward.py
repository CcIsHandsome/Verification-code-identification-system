import tensorflow as tf
import forward
import os

REGULARIZER = None
LEARNING_RATE = 0.0001
STEPS = 10000
TFR_file_path = './dataset/tfrecord_train_dataset.tfrecords'
BATCH_SIZE = 64
MODEL_SAVE_PATH = './model2'
MODEL_NAME = 'class2_model'


def _parase_example(example_proto):
    features = {'img': tf.FixedLenFeature((), tf.string),
                'label': tf.FixedLenFeature([36], tf.int64),
                'shape': tf.FixedLenFeature([2], tf.int64)}
    parase_feature = tf.parse_single_example(example_proto, features=features)
    img = tf.decode_raw(parase_feature['img'], out_type=tf.uint8)
    label = parase_feature['label']
    return img, label


def backward():
    x = tf.placeholder(shape=[None, 50 * 50], dtype=tf.float32)
    y_ = tf.placeholder(shape=[None, 36], dtype=tf.float32)

    y = forward.forward(x, REGULARIZER)

    global_step = tf.Variable(0, trainable=False)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    right_ans = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(right_ans, tf.float32))

    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)

    saver = tf.train.Saver()

    dataset = tf.data.TFRecordDataset(TFR_file_path)
    dataset = dataset.map(_parase_example)
    dataset = dataset.shuffle(48000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = sess.run(fetches=next_batch)
            _, step, loss_val = sess.run([train_op, global_step, loss], feed_dict={x: xs, y_: ys})
            if i % 50 == 0:
                print('step:%d,loss:%g' % (step, loss_val))
            if i % 200 == 0:
                acc_val = sess.run(acc, feed_dict={x: xs, y_: ys})
                print('acc:%g' % acc_val)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), step)


if __name__ == '__main__':
    backward()
