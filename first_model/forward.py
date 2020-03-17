import tensorflow as tf


def get_w(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape=shape, dtype=tf.float32))
    if regularizer is not None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_b(shape):
    b = tf.Variable(tf.constant(0, shape=shape, dtype=tf.float32))
    return b


def forward(x, regularizer):
    x = tf.reshape(x, [-1, 50, 50, 1])

    w1 = get_w(shape=[3, 3, 1, 8], regularizer=regularizer)
    b1 = get_b(shape=[8])
    y1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
    y1 = tf.nn.relu(tf.nn.bias_add(y1, b1))
    y1 = tf.nn.max_pool(y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w2 = get_w(shape=[3, 3, 8, 16], regularizer=regularizer)
    b2 = get_b(shape=[16])
    y2 = tf.nn.conv2d(y1, w2, strides=[1, 1, 1, 1], padding='SAME')
    y2 = tf.nn.relu(tf.nn.bias_add(y2, b2))
    y2 = tf.nn.max_pool(y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w3 = get_w(shape=[3, 3, 16, 32], regularizer=regularizer)
    b3 = get_b(shape=[32])
    y3 = tf.nn.conv2d(y2, w3, strides=[1, 1, 1, 1], padding='SAME')
    y3 = tf.nn.relu(tf.nn.bias_add(y3, b3))
    y3 = tf.nn.max_pool(y3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w4 = get_w(shape=[3, 3, 32, 64], regularizer=regularizer)
    b4 = get_b(shape=[64])
    y4 = tf.nn.conv2d(y3, w4, strides=[1, 1, 1, 1], padding='SAME')
    y4 = tf.nn.relu(tf.nn.bias_add(y4, b4))
    y4 = tf.nn.max_pool(y4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    shaped = y4.get_shape().as_list()
    print(shaped)
    dim = 1
    for i in shaped[1:]:
        dim *= i
    reshaped = tf.reshape(y4, [-1, dim])

    w_f1 = get_w(shape=[dim, 500], regularizer=regularizer)
    b_f1 = get_b(shape=[500])
    y_f1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshaped, w_f1), b_f1))

    w_f2 = get_w(shape=[500, 13], regularizer=regularizer)
    b_f2 = get_b(shape=[13])
    y = tf.nn.bias_add(tf.matmul(y_f1, w_f2), b_f2)

    return y
