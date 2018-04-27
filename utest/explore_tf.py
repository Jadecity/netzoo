import tensorflow as tf

if __name__ == '__main__':
    a = tf.placeholder(tf.int8, [None])
    b = tf.placeholder(tf.int8, [None, None, 3])
    c = tf.reshape(a, [tf.div(tf.shape(a)[0], 3), 3])

    with tf.Session() as ss:
        a, b, c = ss.run((a, b, c), feed_dict={a: [1,2,3,4,5,6], b:[[[1,2,3],
                                                             [4,5,6],
                                                             [7,8,9]]
                                                            ]})

        print(a, b, c)
