import tensorflow as tf
import os

USE_DATASET = True

if __name__ == "__main__":
    model_dir = './model'
    model_name = 'simple'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    graph = tf.Graph()

    if USE_DATASET:
        iterator = None
        x = None
        y = None
        with graph.as_default():
            x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x')
            x_dataset = tf.data.Dataset.from_tensor_slices(x)
            iterator = x_dataset.make_initializable_iterator()
            x_iter = iterator.get_next()
            a = tf.Variable(10, trainable=False, name='const_a')
            y = tf.multiply(x_iter, a, name='y')

        with tf.Session(graph=graph) as sess:
            operattion_list = graph.get_operations()
            sess.run(tf.global_variables_initializer())
            feed_dict = {x: [[1, 1, 1]]}
            sess.run(iterator.initializer, feed_dict=feed_dict)
            saver = tf.train.Saver(tf.global_variables())
            saver.save(sess, os.path.join(
                model_dir, model_name), global_step=0)
            result = sess.run(y)
            print(result)

    else:
        with graph.as_default():
            x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x')
            a = tf.Variable(10, trainable=False, name='const_a')
            y = tf.multiply(x, a, name='y')

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            saver.save(sess, os.path.join(
                model_dir, model_name), global_step=0)
            feed_dict = {x: [[1, 1, 1]]}
            result = sess.run(y, feed_dict=feed_dict)
            print(result)
