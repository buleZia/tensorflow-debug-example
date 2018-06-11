"""
    https://blog.csdn.net/aliceyangxi1987/article/details/71079387
    https://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
LOGDIR = "/home/qiuju/lanweitao/model/Linear"
DATADIR = '/home/qiuju/lanweitao/debug/MNIST_data'
metadata = os.path.join(LOGDIR, 'metadata.tsv')
spritedata = os.path.join(LOGDIR, 'sprite.png')
EMBEDDING_COUNT = 10000

def linear_fn(learning_rate, writer):
    # load data
    mnist = tf.contrib.learn.datasets.mnist.read_data_sets(DATADIR, one_hot=True)

    # regression model
    with tf.name_scope('regression'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        b = tf.Variable(tf.zeros([10]), name='b')
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.matmul(x, W) + b

    # loss
    y_ = tf.placeholder(tf.float32, [None, 10])
    with tf.name_scope('train'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        tf.summary.scalar('cross_entropy', cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        # tf.summary.scalar('train_step', train_step)

    embedding = tf.Variable(tf.zeros([10000, 10]), name="test_embedding")
    assigment = embedding.assign(y)

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = spritedata
    embedding_config.metadata_path = metadata
    # embedding_config.metadata_path = 'metadata.tsv'
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    # eval
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuarcy'):
        accuarcy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuarcy', accuarcy)

    merge_summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    writer.add_graph(sess.graph)
    create_metadata(EMBEDDING_COUNT)

    # train
    for epoch in range(2001):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        s = sess.run(merge_summary, feed_dict={x: batch_xs, y_: batch_ys})
        writer.add_summary(s, epoch)

        if epoch % 500 == 0:
            train_accuarcy = sess.run(accuarcy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            sess.run(assigment, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), epoch)
            print('Accuarcy at epoch {}: {:.4f}'.format(epoch, train_accuarcy))


def create_sprite_image(images):
    """
	return a sprite image consisting of images passed as argument.
	images should be count x width x height
	"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:  # this_filter??
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img
    return spriteimage


def vector_to_matrix_mnist(mnist_digits):
    return np.reshape(mnist_digits, (-1, 28, 28))


def invert_grayscale(mnist_digits):
    return 1 - mnist_digits


def create_metadata(EMBEDDING_COUNT):
    # create metadata
    mnist = tf.contrib.learn.datasets.mnist.read_data_sets(DATADIR, one_hot=False)
    batch_xs, batch_ys = mnist.train.next_batch(EMBEDDING_COUNT)
    to_visualise = batch_xs
    to_visualise = vector_to_matrix_mnist(to_visualise)
    to_visualise = invert_grayscale(to_visualise)

    sprite_image = create_sprite_image(to_visualise)

    plt.imsave(spritedata, sprite_image, cmap='gray')
    plt.imshow(sprite_image, cmap='gray')

    with open(metadata, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(batch_ys):
            f.write("%d\t%d\n" % (index, label))


def main():
    for learning_rate in [0.5]:
        writer = tf.summary.FileWriter(os.path.join(LOGDIR))
        # writer = tf.summary.FileWriter(os.path.join(LOGDIR + hparam_str))

        # model_fn(learning_rate, use_two_conv, use_two_fc, writer)
        linear_fn(learning_rate, writer)


if __name__ == "__main__":
    if os.path.exists(LOGDIR):
        shutil.rmtree(LOGDIR)
    main()