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
LOGDIR = "/home/qiuju/lanweitao/model/Convolution"
DATADIR = '/home/qiuju/lanweitao/debug/MNIST_data'
metadata = os.path.join(LOGDIR, 'metadata.tsv')
spritedata = os.path.join(LOGDIR, 'sprite.png')
EMBEDDING_COUNT = 784


# Define a simple convolutional layer
def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1),
                        name="w")  # TODO what this init?# same with image's channel, kernel counts
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
        # return act
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# And a fully connected layer
def fc_layer(input, channels_in, channels_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="w")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="b")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
        return act


def model_fn(learning_rate, use_two_conv, use_two_fc, writer):
    tf.reset_default_graph()
    # impot data
    # mnist = input_data.read_data_sets("/home/lanweitao/tensorboardstu/2data", one_hot=True)
    mnist = tf.contrib.learn.datasets.mnist.read_data_sets(DATADIR, one_hot=True)

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Create the network
    if use_two_conv:
        conv1 = conv_layer(x_image, 1, 32, "conv1")
        conv_output = conv_layer(conv1, 32, 64, "conv2")  # 7,7,64
    else:
        conv1 = conv_layer(x_image, 1, 64, "conv")  # TODO show problem
        conv_output = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                     padding="SAME")  # without this layer, logits shape is [400]

    flattened = tf.reshape(conv_output, [-1, 7 * 7 * 64])

    if use_two_fc:
        fc1 = fc_layer(flattened, 7 * 7 * 64, 40, "fc1")
        logits = fc_layer(fc1, 40, 10, "fc2")
        embedding_size = 40
        embedding_input = fc1
    else:
        embedding_input = flattened  # last secend layer
        embedding_size = 7 * 7 * 64
        logits = fc_layer(flattened, 7 * 7 * 64, 10, "fc")

    # Compute cross entropy as our loss function
    with tf.name_scope("xent"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y), name="xent")
        tf.summary.scalar('cross_entropy', xent)

    # Use an AdamOptimizer to train the network
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    # Compute the accuracy
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    tf.summary.image('input', x_image, 3)

    embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    assigment = embedding.assign(embedding_input)

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = spritedata
    embedding_config.metadata_path = metadata
    # embedding_config.metadata_path = 'metadata.tsv'
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    # Initialize all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # tsv file
    # if not os.path.exists(metadata):
    #     os.mknod(metadata)

    writer.add_graph(sess.graph)
    create_metadata(EMBEDDING_COUNT)

    # Train for 2000 steps
    for i in range(2001):
        batch = mnist.train.next_batch(100)

        if i % 5 == 0:
            s = sess.run(merged_summary, feed_dict={x: batch[0], y: batch[1]})
            writer.add_summary(s, i)

        # Occasinally report accuracy
        if i % 500 == 0:
            # [train_accuracy] = sess.run([accuracy], feed_dict={x: batch[0], y: batch[1]})
            train_accuracy = accuracy.eval(feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
            sess.run(assigment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})  # show 1024 data
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
            print("step %d, training accuracy %g" % (i, train_accuracy))

        # run the training step
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})


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
            train_accuarcy = sess.run(accuarcy, feed_dict={x: mnist.test.images[:10000], y_: mnist.test.labels[:10000]})
            sess.run(assigment, feed_dict={x: mnist.test.images[:10000], y_: mnist.test.labels[:10000]})
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), epoch)
            print('Accuarcy at epoch {}: {:.4f}'.format(epoch, train_accuarcy))


def make_hparam_string(learning_rate, use_two_conv, use_two_fc):
    conv_param = 'conv2' if use_two_conv else "conv1"
    fc_param = 'fc2' if use_two_fc else "fc1"
    return "lr_%.0E_%s_%s" % (learning_rate, conv_param, fc_param)


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
    for learning_rate in [1E-3]:
        # Try a model with fewer layers
        for use_two_fc in [True]:
            for use_two_conv in [True]:
                hparam_str = make_hparam_string(learning_rate, use_two_conv, use_two_fc)
                print("now training with %s" % (hparam_str))
                writer = tf.summary.FileWriter(os.path.join(LOGDIR))
                # writer = tf.summary.FileWriter(os.path.join(LOGDIR + hparam_str))

                model_fn(learning_rate, use_two_conv, use_two_fc, writer)
                # linear_fn(learning_rate, writer)


if __name__ == "__main__":
    if os.path.exists(LOGDIR):
        shutil.rmtree(LOGDIR)
    main()