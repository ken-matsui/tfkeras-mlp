# coding: utf-8

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.python.keras import backend as K

from mlp.model import MLP

def load_data():
	dirn = './MNIST_data'
	data = mnist.read_data_sets(dirn, one_hot=True)

	print(data.train.num_examples, 'train samples')
	print(data.test.num_examples, 'test samples')
	print(data.validation.num_examples, 'validation samples (not used)')

	return data

if __name__ == '__main__':
	mnist = load_data()
	# tensorflow placeholders
	x = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, 10])
	# define TF graph
	mlp = MLP()
	y_pred = mlp(x)
	loss = tf.losses.softmax_cross_entropy(y_, y_pred)
	train_step = tf.train.AdagradOptimizer(0.05).minimize(loss)
	correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		print('Training...')
		for i in range(10001):
			batch_xs, batch_ys = mnist.train.next_batch(100)
			train_fd = {x: batch_xs, y_: batch_ys, K.learning_phase(): 1}
			train_step.run(feed_dict=train_fd)
			if i % 1000 == 0:
				batch_xv, batch_yv = mnist.test.next_batch(200)
				val_accuracy = accuracy.eval({x: batch_xv, y_: batch_yv, K.learning_phase(): 0})
				print('\tstep, accurary = %6d: %6.3f' % (i, val_accuracy))

		test_fd = {x: mnist.test.images, y_: mnist.test.labels,
					K.learning_phase(): 0}
		test_accuracy = accuracy.eval(feed_dict=test_fd)
		print('Test accuracy:', test_accuracy)

		saver = tf.train.Saver()
		saver.save(sess, './model')