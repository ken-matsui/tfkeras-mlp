# coding: utf-8

import tensorflow as tf
from tensorflow.python import keras

class MLP(object):
	"""
	docstring for MLP
	"""
	def __init__(self):
		super(MLP, self).__init__()
		with tf.name_scope('mlp_model'):
			self.l1 = keras.layers.Dense(units=512, activation='relu')
			self.l2 = keras.layers.Dense(units=512, activation='relu')
			self.l3 = keras.layers.Dense(units=10, activation='softmax')
			self.d1 = keras.layers.Dropout(0.2)
			self.d2 = keras.layers.Dropout(0.2)

	def __call__(self, x):
		h = self.d1(self.l1(x))
		h = self.d2(self.l2(h))
		y = self.l3(h)
		return y
