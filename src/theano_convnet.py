import numpy as np

import theano
import theano.tensor as T
from theano.nnet import conv 

from theano_utils import * 

# ---------------------------------------------------------------------
# This file contains a modular implementation of a convnet using Theano
# Based on examples in Theano Deep Learning Tutorials:
# 	http://www.deeplearning.net/tutorial/
# Implements:
#   - Dropout in hidden layers
# TODO: Regularization in convolution layers
# TODO: Finish the convnet class
# ---------------------------------------------------------------------

class OutputLayer(object):
	'''
	Basically a multiclass logistic regression classifier.
	'''
	def __init__(self, input, n_in, n_out):

		# initialize weights to 0
		self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W')
		self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b')

		# probability of being y given example x: WX + b
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		# prediction is class with highest probability
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)

		# keep track of all weight parameters
		self.params = [self.W, self.b]

	def negative_log_likelihood(self,y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

	def errors(self,y):
		return T.sum(T.neq(self.y_pred, y))


class HiddenLayer(object):
	'''
	A simple hidden layer with choice of activation and dropout.
	'''
	def __init__(self, rng, input, n_in, n_out, activation, dropout_rate=0):
		self.input = input
		self.activation = activation 

		self.W = theano.shared(
			value=np.asarray(0.01*rng.standard_normal(size=(n_in,n_out)), dtype=theano.config.floatX), 
			name='W')
		self.b = theano.shared(
			value=np.zeros((n_out,), dtype=theano.config.floatX),
			name='b')

		out = activation(T.dot(input, self.W) + self.b)
		self.output = (out if dropout_rate == 0 else dropout(out, dropout_rate, rng))

		self.params = [self.W, self.b]


class ConvLayer(object):
	'''
	A convolutional layer using Theano's built-in 2-D convolution and subsampling.
	'''
	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):
		assert image_shape[1] == filter_shape[1]
		self.input = input

		W_bound = np.sqrt(6. / (np.prod(filter_shape[1:]) + filter_shape[0] + np.prod(filter_shape[2:]) / np.prod(poolsize)))
		self.W = theano.shared(
			value=np.asarray(
				rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
				dtype=theano.config.floatX),
			borrow=True)

		self.b = theano.shared(
			value=np.zeros((filter_shape[0],), dtype=theano.config.floatX),
			borrow=True)

		conv_out = conv.conv2d(
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			image_shape=image_shape)

		pooled_out = downsample.max_pool_2d(
			input=conv_out,
			ds=poolsize,
			ignore_border=True)

		self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))

		self.params = [self.W, self.b]


class ConvNet(object):
	'''
	A deep convolutional neural network with dropout in hidden layers.
	You can adjust the number and size of each layer, 
	as well as the presence and value of dropout rate.
	'''
	pass 

