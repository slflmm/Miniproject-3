import numpy as np

import theano
import theano.tensor as T
from theano.nnet import signal 

from theano_utils import * 

# ---------------------------------------------------------------------
# Implements convolutional network architecture using Theano.
# Based on examples in Theano Deep Learning Tutorials:
# 	http://www.deeplearning.net/tutorial/
# Implements dropout in hidden layers (convolutions don't need it).
# ---------------------------------------------------------------------

class OutputLayer(object):
	'''
	Basically a multiclass logistic regression classifier.
	'''
	def __init__(self, input, n_in, n_out, W=None, b=None):

		if W is None:
			W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W')
		if b is None:
			b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b')

		self.W = W
		self.b = b

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
	def __init__(self, rng, input, n_in, n_out, activation, dropout_rate=0, W=None, b=None):
		self.input = input
		self.activation = activation 

		if W is None:
			W = theano.shared(
				value=np.asarray(0.01*rng.standard_normal(size=(n_in,n_out)), dtype=theano.config.floatX), 
				name='W')
		if b is None:
			b = theano.shared(
				value=np.zeros((n_out,), dtype=theano.config.floatX),
				name='b')

		self.W = W 
		self.b = b

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
	You can adjust:
	- The number of layers and size of each layer 
	- The presence and value of dropout rate
	- Different activations in layers
	'''
	def __init__(self, rng, input, conv_filter_shapes, image_shapes, poolsizes, hidden_layer_sizes, n_outputs, dropout_rates, activations):

		print 'Building the convnet'

		assert len(hidden_layer_sizes) - 1 == len(dropout_rates)
		
		self.layers = []
		self.dropout_layers = []

		# don't do dropout on input going into convolution...
		next_layer_input = input 

		# add the convolution layers
		conv_count = 0
		for filter_shape, image_shape in zip(conv_filter_shapes, image_shapes):

			# keep a set for dropout training...
			next_dropout_layer = ConvLayer(rng,
				input=next__layer_input,
				filter_shape=filter_shape,
				image_shape=image_shape,
				poolsize=poolsizes[conv_count])
			self.dropout_layers.append(next_dropout_layer)
			next_dropout_layer_input = next_dropout_layer.output 

			# for convolutions, non-dropout is the same layer
			next_hidden_layer = next_dropout_layer
			self.layers.append(next_hidden_layer)
			next_layer_input = next_hidden_layer.output

			conv_count += 1

		# prepare dropout on layer input
		next_dropout_layer_input = dropout(next_layer_input, dropout_rates[0], rng)

		# add the hidden layers
		hidden_count = 0
		for n_in, n_out in zip(hidden_layer_sizes, hidden_layer_sizes[1]):

			# the dropout layers for training...
			next_dropout_layer = HiddenLayer(rng, 
				input=next_dropout_layer_input, 
				n_in=n_in, n_out=n_out, 
				activation=activations[hidden_count], 
				dropout_rate=dropout_rates[hidden_count + 1])
			self.dropout_layers.append(next_dropout_layer)
			next_dropout_layer_input = next_dropout_layer.output

			# corresponding regular layers
			next_hidden_layer = HiddenLayer(rng, 
				input=next_layer_input, 
				n_in=n_in, n_out=n_out, 
				activation=activations[hidden_count], 
				W=next_dropout_layer.W*(1 - dropout_rates[hidden_count]), 
				b=next_dropout_layer.b)
			self.layers.append(next_hidden_layer)
			next_layer_input = next_hidden_layer.output

			hidden_count += 1

		n_in = next_hidden_layer.W.size[1]
		n_out = n_outputs 
		dropout_output = OutputLayer(input=next_dropout_layer_input, n_in=n_in, n_out=n_out)
		self.dropout_layers.append(dropout_output)

		output_layer = OutputLayer(input=next_dropout_layer_input, n_in=n_in, n_out=n_out, W=dropout_output.W*(1 - dropout_rates[-1]), b=dropout_output.b)
		self.layers.append(output_layer)

		# errors for training (dropout) and validation (no dropout)...
		self.dropout_nll = self.dropout_layers[-1].negative_log_likelihood
		self.dropout_errors = self.dropout_layers[-1].errors

		self.nll = self.layers[-1].negative_log_likelihood
		self.errors = self.layers[-1].errors 

		# dropout parameters will be used to calculate gradients during training
		self.params = [ param for layer in self.dropout_layers for param in layer.params ]






