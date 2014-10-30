import numpy as np

import theano
import theano.tensor as T

def dropout(layer_output, dropout_rate, rng):
	'''
	Applies dropout on the provided layer output.
	'''
	srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
	mask = srng.binomial(n=1, p=1-dropout_rate, size=layer_output.shape)
	return layer_output * T.cast(mask, theano.config.floatX)

def rectified_linear(x):
	'''
	Rectified linear unit activation. Seems to work very well in deep nets.
	'''
	return T.maximum(0.0, x)

def sigmoid(x):
	'''
	Canonical sigmoid activation.
	'''
	return T.nnet.sigmoid(x)

def tanh(x):
	'''
	Activation. A lot like sigmoid.
	'''
	return T.tanh(x)

def shared_dataset(train_x, train_y, valid_x=None, valid_y=None):
	# print np.asarray(train_x,dtype=theano.config.floatX).reshape(10000,1,48,48).shape
	# temp = np.asarray(train_x,dtype=theano.config.floatX).reshape(len(train_x),1,48,48)
	temp = np.asarray(train_x,dtype=theano.config.floatX)
	shared_train_x = theano.shared(temp, borrow=True)
	temp = np.asarray(train_y,dtype=theano.config.floatX)
	shared_train_y = T.cast(theano.shared(temp, borrow=True), 'int32')

	if valid_x is not None:
		temp = np.asarray(valid_x, dtype=theano.config.floatX)#.reshape(len(valid_x),1,48,48)
		shared_valid_x = theano.shared(temp, borrow=True)
		shared_valid_y = T.cast(theano.shared(np.asarray(valid_y, dtype=theano.config.floatX), borrow=True), 'int32')

	return shared_train_x, shared_train_y, shared_valid_x, shared_valid_y

def get_final_image_size(filter_shapes, image_shapes, pool):
	if pool is not None:
		current = (image_shapes[-1][2] - filter_shapes[-1][2] + 1)/2
	else:
		current = image_shapes[-1][2] - filter_shapes[-1][2] + 1
	return current*current

