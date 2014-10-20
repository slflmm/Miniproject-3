import numpy as np

import theano
import theano.tensor as T

def dropout(layer_output, dropout_rate, rng):
	'''
	Applies dropout on the provided layer output.
	'''
	srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
	mask = srng.binomial(n=1, p=1-p, size=layer_output.shape)
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
