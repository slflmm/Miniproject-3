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

# TODO: Some activation functions...