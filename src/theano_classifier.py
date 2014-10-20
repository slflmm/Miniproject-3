import numpy as np

import theano
import theano.tensor as T

from theano_utils import * 

# ---------------------------------------------------------------------
# Training, validating, and predicting with a convnet.
# Based on examples in Theano Deep Learning Tutorials:
# 	http://www.deeplearning.net/tutorial/
# TODO: prettify?
# ---------------------------------------------------------------------

class Trainer(object):
	'''
	Training and validation of a neural net.
	Also gives predictions.
	'''

	def __init__(self, neural_network):
		self.classifier = neural_network

	def train(self, learning_rate, n_epochs, batch_size, train_set_x, train_set_y, valid_set_x=None, valid_set_y=None):
		'''
		Compiles functions for training, then trains.
		As of right now, sticks to basic SGD on minibatches.
		Returns average training cost and average validation cost of final model if a validation set is provided.
		
		TODO: Implement RPROP training algorithm or momentum?
		
		You should cross-validate over:
		- Learning learning_rate
		- Number of epochs 
		- Batch size 
		'''
		n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

		x = T.tensor3('x')
		y = T.ivector('y')
		index = T.lscalar()

		# compile validation function if you have a validation set
		if valid_set_x is not None: 
			validate_model = theano.function(inputs=[index], 
				outputs=classifier.errors(y),
				givens={
					x: valid_set_x[index * batch_size:(index + 1) * batch_size],
					y: valid_set_y[index * batch_size:(index + 1) * batch_size]
				})


		# Compute gradients
		grads = T.grad(dropout_cost, self.classifier.params)

		# SGD weights update
		updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

		# Compile training function that returns training cost, and updates model parameters. 
		train_output, train_errors = classifier.dropout_nll(y), classifier.dropout_errors(y)
		train_model = theano.function(inputs=[index], 
			outputs=[train_output, train_errors],
			updates=updates,
			givens={
				x: train_set_x[index * batch_size:(index + 1) * batch_size],
				y: train_set_y[index * batch_size:(index + 1) * batch_size]
			})

		# Then do the training and validation!
		training_error = 0
		validation_error = None
		epoch = 0

		start_time = time.clock()

		while (epoch < n_epochs):
			epoch = epoch + 1

			# train on all examples in minibatches 
			# if you're on the last epoch, track your average error
			for minibatch_index in xrange(n_train_batches):
				minibatch_avg_cost, minibatch_error = train_model(minibatch_index)
				if epoch == n_epochs:
					training_error += minibatch_error / n_train_batches

		# get validation error
		if valid_set_x is not None:
			validation_errors = [validate_model(i) for i in xrange(n_valid_batches)]
			validation_error = np.mean(validation_errors)

		end_time = time.clock()

		print 'Finished training!\n The code ran for %.2fm.' % ((end_time - start_time) / 60.)

		return training_error, validation_error

		
	def predict(self, test_set):
		'''
		Should return array of predictions for test_set.
		'''
		# compile function to return prediction
		predict_model = theano.function(inputs=[index],
			outputs=classifier.output_layer.y_pred,
			givens={
				x: test_set
			})

		# actually run the prediction and return it
		return predict_model(test_set)




