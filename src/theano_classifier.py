import numpy as np
import time


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

	def train(self, 
		learning_rate, 
		n_epochs, 
		batch_size):
		'''
		Compiles functions for training, then trains.
		Learns by doing SGD on minibatches.
		Returns average training cost and average validation cost of final model if a validation set is provided.
		
		You should cross-validate over:
		- Learning rate
		- Number of epochs 
		- Batch size
		'''

		n_train_batches = self.classifier.train_set_x.get_value(borrow=True).shape[0] / batch_size
		n_valid_batches = self.classifier.valid_set_x.get_value(borrow=True).shape[0] / batch_size
		n_test_batches = self.classifier.test_set.get_value(borrow=True).shape[0]/batch_size
		# x = T.matrix('x')
		# y = T.ivector('y')
		# index = T.lscalar()

		# # compile validation function if you have a validation set
		# if valid_set_x is not None and valid_set_y is not None: 
		# 	validate_model = theano.function(inputs=[index], 
		# 		outputs=self.classifier.errors(y),
		# 		givens={
		# 			x: valid_set_x[index * batch_size:(index + 1) * batch_size],
		# 			y: valid_set_y[index * batch_size:(index + 1) * batch_size]
		# 		},
		# 		on_unused_input='ignore')

		# # Compute gradients
		# grads = T.grad(dropout_cost, self.classifier.params)

		# # SGD weights update
		# updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

		# # Compile training function that returns training cost, and updates model parameters. 
		# train_output, train_errors = self.classifier.dropout_nll(y), self.classifier.dropout_errors(y)
		# train_model = theano.function(inputs=[index], 
		# 	outputs=[train_output, train_errors],
		# 	updates=updates,
		# 	givens={
		# 		x: train_set_x[index * batch_size:(index + 1) * batch_size],
		# 		y: train_set_y[index * batch_size:(index + 1) * batch_size]
		# 	})

		# Then do the training and validation!
		training_error = 0
		validation_error = None
		epoch = 0

		best_val_error = 1
		best_predict = []
		best_val_predict = []

		start_time = time.clock()

		while (epoch < n_epochs):
			epoch = epoch + 1

			# train on all examples in minibatches 
			# if you're on the last epoch, track your average error
			for minibatch_index in xrange(n_train_batches):
				minibatch_avg_cost, minibatch_error = self.classifier.train_model(minibatch_index)
				if epoch == n_epochs:
					training_error += minibatch_error / n_train_batches
				# print 'Training error at epoch %d is %f' %(epoch, minibatch_avg_cost)

			# print 'Completed epoch %d. Code has run for %.2fm.' %(epoch, (time.clock() - start_time)/60)

			if self.classifier.valid_set_x is not None and epoch%10==0:
				validation_errors = [self.classifier.validate_model(i) for i in xrange(n_valid_batches)]
				val_error =  np.mean(validation_errors)/batch_size
				print 'Validation error at epoch %d is %f' % (epoch, np.mean(validation_errors)/batch_size)
				if val_error < best_val_error:
					best_val_error = val_error
					best_val_predict = [self.classifier.predict_model(i) for i in xrange(n_valid_batches)]
					best_predict = [self.classifier.predict_model(i) for i in xrange(n_test_batches)]

			self.classifier.decay_learning_rate()

		# get validation error
		if self.classifier.valid_set_x is not None:
			validation_errors = [self.classifier.validate_model(i) for i in xrange(n_valid_batches)]
			validation_error = np.mean(validation_errors)

		end_time = time.clock()

		print 'Finished training!\n The code ran for %.2fm.' % ((end_time - start_time) / 60.)

		return best_val_error, best_val_predict, best_predict

		
	# def predict(self, test_set):
	# 	'''
	# 	Should return array of predictions for test_set.
	# 	'''
	# 	# pad for batch size; 20,000/512 = 39.06 so we need 40*512
	# 	pad = np.zeros((20480,test_set.shape[1]))
	# 	pad[:20000] = test_set

	# 	predictions = [self.classifier.predict_model(i) for i in xrange(n_valid_batches)]

	# 	padded_pred = self.classifier.predict_model(pad)
	# 	pred = padded_pred[:20000]

	# 	return pred




