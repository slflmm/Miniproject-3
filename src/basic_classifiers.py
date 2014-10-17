import numpy as np
import utils


class BasicClassifier(object):
	'''
	Template for a classifier.
	'''
	def __init__(self):
		pass

	def train(self, examples, outputs):
		pass

	def predict(self, example):
		pass 

class Perceptron(BasicClassifier):
	'''
	A simple multiclass perceptron.
	'''
	def __init__(self, alpha=0.1, activation=lambda x: utils.step(x), n_iter=10):
		self.alpha = alpha
		self.activation = activation
		self.n_iter = n_iter

	def train(self, examples, outputs):
		'''
		Trains a perceptron.
		Expects arrays of examples and outputs (as one-hot vectors).
		'''
		n_examples = examples.shape[0]
		n_features = examples.shape[1]
		n_outputs = outputs.shape[1]

		self.w = np.random.uniform(-0.5,0.5,(n_features + 1, n_outputs))
		bias_examples = np.ones((n_examples, n_features + 1))
		bias_examples[:,1:] = examples

		# until max iterations are reached or data is completely classified...
		for _ in range(self.n_iter):
			iter_error = 0
			for x, y in zip(bias_examples, outputs):
				# get output and error
				out = self.activation(np.dot(self.w.T, x))
				error = y - out
				# if you classified wrong:
				if np.sum(error) != 0:
					iter_error += 1
					# modify the relevant weights
					self.w[:,np.argmax(out)] -= self.alpha * x
					self.w[:,np.argmax(y)] += self.alpha * x
			if (iter_error < 1) : break

	def predict(self, example):
		'''
		Predicts the output for example.
		'''
		n_features = example.size
		bias_example = np.ones(n_features + 1)
		bias_example[1:] = example
		return np.argmax(self.activation(np.dot(self.w.T, bias_example)))
