import numpy as np
import csv
import random

class CrossValidation(object):
    '''
    Iterator that returns 1/k of the data as validation data and
    the rest as training data, for every of the k pieces.
    '''
    def __init__(self, examples, outputs, k=10):
        assert len(examples) == len(outputs)

        self.examples = examples
        self.outputs = outputs
        self.k = len(outputs) // k
        self.i = 0

    def __iter__(self):
        return self

    def next(self):
        s, e = self.i * self.k, (self.i + 1) * self.k
        if s >= len(self.examples):
            raise StopIteration
        self.i += 1

        train_data = np.concatenate((self.examples[:s,:],self.examples[e:,:]))
        train_result = np.concatenate((self.outputs[:s],self.outputs[e:]))

        test_data = self.examples[s:e,:]
        test_result = self.outputs[s:e]

        return train_data, train_result, test_data, test_result

def get_confusion_matrix(actual, predicted):
    '''
    Returns the confusion matrix
    '''
    m = np.zeros((10,10))
    for a, b in zip(actual, predicted):
    	m[a,b] += 1

    class_totals = np.sum(m, axis=1)

    for i in xrange(10):
    	m[i] = m[i]*1. / class_totals[i]*1.

    return m

def write_test_output(output_data):
    '''
    Writes a set of predictions to file
    '''
    with open('test_output.csv', 'wb') as fp:
        writer = csv.writer(fp, quoting=csv.QUOTE_ALL)
        writer.writerow(['Id', 'Prediction'])  # write header
        for i, category in enumerate(output_data):
            writer.writerow((str(i+1), category))

def step(x):
	'''
	Just the step function.
	Works for numbers and arrays.
	'''
	return np.sign(x)

def sigmoid_one(x):
    return 1.0 / (1.0 + np.exp(-x))

sigmoid = np.vectorize(sigmoid_one)

def sigmoid_prime_one(x):
    return sigmoid(x)*(1 - sigmoid(x))

sigmoid_prime = np.vectorize(sigmoid_prime_one)


def randomize_params(alphas, n_layers, nodes_layer):
    alpha = random.choice(alphas)
    n_lay = random.choice(n_layers)
    layers = [48*48]
    for _ in xrange(n_lay):
        layers.append(random.choice(nodes_layer))
    layers.append(10)

    return alpha, layers