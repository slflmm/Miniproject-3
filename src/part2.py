from basic_classifiers import *
from features import *
from sklearn import preprocessing
from sklearn import decomposition
from utils import *

# # ------------------------
# # Loading training set
# # ------------------------
print "Loading train output..."
categories = loadnp("/Users/stephanielaflamme/Desktop/data_and_scripts/train_outputs.npy")

print "Loading train input..."
examples = loadnp("/Users/stephanielaflamme/Desktop/Numpy sets/train_inputs_standardized.npy")

# ----------------
# Loading test set
# ----------------
# print "Loading test input..."
# test_data = loadnp("/Users/stephanielaflamme/Desktop/Numpy sets/test_inputs_%s.npy" %s)

# ------------------------
# Getting test predictions
# ------------------------
# classifier = Perceptron(alpha=0.0005, n_iter=25)
# # train on the entire training set
# classifier.train(examples, np.asarray(map(one_hot_vectorizer, categories)))
# predictions = map(classifier.predict, test_data)

# write_test_output(predictions)

# ------------------------------
# RANDOM SEARCH CROSS-VALIDATION
# ------------------------------

print 'Beginning random search...'

# the parameter values under consideration
alphas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
n_layers = [1, 2]
nodes_layer = [10, 15, 20, 25, 30]


# this is where we'll save the cross-validation results
# also note that we store the parameters we used!
cross_val_alphas = []
cross_val_layers = []
cross_val_results = []
cross_val_train_results = []
cross_val_confusion_matrices = []

predictions = []
success_rates = []
train_success_rates = []


for _ in range(10):
	alpha, layers = randomize_params(alphas, n_layers, nodes_layer)

	cross_val_alphas.append(alpha)
	cross_val_layers.append(layers)

	# do cross-validation with current parameters... use k=2 because it takes a long time to run
	for data in CrossValidation(examples, categories, k=2):
	    train_data, train_result, valid_data, valid_result = data

	    # make and train classifier...
	    classifier = NeuralNet(layers=layers, alpha=alpha, epochs=20)
	    classifier.train(train_data, np.asarray(map(one_hot_vectorizer, train_result)))

	    # get validation results for this fold
	    guesses = map(classifier.predict, valid_data)
	    correct = filter(lambda x: x[0] == x[1], zip(guesses, valid_result))
	    ratio = len(correct)*1. / len(valid_result)
	    success_rates.append(ratio)

	    predictions.extend(guesses)

	# get the interesting results for this parameter configuration
	confusion_matrix = get_confusion_matrix(categories, predictions)
	success_ratio = sum(success_rates) / len(success_rates) * 100

	print 'Validation accuracy for alpha=%f, %d hidden layer(s) with layer sizes ' %(alpha, len(layers)-2), layers
	print '     %f' % success_ratio

	cross_val_confusion_matrices.append(confusion_matrix)
	cross_val_results.append(success_ratio)

	# save all the interesting results
	np.save('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/results/nn_alphas_STEPH', cross_val_alphas)
	np.save('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/results/nn_layers_STEPH', cross_val_layers)
	np.save('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/results/nn_crossval_confmatrices_STEPH', cross_val_confusion_matrices)
	np.save('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/results/nn_crossval_results_STEPH', cross_val_results)






