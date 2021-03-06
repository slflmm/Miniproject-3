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
# choice of 3 feature sets
feature_sets = ['standardized', 'pca', 'gabor']
examples = loadnp("/Users/stephanielaflamme/Desktop/Numpy sets/train_inputs_"+ feature_sets[1]+".npy")

# ----------------
# Loading test set
# ----------------
print "Loading test input..."
test_data = loadnp("/Users/stephanielaflamme/Desktop/Numpy sets/test_inputs_"+ feature_sets[1]+".npy")

# ------------------------
# Getting test predictions
# ------------------------
classifier = Perceptron(alpha=0.0005, n_iter=25)
# train on the entire training set
classifier.train(examples, np.asarray(map(one_hot_vectorizer, categories)))
predictions = map(classifier.predict, test_data)

write_test_output(predictions)

# ---------------------------
# GRIDSEARCH CROSS-VALIDATION
# ---------------------------

# print 'Beginning gridsearch...'

# # the parameter values under consideration
# alphas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
# n_iters = [10, 15, 20, 25, 30, 35]

# # this is where we'll save the cross-validation results
# cross_val_results = []
# cross_val_train_results = []
# cross_val_confusion_matrices = []

# for n_iter in n_iters:
# 	for alpha in alphas:
# 		predictions = []
# 		success_rates = []
# 		train_success_rates = []
# 		# do coss-validation with current parameters
# 		for data in CrossValidation(examples, categories, k=5):
# 		    train_data, train_result, valid_data, valid_result = data

# 		    classifier = Perceptron(alpha=alpha, n_iter=n_iter)
# 		    # train with one-hot outputs...
# 		    classifier.train(train_data, np.asarray(map(one_hot_vectorizer, train_result)))

# 		    training_guesses = map(classifier.predict, train_data)
# 		    training_correct = filter(lambda x: x[0] == x[1], zip(training_guesses, train_result))
# 		    training_ratio = len(training_correct)*1. / len(train_result)
# 		    train_success_rates.append(training_ratio)

# 		    guesses = map(classifier.predict, valid_data)
# 		    correct = filter(lambda x: x[0] == x[1], zip(guesses, valid_result))
# 		    ratio = len(correct)*1. / len(valid_result)
# 		    success_rates.append(ratio)

# 		    predictions.extend(guesses)

# 		# get the interesting results for this parameter configuration
# 		confusion_matrix = get_confusion_matrix(categories, predictions)
# 		train_success_ratio = sum(train_success_rates) / len(train_success_rates) * 100
# 		success_ratio = sum(success_rates) / len(success_rates) * 100

# 		print 'Cross-val accuracy for alpha=%f, n_iter=%d: %f' % (alpha, n_iter, success_ratio)

# 		cross_val_confusion_matrices.append(confusion_matrix)
# 		cross_val_results.append(success_ratio)
# 		cross_val_train_results.append(train_success_ratio)

# # save all the interesting results
# np.save('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/results/perceptron_crossval_confmatrices', cross_val_confusion_matrices)
# np.save('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/results/perceptron_crossval_results', cross_val_results)
# np.save('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/results/perceptron_crossval_training_accuracy', cross_val_train_results)






