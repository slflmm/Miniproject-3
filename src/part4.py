from theano_classifier import *
from theano_convnet import *
from theano_utils import *
from features import *
from utils import *


# # ------------------------
# # Loading training set
# # ------------------------
print "Loading train output..."
categories = loadnp("/home/ml/slafla2/Miniproject-3/src/train_outputs.npy")
# print categories.shapes

print "Loading train input..."
examples = loadnp("/home/ml/slafla2/Miniproject-3/src/train_inputs.npy")

def contrast_normalize(x):
	min_x = min(x)
	max_x = max(x)
	res = (x - min_x)*255/(max_x - min_x)
	return np.array(res)

print "Doing contrast normalization..."
examples = map(contrast_normalize, examples)
examples = np.array(examples)
print examples.shape

# # ------------------------
# # Getting test predictions
# # ------------------------
# print "Loading test input..."
# test_data = loadnp("/Users/stephanielaflamme/Desktop/Numpy sets/test_inputs_"+ feature_sets[1]+".npy")

# classifier = Perceptron(alpha=0.0005, n_iter=25)
# # train on the entire training set
# classifier.train(examples, np.asarray(map(one_hot_vectorizer, categories)))
# predictions = map(classifier.predict, test_data)

# first_layer_filter_sizes = [5,6,7,8,9,10] # based on deeplearning.net tips
# first_layer_number_filters = range(10,5,50)
# n_convlayers = [1,2,3] # any more than this and we have a problem
# n_hiddenlayers = [1,2] # more than this and we have a time problem
print "Starting cross-validation..."

# Try to keep filter sizes 5-8
# At least 10 filters per layer
# 3 layers is good
# Only one hidden layer required
for data in CrossValidation(examples, categories, k=10):
	train_data, train_result, valid_data, valid_result = data

	# print "Generating new examples..."
	# new_data = map(lambda x,y: add_perturbation(x,y), train_data, train_result)
	# new_examples = np.asarray(map(lambda x: x[0], new_data))
	# new_outputs = np.asarray(map(lambda y: y[1], new_data)) 
	# print "Combining..."
	# train_input_expanded = np.asarray(zip(train_data, new_examples)).reshape((2*len(train_data), -1))
	# train_output_expanded = np.asarray(zip(train_result, new_outputs)).flatten()
	# np.save('train_inputs_expanded', train_input_expanded)
	# np.save('train_outputs_expanded', train_output_expanded)

	print 'Building convnet...'
	n_epochs = 400
	batch_size = 256
	learning_rate = 0.25
	net = ConvNet(rng = np.random.RandomState(1234),
		# we're getting 720 instead of 320, why?  
		# next image shape is (previous_image_shape - filter_size + 1) / poolsize
		# after  (20,1,7,7) images are (48-7+1 = 42) --> 21 x 21, then (21-6+1 = 16) --> 8x8 
		# after (20, 1, 5, 5) images are (48-5+1 = 44) --> 22 x 22, then (22-5+1 = 18) --> 9x9, then... 
		# (48-9+1=40) => 20x20, then (20-5+1 = 16)=> 8, then (8-5+1=4)=> 2
		# (48-7+1 = 42) => 21x21, then (21-6+1=16)=> 8x8, then (8-4+1=5)=> 5x5, and finally (5-3+1)=> 3x3
		conv_filter_shapes = [(32, 1, 7, 7), (64, 32, 6, 6),[80, 64, 4, 4], [96, 80, 3, 3]], #(22, 22) output, shape ()
		image_shapes = [(batch_size, 1,48,48),(batch_size, 32, 21, 21), (batch_size, 64, 8, 8), (batch_size, 80, 5, 5)], # (9, 9) output, shape (20,50,22,22) #80*2*2=320 but not getting that
		poolsizes=[(2,2),(2,2), None, None],
		hidden_layer_sizes=[200],
		n_outputs=10,
		learning_rate=learning_rate,
		dropout_rate=0.5,
		activations=[rectified_linear],
		batch_size=batch_size,
		# train_set_x = train_input_expanded,
		# train_set_y = train_output_expanded,
		train_set_x=train_data,
		train_set_y=train_result,
		valid_set_x=valid_data,
		valid_set_y=valid_result
		)
	print 'Making the trainer...'
	learner = Trainer(net)

	print 'Training...'
	trainerr, validerr = learner.train(learning_rate,n_epochs,batch_size)

	print "Training error: %f" % trainerr
	print "Validation error: %f" % validerr

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


