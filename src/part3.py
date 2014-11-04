from sklearn.svm import SVC
from sklearn import linear_model
import numpy as np
import random
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.pylab import *
import matplotlib.pyplot as plt
from skimage.filter import gabor_kernel
from scipy import ndimage
from skimage.util import img_as_float
import csv

'''
	Feature value = energy of response (sum of squares) aka Frobeniius norm
'''
def get_gabor_features(image, gaborKernels):
	image = np.array(image)
	#original_pixels = image.flat
	features = []
	#features.extend(original_pixels)
	for i, k in enumerate(gaborKernels):
		convimg = ndimage.convolve(image.reshape(48,48), k, mode='wrap')
		features.extend(convimg.flat)

		#features.append(convimg.mean())
		#features.append(convimg.var())
		#features.append(linalg.norm(convimg))
		#features.append(linalg.norm(convimg, 1))
	return features

'''
	Generates kernels to be used in feature extraction or as a kernel for SVM.
	numTheta: int, number of thetas to use between 0,1 evenly spaced out
	sigmaMin: int, Min sigma to consider (eg. 1)
	sigmaMax: int, Max sigma to consider (eg. 3)
	freqMin: float, min frequency to start off with (eg. 0.05)
	freqMax: float, max frequency to end with (eg. 0.25)
'''
def getGaborKernels(n_theta = 4, sigmas=[1,3], frequencies=[0.05, 0.25]):
	gaborKernels = []
	for theta in range(n_theta):
		theta = theta / float(n_theta) * np.pi
		for sigma in sigmas:
			for frequency in frequencies:
				kernel = np.real(gabor_kernel(frequency, theta, sigma_x = sigma, sigma_y = sigma))
				gaborKernels.append(kernel)
	return gaborKernels

'''
	In our case, X should be the list of Ids so that images are not copied needlessly
'''
def splitValidTest(X, Y, validRatio):
	X = X.tolist()
	Y = Y.tolist()
	valid = []
	valid_y = []
	length = len(X)
	while (len(valid) < validRatio * length):
		chosen_index = random.randint(0, len(X) - 1)
		valid.append(X[chosen_index])
		valid_y.append(Y[chosen_index])
		del X[chosen_index]
		del Y[chosen_index]
	print "len valid: " + str(len(valid)) + "len Y" + str(len(valid_y))
	print "len train: " + str(len(X)) + "len Y" + str(len(Y))
	return (valid, X, valid_y, Y)

def loadnp(filename):
	data = np.load(filename)
	return data

def saveMatrix(mat):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	cax = ax.matshow(mat, cmap=cm.jet)
	fig.colorbar(cax)

	for x in xrange(10):
		for y in xrange(10):
			ax.annotate('%4.2f' % (mat[x][y]), xy=(y,x), horizontalalignment='center', verticalalignment='center', color='white')

	plt.xticks(np.arange(10))
	plt.yticks(np.arange(10))
	ax.set_title('True label', fontsize=16)
	ax.set_ylabel('Prediction', fontsize=16)

	plt.savefig('SVM_gabor_confusion.pdf')

def saveAccPlot(niters, valid_acc, train_acc, var):
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	print niters
	v, = ax.plot(niters, valid_acc, marker='D', color='red', label='Validation')
	t, = ax.plot(niters, train_acc, marker='D', color='green', linestyle=':', label='Training')

	# legend
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles, labels, loc=0)

	# labels
	ax.set_xlabel(var)
	ax.set_ylabel('Mean accuracy')
	ax.set_xscale('log')
	plt.savefig('SVM_standard_rate2.pdf')

'''
Done on the fly with the standard dataset because it is rather large
'''
def runKFoldWithGabor(numFolds, trainInput, trainOutput):
	kfolds = cross_validation.StratifiedKFold(trainOutput, n_folds=numFolds)
	conf_matrices = []
	unique = np.unique(trainOutput)
	kernels = getGaborKernels()

	for train_index, test_index in kfolds:
		(trainSet, validSet) = (trainInput[train_index], trainInput[test_index])
		(trainSetY, validSetY) = (trainOutput[train_index], trainOutput[test_index])
		print "len valid: " + str(len(validSet)) + "len Y" + str(len(validSetY))
		print "len train: " + str(len(trainSet)) + "len Y" + str(len(trainSetY))
		print "training SGD Classifier..."
		data = []
		data_y = []
		counter = 0
		clf = linear_model.SGDClassifier()

		for i in xrange(len(trainSet)):
			ex = trainSet[i]
			f = get_gabor_features(ex, kernels)
			data.append(f)
			data_y.append(trainSetY[i])
			if(len(data)%250 == 0 or i == len(trainSet) - 1):
				counter = counter + 1
				print str(len(data) * counter)
				clf.partial_fit(data, data_y ,classes=unique)
				data = []
				data_y = []
		data = []
		print "checking against Valid Set..."
		for i in xrange(len(validSet)):
			ex = validSet[i]
			f = get_gabor_features(ex, kernels)
			data.append(f)
		predictedY = clf.predict(data)
		cMatrix = confusion_matrix(validSetY, predictedY)
		cMatrix = cMatrix/cMatrix.astype(np.float).sum(axis=0)
		conf_matrices.append(cMatrix)

	final_matrix = np.mean(conf_matrices, axis=0)
	saveMatrix(final_matrix)

def runKFoldWithStandardTestIterations(numFolds, trainInput, trainOutput, numIterations):
	kfolds = cross_validation.StratifiedKFold(trainOutput, n_folds=numFolds)
	conf_matrices = []
	accuracyV = []
	accuracyT = []

	for train_index, test_index in kfolds:
		(trainSet, validSet) = (trainInput[train_index], trainInput[test_index])
		(trainSetY, validSetY) = (trainOutput[train_index], trainOutput[test_index])
		print "len valid: " + str(len(validSet)) + "len Y" + str(len(validSetY))
		print "len train: " + str(len(trainSet)) + "len Y" + str(len(trainSetY))
		print "training SGD Classifier..."
		clf = linear_model.SGDClassifier(n_iter=numIterations)
		clf.fit(trainSet, trainSetY)
		print "checking against Valid Set..."
		predictedY = clf.predict(validSet)
		predX = clf.predict(trainSet)
		cMatrix = confusion_matrix(validSetY, predictedY)
		cMatrix = cMatrix/cMatrix.astype(np.float).sum(axis=0)
		conf_matrices.append(cMatrix)
		accV = accuracy_score(validSetY, predictedY)
		accT = accuracy_score(trainSetY, predX)
		accuracyV.append(accV)
		accuracyT.append(accT)

	final_matrix = np.mean(conf_matrices, axis=0)
	saveMatrix(final_matrix)
	return np.mean(accuracyV), np.mean(accuracyT)

def runKFoldWithStandardTestLearningRate(numFolds, trainInput, trainOutput, l_rate):
	kfolds = cross_validation.StratifiedKFold(trainOutput, n_folds=numFolds)
	conf_matrices = []
	accuracyV = []
	accuracyT = []

	for train_index, test_index in kfolds:
		(trainSet, validSet) = (trainInput[train_index], trainInput[test_index])
		(trainSetY, validSetY) = (trainOutput[train_index], trainOutput[test_index])
		print "len valid: " + str(len(validSet)) + "len Y" + str(len(validSetY))
		print "len train: " + str(len(trainSet)) + "len Y" + str(len(trainSetY))
		print "training SGD Classifier..."
		clf = linear_model.SGDClassifier(learning_rate='constant', eta0 = l_rate)
		clf.fit(trainSet, trainSetY)
		print "checking against Valid Set..."
		predictedY = clf.predict(validSet)
		predX = clf.predict(trainSet)
		cMatrix = confusion_matrix(validSetY, predictedY)
		cMatrix = cMatrix/cMatrix.astype(np.float).sum(axis=0)
		conf_matrices.append(cMatrix)
		accV = accuracy_score(validSetY, predictedY)
		accT = accuracy_score(trainSetY, predX)
		accuracyV.append(accV)
		accuracyT.append(accT)

	final_matrix = np.mean(conf_matrices, axis=0)
	saveMatrix(final_matrix)
	return np.mean(accuracyV), np.mean(accuracyT)

# ------------------------
# Getting test predictions using SVM and gabor features 
# ------------------------
trainInput = loadnp('C:/Users/MicroMicro/Documents/Benjamin/Anaconda/Miniproject-3/src/train_inputs_standardized.npy')
trainOutput = loadnp("C:/Users/MicroMicro/Documents/Benjamin/Anaconda/Miniproject-3/src/train_outputs.npy")
testInput = loadnp("C:/Users/MicroMicro/Documents/Benjamin/Anaconda/Miniproject-3/src/test_inputs_standardized.npy")

print "training SGD Classifier..."
clf = linear_model.SGDClassifier()
clf.fit(trainInput, trainOutput)
predictedY = clf.predict(testInput)
	
test_output_file = open('test_output_SVM.csv', "wb")
writer = csv.writer(test_output_file, delimiter=',') 
writer.writerow(['Id', 'Prediction']) # write header
id = 1
for y in predictedY:
	row = [id, y]
	writer.writerow(row)
	id = id + 1
test_output_file.close()


#let's run some experiments! First fix optimal learning rate and then vary iteration numbers
#validaccs = []
#trainaccs = []
#iters = range(1,100)

#for iter in iters:
#	(v, t) = runKFoldWithStandardTestIterations(5, trainInput, trainOutput, iter)
#	print "accuracy with " + str(iter) + " iterations is " + str(v)
#	validaccs.append(v)
#	trainaccs.append(t)
#	with open("iterTestData.npy",'a') as f_handle:
#		np.savetxt(f_handle, [v])
#		np.savetxt(f_handle, [t])

#rates = []
#counter = 0.000005
#print rates
#while counter < 0.0001:
#	(v, t) = runKFoldWithStandardTestLearningRate(5, trainInput, trainOutput, counter)
#	print "accuracy with " + str(counter) + " rate is " + str(v)
#	validaccs.append(v)
#	trainaccs.append(t)
#	rates.append(counter)
#	with open("rateTestData2.npy",'a') as f_handle:
#		np.savetxt(f_handle, [v])
#		np.savetxt(f_handle, [t])
#	counter = counter + 0.000002

#with open("rateTestData.npy", 'r') as f_handle:
#	data = loadtxt(f_handle)
#	counter = 0.0001
#	for i in xrange(len(data)):
#		if(i%2 == 0):
#			validaccs.append(data[i])
#			rates.append(counter)
#			counter = counter + 0.0004
#		else:
#			trainaccs.append(data[i])
#saveAccPlot(rates, validaccs, trainaccs, "Learning Rate")
