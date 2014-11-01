from skimage.filter import gabor_kernel
import numpy as np
from numpy import linalg
from skimage import data
from scipy import ndimage
from skimage.util import img_as_float
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.preprocessing import normalize
import csv
import copy
import random
import itertools

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec


'''
	Feature value = energy of response (sum of squares) aka Frobeniius norm
'''
def get_gabor_features(image, gaborKernels):
	features = []
	for k in gaborKernels:
		convimg = ndimage.convolve(image.reshape(48,48), k, mode='wrap')
		features.append(convimg.mean())
		features.append(convimg.var())
		features.append(linalg.norm(convimg))
		features.append(linalg.norm(convimg, 1))
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
		print len(valid)
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

def normalizeSet(trainX):
	normTrainX = normalize(trainX)
	return normTrainX

def fixOutputs(validY, trainY):
	vY = []
	tY = []
	for y in validY:
		tmp = int(y[1])
		vY.append(tmp)
	for y in trainY:
		tmp = int(y[1])
		tY.append(tmp)

	return (vY, tY)

def one_hot_vectorizer(n):
	v = np.zeros(10)
	v[n] = 1
	return v



def save_train_features():
	# ------------------------
	# Loading raw training set
	# ------------------------
	print "Loading train output..."
	train_output = loadnp("C:/Users/MicroMicro/Documents/Benjamin/Anaconda/Miniproject-3/src/train_outputs.npy")

	print "Loading train input..."
	train_input = loadnp("C:/Users/MicroMicro/Documents/Benjamin/Anaconda/Miniproject-3/src/train_inputs.npy")

	# ----------------------
	# Standardizing features (and saving)
	# ----------------------
	print "Standardizing features (x-mean / sigma)..."
	# keep the scaling so that we can later use it on the test set!
	scaler = preprocessing.StandardScaler().fit(train_input)
	examples = scaler.transform(train_input)
	np.save('train_inputs_standardized', examples)

	# -----------------------
	# PCA features (and saving)
	# -----------------------
	print "Getting PCA features..."
	pca = decomposition.PCA()
	examples = pca.fit_transform(examples)
	np.save('train_inputs_pca', examples)

	# --------------
	# Gabor features
	# --------------
	print "Get gabor features using default values"
	examples = loadnp("C:/Users/MicroMicro/Documents/Benjamin/Anaconda/Miniproject-3/src/train_inputs_standardized.npy")
	kernels = getGaborKernels()
	# data = []
	# for ex in examples:
	# 	f = get_gabor_features(ex, kernels)
	# 	data.append(f)
	# 	print str(len(data))

	# this will make it go at least a bit faster than looping
	data = map(lambda x: get_gabor_features(x, kernels), examples)
	
	print "Normalizing..."
	#scaler = preprocessing.StandardScaler().fit(data)
	#examples = scaler.transform(data)
	np.save('train_inputs_gabor', data)

def save_test_features():
	print "Loading train input..."
	train_input = loadnp("/Users/stephanielaflamme/Desktop/data_and_scripts/train_inputs.npy")
	train_standardized = loadnp("/Users/stephanielaflamme/Desktop/Numpy sets/train_inputs_standardized.npy")

	print "Loading test input..."
	test_input = loadnp("/Users/stephanielaflamme/Desktop/data_and_scripts/test_inputs.npy")

	# use the scaling from the training set
	print "Getting standardized features..."
	scaler = preprocessing.StandardScaler().fit(train_input)
	examples = scaler.transform(test_input)
	np.save('test_inputs_standardized', examples)

	print "Getting PCA features..."
	pca = decomposition.PCA()
	pca.fit(train_standardized)
	examples = pca.transform(examples)
	np.save('test_inputs_pca', examples)



# save_train_features()
