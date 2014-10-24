from skimage.filter import gabor_kernel
import numpy as np
from skimage import data
from scipy import ndimage
from skimage.util import img_as_float
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import csv
import copy
import random


'''
	Taken from an example @ http://scikit-image.org/docs/dev/auto_examples/plot_gabor.html
'''
def get_gabor_features(image, gaborKernels):
	features = np.zeros((len(gaborKernels), 2), dtype = np.double)
	count = 0
	for k in gaborKernels:
		filt = ndimage.convolve(image, k, mode='wrap')
		features[count, 0] = filt.mean()
		features[count, 1] = filt.var()
		count = count + 1
	return features.flatten()

'''
	Generates kernels to be used in feature extraction or as a kernel for SVM.
	numTheta: int, number of thetas to use between 0,1 evenly spaced out
	sigmaMin: int, Min sigma to consider (eg. 1)
	sigmaMax: int, Max sigma to consider (eg. 3)
	freqMin: float, min frequency to start off with (eg. 0.05)
	freqMax: float, max frequency to end with (eg. 0.25)
'''
def getGaborKernels(numTheta, sigmaMin, sigmaMax, freqMin, freqMax):
	gaborKernels = []
	for theta in range(numTheta):
		theta = theta / float(numTheta) * np.pi
		for sigma in (sigmaMin, sigmaMax):
			for frequency in (freqMin, freqMax):
				kernel = np.real(gabor_kernel(frequency, theta, sigma_x = sigma, sigma_y = sigma))
				gaborKernels.append(kernel)
	return gaborKernels


'''
	In our case, X should be the list of Ids so that images are not copied needlessly
'''
def splitValidTest(X, Y, validRatio):
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


# kernels = getGaborKernels(4, 1, 3, 0.05, 0.25)
# shrink = (slice(0,None,3), slice(0,None,3))
# gabor_set = []
# for i in xrange(len(trainInput)):
# 	index = i + 1
# 	print index
# 	img = img_as_float(data.load("C:/Users/MicroMicro/Documents/Benjamin/Anaconda/MP3Data/train_images/" + str(index) + ".png"))
# 	print img
# 	feats = get_features(img, kernels)
# 	gabor_set.append(feats)

# print "saving kernel feature set..."
# np.save('gabor_feats', gabor_set)

#(validSet, trainSet, validSetY, trainSetY) = splitValidTest(trainInput, trainOutput, 0.1)
#print "done splitting... "

#(validSetY, trainSetY) = fixOutputs(validSetY, trainSetY)
#print validSetY
#print trainSetY

#print "Testing using pixel features... training..."
#clf = SVC(kernel='linear')
#clf.fit(trainSet, trainSetY)

#print "checking against Valid Set..."
#acc = clf.score(validSet, validSetY)
#print acc
