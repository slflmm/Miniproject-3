from skimage.filter import gabor_kernel
import numpy
from skimage import data
from scipy import ndimage
from skimage.util import img_as_float


'''
	Taken from an example @ http://scikit-image.org/docs/dev/auto_examples/plot_gabor.html
'''
def get_features(image, gaborKernels):
	features = numpy.zeroes((len(gaborKernels), 2), dtype = numpy.double)
	count = 0
	for k in gaborKernels:
		filt = ndimage.convolve(image, k, mode='wrap')
		features[count, 0] = filt.mean()
		features[count, 1] = filt.var()
		count = count + 1
	return features

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
		theta = theta / numTheta. * numpy.pi
		for sigma in (sigmaMin, sigmaMax):
			for frequency in (freqMin, freqMax):
				kernel = numpy.real(gabor_kernel(frequency, theta, sigma_x = sigma, sigma_y = sigma))
				gaborKernels.append(kernel)
	return gaborKernels
