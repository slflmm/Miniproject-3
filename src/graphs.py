import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import *
from features import *

# ------------------
# PERCEPTRON RESULTS
# ------------------
class PerceptronPlotter(object):

	def __init__(self):
		# these were the parameters considered
		self.alphas = np.array([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
		self.niters = np.array([10, 15, 20, 25, 30, 35])

		# load crossval results
		self.crossval_results = np.load('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/results/perceptron_crossval_results.npy')

		self.crossval_matrices = np.load('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/results/perceptron_crossval_confmatrices.npy')

		self.crossval_training = np.load('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/results/perceptron_crossval_training_accuracy.npy')

		self.argmax = np.argmax(self.crossval_results)
		self.best = np.max(self.crossval_results)
		self.best_alpha = self.alphas[self.argmax % 7]
		self.best_niters = self.niters[self.argmax / 7]

	def best_results(self):
		print 'Best PERCEPTRON: '
		print 'accuracy = %f' % self.best
		print 'alpha = %f' % self.best_alpha
		print 'n_iters = %d' % self.best_niters


	def gridsearch(self):
		accuracies = self.crossval_results.reshape((len(self.niters),len(self.alphas)))

		fig = plt.figure(figsize=(7,6))
		ax1 = fig.add_subplot(111)

		ax1.tick_params(direction='out', which='both')
		ax1.set_xlabel('Learning rate')
		ax1.set_ylabel('Number of iterations')
		ax1.set_xticks(self.alphas)
		ax1.set_yticks(self.niters)

		cax = ax1.contourf(self.alphas, self.niters, accuracies, np.arange(22, 27, 0.25), extend='both')
		# cs=ax1.contour(alphas, niters, accuracies, np.arange(20,27,0.25),colors='k')
		# ax1.clabel(cs, fmt = '%d', colors = 'k')

		ax1.set_xscale('log')

		cbar = fig.colorbar(cax)
		plt.savefig('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/report/perceptron_gridsearch.pdf')


	def conf_matrix(self):
		fig = plt.figure()
		ax = fig.add_subplot(111)

		mat = self.crossval_matrices[self.argmax].tolist()
		cax = ax.matshow(mat, cmap=cm.jet)
		fig.colorbar(cax)

		for x in xrange(10):
			for y in xrange(10):
				ax.annotate('%4.2f' % (mat[x][y]), xy=(y,x), horizontalalignment='center', verticalalignment='center', color='white')

		plt.xticks(np.arange(10))
		plt.yticks(np.arange(10))
		ax.set_title('True label', fontsize=16)
		ax.set_ylabel('Prediction', fontsize=16)

		plt.savefig('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/report/perceptron_confusion.pdf')


	def plt_learningrate(self):
		'''
		Perceptron's learning rate -- validation vs training set 
		Uses the best n_iter, varies learning rate
		'''
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)

		crossval_results = self.crossval_results
		crossval_training = self.crossval_training
		argmax = self.argmax

		valid_acc = crossval_results[(argmax/7)*7:(argmax/7)*7 + 7]
		train_acc = crossval_training[(argmax/7)*7:(argmax/7)*7 + 7]

		v, = ax.plot(self.alphas, valid_acc, marker='D', color='green', label='Validation, %d iterations' % self.best_niters)
		t, = ax.plot(self.alphas, train_acc, marker='D', color='green', linestyle=':', label='Training, %d iterations' % self.best_niters)

		# legend
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles, labels, loc=0)

		# labels
		ax.set_xlabel('Learning rate')
		ax.set_ylabel('Mean accuracy')
		ax.set_xscale('log')

		plt.savefig('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/report/perceptron_learningrate.pdf')


	def plt_niters(self):
		'''
		Perceptron's number of iterations -- validation vs training set 
		Uses the best alpha, varies number of iterations.
		'''
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)

		crossval_results = self.crossval_results
		crossval_training = self.crossval_training
		argmax = self.argmax

		valid_acc = []
		for i in xrange(len(self.niters)):
			valid_acc.append(crossval_results[7*i + argmax%7])
		train_acc = []
		for i in xrange(len(self.niters)):
			train_acc.append(crossval_training[7*i + argmax%7])

		v, = ax.plot(self.niters, valid_acc, marker='D', color='green', label='Validation, alpha=%5.4f' % self.best_alpha)
		t, = ax.plot(self.niters, train_acc, marker='D', color='green', linestyle=':', label='Training, alpha=%5.4f' % self.best_alpha)

		# legend
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles, labels, loc=0)

		# labels
		ax.set_xlabel('Number of iterations')
		ax.set_ylabel('Mean accuracy')

		plt.savefig('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/report/perceptron_iterations.pdf')

	def all_plots(self):
		'''
		Builds all plots and prints all info.
		'''
		self.best_results()
		self.gridsearch()
		self.conf_matrix()
		self.plt_niters()
		self.plt_learningrate()


# ---------------------
# GABOR FILTER EXAMPLES
# ---------------------
class GaborPlotter(object):

	def __init__(self):
		self.n_theta = 4
		self.sigmas=[1,3]
		self.frequencies=[0.05, 0.25]
		self.kernels = getGaborKernels(self.n_theta, self.sigmas, self.frequencies)

	def show_samples(self):
		fig = plt.figure()
		for i in range(4):
			plt.subplot(2,2,i+1)
			plt.imshow(np.real(self.kernels[i + 4*i]), cmap=plt.cm.gray, interpolation='nearest')
			# the loop goes theta in range n_theta, then sigmas, then frequencies
			# theta = (i%4)*1. / self.n_theta
			if (i==1):
				plt.title(r'$\theta=\pi/%d$, $\sigma=%1.f$, $freq=%3.2f$' % (self.n_theta, self.sigmas[i/2], self.frequencies[i%2]), fontsize=16)
			elif (i==2):
				plt.title(r'$\theta=\pi/%d$, $\sigma=%1.f$, $freq=%3.2f$' % (self.n_theta/2, self.sigmas[i/2], self.frequencies[i%2]), fontsize=16)
			elif (i != 0):
				plt.title(r'$\theta=%d\pi/%d$, $\sigma=%1.f$, $freq=%3.2f$' % (i%4, self.n_theta, self.sigmas[i/2], self.frequencies[i%2]), fontsize=16)
			else:
				plt.title(r'$\theta=0$, $\sigma=%1.f$, $freq=%3.2f$' % (self.sigmas[i/2], self.frequencies[i%2]), fontsize=16)

			plt.gca().xaxis.set_major_locator(plt.NullLocator())
			plt.gca().yaxis.set_major_locator(plt.NullLocator())

		plt.savefig('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/report/gabors.pdf')

		plt.close()

def convnet_visualize():

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	epochs = range(10,350,10)

	convnet2 = [252.200000,190.500000,154.700000,130.500000,124.400000,119.100000,109.800000,105.000000,109.600000,107.600000,97.800000,98.200000,99.200000,100.600000,99.900000,95.900000,98.800000,93.200000] 
	convnet2 = map(lambda x: 1 - x/500, convnet2)
	convnet2+= [None]*(len(epochs)-len(convnet2))
	convnet3 = [221.000000,175.200000,148.600000,135.400000,133.000000, 128.100000,118.000000,115.500000,115.100000,112.900000,113.100000,111.300000,111.600000,107.200000]
	convnet3 = map(lambda x: 1 - x/500, convnet3)
	convnet3 += [None]*(len(epochs)-len(convnet3))
	convnet4 = [233.300000,180.900000,150.900000,133.900000,118.500000,115.400000,111.900000,106.200000,105.700000,103.500000,93.800000,96.200000,93.800000,93.400000,90.100000,90.000000,93.600000,90.400000,87.500000,89.900000,88.000000,85.800000,88.800000]
	convnet4 = map(lambda x: 1 - x/500, convnet4)
	convnet4 += [None]*(len(epochs)-len(convnet4))
	convnet5 = [223.000000,154.700000,112.900000,103.100000,87.500000,80.300000,78.100000,73.000000,72.500000,71.500000,70.700000,69.300000,67.500000,66.300000,66.300000,65.000000,64.700000,66.600000,64.700000,59.000000,63.800000,65.000000,65.600000,63.700000,63.400000, 66.100000,63.900000,65.800000,64.100000,64.400000,67.300000,63.300000,63.700000,60.800000]
	convnet5= map(lambda x: 1 - x/500, convnet5)
	convnet5 += [None]*(len(epochs)-len(convnet5))

	c2, = ax.plot(epochs, convnet2, marker='D', color='green', label='Basic')
	c3, = ax.plot(epochs, convnet3, marker='x', color='red', label='Perturbed')
	c4, = ax.plot(epochs, convnet4, marker='o', color='blue', label='Larger filters')
	c5, = ax.plot(epochs, convnet5, marker='v', color='magenta', label='3 conv layers')

	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles, labels, loc=0)

	# labels
	ax.set_xlabel('Number of epochs')
	ax.set_ylabel('Validation accuracy')

	plt.savefig('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/report/convnet_comparison.pdf')

# ------------------------------------------
# Do the actual graphing of your choice here
# ------------------------------------------
convnet_visualize()




