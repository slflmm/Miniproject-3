import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import *

# ------------------
# PERCEPTRON RESULTS
# ------------------

# these were the parameters considered
alphas = np.array([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
niters = np.array([10, 15, 20, 25, 30, 35])

# load crossval results
crossval_results = np.load('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/src/results/perceptron_crossval_results.npy')

crossval_matrices = np.load('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/src/results/perceptron_crossval_confmatrices.npy')

crossval_training = np.load('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/src/results/perceptron_crossval_training_accuracy.npy')

# ------------
# Best results
# ------------
argmax = np.argmax(crossval_results)
best = np.max(crossval_results)
best_alpha = alphas[argmax % 7]
best_niters = niters[argmax / 7]
print 'Best PERCEPTRON: '
print 'accuracy = %f' % best
print 'alpha = %f' % best_alpha
print 'n_iters = %d' % best_niters

# -------------------------------- 
# Perceptron's crossval gridsearch
# --------------------------------
accuracies = crossval_results.reshape((len(niters),len(alphas)))

fig = plt.figure(figsize=(7,6))
ax1 = fig.add_subplot(111)

ax1.tick_params(direction='out', which='both')
ax1.set_xlabel('Learning rate')
ax1.set_ylabel('Number of iterations')
ax1.set_xticks(alphas)
ax1.set_yticks(niters)

cax = ax1.contourf(alphas, niters, accuracies, np.arange(21, 27, 0.5), extend='both')
# cs=ax1.contour(alphas, niters, accuracies, np.arange(20,27,0.25),colors='k')
# ax1.clabel(cs, fmt = '%d', colors = 'k')

ax1.set_xscale('log')

cbar = fig.colorbar(cax)
plt.savefig('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/report/perceptron_gridsearch.pdf')


# -----------------------------
# Perceptron's confusion matrix
# -----------------------------

fig = plt.figure()
ax = fig.add_subplot(111)

mat = crossval_matrices[argmax].tolist()
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

# -------------------------------------------------------- 
# Perceptron's learning rate -- validation vs training set
# (using the best number of iterations, vary learning rate)
# --------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

valid_acc = crossval_results[(argmax/7)*7:(argmax/7)*7 + 7]
train_acc = crossval_training[(argmax/7)*7:(argmax/7)*7 + 7]

v, = ax.plot(alphas, valid_acc, marker='D', color='green', label='Validation, %d iterations' % best_niters)
t, = ax.plot(alphas, train_acc, marker='D', color='green', linestyle=':', label='Training, %d iterations' % best_niters)

# legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=0)

# labels
ax.set_xlabel('Learning rate')
ax.set_ylabel('Mean accuracy')
ax.set_xscale('log')

plt.savefig('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/report/perceptron_learningrate.pdf')

# -------------------------------------------------------- 
# Perceptron's iterations -- validation vs training set
# (using the best learning rate, vary iterations)
# --------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

valid_acc = []
for i in xrange(len(niters)):
	valid_acc.append(crossval_results[7*i + argmax%7])
train_acc = []
for i in xrange(len(niters)):
	train_acc.append(crossval_training[7*i + argmax%7])

v, = ax.plot(niters, valid_acc, marker='D', color='green', label='Validation, alpha=%5.4f' % best_alpha)
t, = ax.plot(niters, train_acc, marker='D', color='green', linestyle=':', label='Training, alpha=%5.4f' % best_alpha)

# legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=0)

# labels
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Mean accuracy')

plt.savefig('/Users/stephanielaflamme/Dropbox/COMP 598/Miniproject3/report/perceptron_iterations.pdf')






