from basic_classifiers import *

# testing perceptron on 'and' function
x = np.array([[0,0],[0,1],[1,0],[1,1]])
# one-hot version where [1,0] means false (0), [0,1] true (1)
y = np.array([[1,0],[1,0],[1,0],[0,1]]) 

p = Perceptron()
p.train(x,y)

for e, o in zip(x,y):
	print p.predict(e), o
