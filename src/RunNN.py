import numpy as np
import utils
import random
import NNClassifier as nnClassifier

trainingInput = np.load('standardized_train_inputs.npy')
trainingOutput = np.load('train_outputs.npy')

nFeatures = trainingInput.shape[1]

trainingInput = trainingInput[:100,:]
trainingOutput = trainingOutput[:100]
#for i in range(15):
#    alpha, beta, nodes, nIter, nLayers = utils.RandomHyperParams(nFeatures)
#    print("lmbda = {0}, alpha = {1}, beta = {2}, nodes = {3}, nIter = {4}, nLayers = {5}".format(lmbda, alpha, beta, nodes, nIter, nLayers))

lmbda = 5
alpha = 1e-05
beta = 0.1
nodes = 31
nIter = 170
nLayers = 2

if nLayers == 1:
    nnc = nnClassifier.NeuralNetworkClassifier((nFeatures, nodes, 10))
    print("nnc({0}, {1}, {2})".format(nFeatures, nodes, 10))
else:
    nodes2 = int(nodes / 2)
    nnc = nnClassifier.NeuralNetworkClassifier((nFeatures, nodes, nodes2, 10))
    print("nnc({0}, {1}, {2}, {3})".format(nFeatures, nodes, nodes2, 10))


_trainingAccuracy = []
_testAccuracy = []

nTrainBatches = int(trainingInput.shape[0] / nnc.nBatchSize)

trainingAccuracy = 0
testAccuracy = 0

nIt = 0
for xMiniBatch, yMiniBatch in nnc.miniBatches(trainingInput, trainingOutput, nnc.nBatchSize):
    miniBatchAcc = nnc.TrainingEpoch(xMiniBatch, yMiniBatch, alpha, lmbda, beta)
    if nIt == nIter - 1:
        trainingAccuracy = miniBatchAcc
        _trainingAccuracy.append(trainingAccuracy)
        print("--trainingAccuracy {0}".format(trainingAccuracy))
        break
    nIt += 1

yPredict = nnc.FeedForward(trainingInput)
trainingAccuracy = nnc.Evaluate(yPredict, trainingOutput)
print("--Accuracy on Training Data {0}".format(trainingAccuracy))

testInput = np.load('test_inputs_standardized.npy')
yTest = nnc.FeedForward(testInput)
y = yTest.argmax(axis=1)
id = np.arange(1,len(y)+1)
print(id)
print(y)