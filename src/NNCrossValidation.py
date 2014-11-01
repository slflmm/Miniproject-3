import numpy as np
import utils
import random
import NNClassifier as nnClassifier

trainingInput = np.load('standardized_train_inputs.npy')
trainingOutput = np.load('train_outputs.npy')

nFeatures = trainingInput.shape[1]

#trainingInput = trainingInput[:100,:]
#trainingOutput = trainingOutput[:100]
#for i in range(15):
#    alpha, beta, nodes, nIter, nLayers = utils.RandomHyperParams(nFeatures)
#    print("alpha = {0}, beta = {1}, nodes = {2}, nIter = {3}, nLayers = {4}".format(alpha, beta, nodes, nIter, nLayers))
print("RUN 5...")

lmbda = 5
alpha = 0.1
beta = 0.3
nodes = 31
nIter = 160
nLayers = 2
print("lmbda = {0}, alpha = {1}, nodes = {2}, nIter = {3}, nLayers = {4}".format(lmbda, alpha, nodes, nIter, nLayers))

if nLayers == 1:
    nnc = nnClassifier.NeuralNetworkClassifier((nFeatures, nodes, 10))
    print("nnc({0}, {1}, {2})".format(nFeatures, nodes, 10))
else:
    nodes2 = int(nodes / 2)
    nnc = nnClassifier.NeuralNetworkClassifier((nFeatures, nodes, nodes2, 10))
    print("nnc({0}, {1}, {2}, {3})".format(nFeatures, nodes, nodes2, 10))

crossValid = utils.CrossValidation(trainingInput, trainingOutput, 5)

_trainingAccuracy = []
_validationAccuracy = []

for kFold in range(5):
    print("fold {0}".format(kFold))

    xTrain, yTrain, xValid, yValid = crossValid.next()

    nTrainBatches = int(xTrain.shape[0] / nnc.nBatchSize)

    #print("---#trainBatches {0}".format(nTrainBatches))

    trainingAccuracy = 0
    validationAccuracy = 0

    xy = np.insert(xTrain, nFeatures, yTrain, axis = 1)

    for nIt in range(nIter):
        random.shuffle(xy)
        miniBatches = [
            xy[k : k+nnc.nBatchSize]
            for k in range(0, nTrainBatches, nnc.nBatchSize)]
        for miniBatch in miniBatches:
            lenMiniBatch = len(miniBatch)
            xMiniBatch = miniBatch[:, :-1]
            yMiniBatch = miniBatch[:, -1]
            miniBatchAcc = nnc.TrainingEpoch(xMiniBatch, yMiniBatch, lenMiniBatch, alpha, lmbda)#, beta)

        if nIt == nIter - 1:
            trainingAccuracy = miniBatchAcc
            _trainingAccuracy.append(trainingAccuracy)
            print("--trainingAccuracy {0}".format(trainingAccuracy))

    yPredict = nnc.FeedForward(xValid)
    validationAccuracy = nnc.Evaluate(yPredict, yValid)
    _validationAccuracy.append(validationAccuracy)

    print("--validationAccuracy {0}".format(validationAccuracy))

print("Training Accuracy {0} : Validation Accuracy {1}".format((np.mean(_trainingAccuracy)),np.mean(_validationAccuracy)))


#testInput = np.load('test_inputs_standardized.npy')
'''
        for nBatchIndex in range(nTrainBatches):
            n = nnc.nBatchSize * nBatchIndex
            m = nnc.nBatchSize * (nBatchIndex + 1)
            xTrainBatch = xTrain[n:m,:]
            yTrainBatch = yTrain[n:m]
            miniBatchAcc = nnc.TrainingEpoch(xTrainBatch, yTrainBatch, alpha, beta)

        if nIt == nIter:
            trainingAccuracy = miniBatchAcc
            _trainingAccuracy.append(trainingAccuracy)
            print("--trainingAccuracy {0}".format(trainingAccuracy))

        trainingAccuracy = nnc.TrainingEpoch(xTrain, yTrain, alpha, beta)
        if nIt == nIter:
            _trainingAccuracy.append(trainingAccuracy)
            print("--trainingAccuracy {0}".format(trainingAccuracy))
        '''