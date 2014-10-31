import numpy as np
import utils
import NNClassifier as nnClassifier

trainingInput = np.load('standardized_train_inputs.npy')
trainingOutput = np.load('train_outputs.npy')

nFeatures = trainingInput.shape[1]

#trainingInput = trainingInput[:1000,:]
#trainingOutput = trainingOutput[:1000]

alpha, beta, nodes, nIter, nLayers = utils.RandomHyperParams(nFeatures)
print("alpha: {0}, beta: {1}, nodes: {2}, nIter: {3}, nLayers: {4}".format(alpha, beta, nodes, nIter, nLayers))

if nLayers == 1:
    nnc = nnClassifier.NeuralNetworkClassifier((nFeatures, nodes, 10))
else:
    nnc = nnClassifier.NeuralNetworkClassifier((nFeatures, nodes, nodes, 10))

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

    for nIt in range(nIter+1):
        for nBatchIndex in range(nTrainBatches):
            n = nnc.nBatchSize * nBatchIndex
            m = nnc.nBatchSize * (nBatchIndex + 1)
            xTrainBatch = xTrain[n:m,:]
            yTrainBatch = yTrain[n:m]
            miniBatchAcc = nnc.Learning(xTrainBatch, yTrainBatch, alpha, beta)

        if nIt == nIter:
            trainingAccuracy = miniBatchAcc
            _trainingAccuracy.append(trainingAccuracy)
            print("--trainingAccuracy {0}".format(trainingAccuracy))


    yPredict = nnc.FeedForward(xValid)
    validationAccuracy = nnc.Evaluate(yPredict, yValid)
    _validationAccuracy.append(validationAccuracy)

    print("--validationAccuracy {0}".format(validationAccuracy))

print("Training Accuracy {0} : Validation Accuracy {1}".format((np.mean(_trainingAccuracy)),np.mean(_validationAccuracy)))
#testInput = np.load('test_inputs_standardized.npy')
