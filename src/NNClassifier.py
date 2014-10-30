import numpy as np
import math
import random
import utils

"----- Neural Network Classifier -----"
class NeuralNetworkClassifier:

    layerCount = 0
    nFeatures = 2303
    shape = None
    weights = []
    transferFuncs = []

    alphas = [0.1, 0.01, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.000005]
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    _nodes = math.floor(math.log2(nFeatures))
    nodesPerLayer = [_nodes+10, _nodes+20, _nodes+30, _nodes+40, _nodes+50, _nodes+60, _nodes+70, _nodes+80, _nodes+90, _nodes+100]
    iterations = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    layers = [1, 2]

    "----- Initialization -----"
    def __init__(self, layerSize):

        self.layerCount = len(layerSize) - 1
        self.shape = layerSize

        self._inputLayer = []
        self._outputLayer = []
        self._prevDeltaWeight = []

        for (x, y) in zip(layerSize[:-1], layerSize[1:]):
           self.weights.append(np.random.normal(scale=0.1, size = (y, x))) #initialize all weights to small random number
           self._prevDeltaWeight.append(np.zeros((y, x)))

        lFuncs = []
        for i in range(self.layerCount):
            if i == self.layerCount - 1:
                lFuncs.append(utils.linear) #for output layer
            else:
                lFuncs.append(utils.sigmoid) #for hidden layers

        self.transferFuncs = lFuncs

    "----- Compute the output of all units in the network -----"
    def FeedForward(self, input):

        self._inputLayer = []
        self._outputLayer = []

        for index in range(self.layerCount):
            if index == 0:
                inputLayer = self.weights[0].dot(input.T)
            else:
                inputLayer = self.weights[index].dot(self._outputLayer[-1])

            self._inputLayer.append(inputLayer)
            "Compute output using appropriate transfer function"
            self._outputLayer.append(self.transferFuncs[index](inputLayer))

        return self._outputLayer[-1].T

    "----- Learning Function: back propagation and gradient descent -----"
    def Learning(self, input, target, alpha = 0.1, beta = 0.5):
        delta = []

        self.FeedForward(input)

        "Compute delta updates in decreasing order of the layers"
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                output_delta = self._outputLayer[index] - target.T
                error = np.sum(output_delta**2)
                delta.append(output_delta * self.transferFuncs[index](self._inputLayer[index], False))
            else:
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:, :] * self.transferFuncs[index](self._inputLayer[index], False))

        "Update all the weights in the network"
        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index

            if index == 0:
                layerOutput = input.T
            else:
                layerOutput = self._outputLayer[index - 1]

            outputXdelta = layerOutput[None, :, :].transpose(2, 0, 1) * delta[delta_index][None, :, :].transpose(2, 1, 0)
            curDeltaWeight = np.sum(outputXdelta, axis = 0)

            deltaWeight = alpha * curDeltaWeight + beta * self._prevDeltaWeight[index]

            L1 = alpha * deltaWeight
            L2 = alpha**2 * deltaWeight

            self.weights[index] -= L1
            self.weights[index] -= L2

            self._prevDeltaWeight[index] = deltaWeight

        return error

    def RandHyperParams(self):

        alpha = random.choice(self.alphas)
        beta = random.choice(self.betas)
        nodes = random.choice(self.nodesPerLayer)
        iters = random.choice(self.iterations)
        layers = random.choice(self.layers)
        print(alpha, beta, nodes, iters, layers)
        return alpha, beta, nodes, iters, layers

    #def RunCrossValidation(self, trainingInput, trainingOutput, 5):

    #    cv = utils.CrossValidation(trainingInput, trainingOutput, 5)


if __name__ == "__main__":
    trainingInput = np.load('train_inputs.npy')
    trainingOutput = np.load('train_outputs.npy')

    #test on first 10 samples
    trainingInput = trainingInput[:100,:]
    trainingOutput = trainingOutput[:100]

    nFeatures = trainingInput.shape[1]

    nnc = NeuralNetworkClassifier((nFeatures, 20, 10))

    #nnc.RunCrossValidation(trainingInput, trainingOutput, 5)

    for i in range(15):
        alpha, beta, nodes, iters, layers = nnc.RandHyperParams()

        cv = utils.CrossValidation(trainingInput, trainingOutput, 5)

        for j in range(5):
            xTrain, yTrain, xValid, yValid = cv.next()
            if layers == 1:
                nnc = NeuralNetworkClassifier((nFeatures, nodes, 10))
            else:
                nnc = NeuralNetworkClassifier((nFeatures, nodes, nodes, 10))

            for i in range(iters + 1):
                err = nnc.Learning(xTrain, yTrain, alpha, beta)

    testOutput = nnc.FeedForward(trainingInput)
    for i in range(trainingInput.shape[0]):
        print("{0}: {1}".format(i, np.argmax(testOutput[i])))

    print("-------------")
    testInput = np.load('test_inputs.npy')
    testInput = testInput[:10,:]
    testOutput = nnc.FeedForward(testInput)
    for i in range(testInput.shape[0]):
        print("{0}: {1}".format(i, np.argmax(testOutput[i])))
