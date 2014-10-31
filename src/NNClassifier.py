import numpy as np
import utils

"----- Neural Network Classifier -----"
class NeuralNetworkClassifier:

    layerCount = 0
    nBatchSize = 500
    shape = None
    weights = []
    transferFuncs = []

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

        yPredit = self.FeedForward(input)
        accuracy = self.Evaluate(yPredit, target.T)

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

        return accuracy

    def Evaluate(self, yPredict, yActual):

        y = []

        for i in range(yActual.shape[0]):
            y.append(np.argmax(yPredict[i]))

        accuracy = np.sum(y == yActual)
        accuracy /= yActual.shape[0]

        return accuracy