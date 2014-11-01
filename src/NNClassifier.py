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
        #self._prevDeltaWeight = []

        for (x, y) in zip(layerSize[:-1], layerSize[1:]):
           self.weights.append(np.random.normal(scale=0.1, size = (y, x+1))) #initialize all weights to small random number
           #self._prevDeltaWeight.append(np.zeros((y, x+1)))

        lFuncs = []
        for i in range(self.layerCount):
            if i == self.layerCount - 1:
                lFuncs.append(utils.linear) #for output layer
            else:
                lFuncs.append(utils.sigmoid) #for hidden layers

        self.transferFuncs = lFuncs

    "----- Compute the output of all units in the network -----"
    def FeedForward(self, input):
        nSamples = input.shape[0]

        self._inputLayer = []
        self._outputLayer = []

        for index in range(self.layerCount):
            if index == 0:
                inputLayer = self.weights[0].dot(np.vstack([input.T, np.ones([1, nSamples])]))
            else:
                inputLayer = self.weights[index].dot(np.vstack([self._outputLayer[-1], np.ones([1, nSamples])]))

            self._inputLayer.append(inputLayer)
            "Compute output using appropriate transfer function"
            self._outputLayer.append(self.transferFuncs[index](inputLayer))

        return self._outputLayer[-1].T

    "----- TrainingEpoch: back propagation and gradient descent -----"
    def TrainingEpoch(self, input, target, lenMiniBatch, alpha = 0.1, lmbda = 0.0):#beta = 0.5):
        delta = []
        nSamples = input.shape[0]

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
                delta.append(delta_pullback[:-1, :] * self.transferFuncs[index](self._inputLayer[index], False))

        "Update all the weights in the network"
        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index

            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1, nSamples])])
            else:
                layerOutput = np.vstack([self._outputLayer[index - 1], np.ones([1, self._outputLayer[index - 1].shape[1]])])

            outputXdelta = layerOutput[None, :, :].transpose(2, 0, 1) * delta[delta_index][None, :, :].transpose(2, 1, 0)
            curDeltaWeight = np.sum(outputXdelta, axis = 0)
            deltaWeight = (1-alpha*(lmbda/nSamples))*self.weights[index] - (alpha/lenMiniBatch) * curDeltaWeight #+ (beta/lenMiniBatch) * self._prevDeltaWeight[index]

            self.weights[index] = deltaWeight

            #._prevDeltaWeight[index] = deltaWeight

        return accuracy

    def Evaluate(self, yPredict, yActual):

        y = []

        for i in range(yActual.shape[0]):
            y.append(np.argmax(yPredict[i]))

        accuracy = np.sum(y == yActual)
        accuracy /= yActual.shape[0]

        return accuracy

'''
    "----- Update Mini Batch -----"
    def UpdateMiniBatch(self, miniBatch, alpha):

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in miniBatch:
            delta_nabla_b, delta_nabla_w = self.TrainingEpoch(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
'''