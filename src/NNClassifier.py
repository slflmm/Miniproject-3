import numpy as np

"----- Transfer Functions -----"
def linear(x, firstLayer = True):
    if firstLayer:
        return x
    else:
        return 1.0

def sigmoid(x, firstLayer = True):
    if firstLayer:
        return 1 / (1 + np.exp(-x))
    else:
        out = sigmoid(x)
        return out * (1.0 - out)

def gaussian(x, firstLayer = True):
     if firstLayer:
         return np.exp(-x**2)
     else:
         return -2 * x * np.exp(-x**2)

def tanh(x, firstLayer = True):
     if firstLayer:
         return np.tanh(x)
     else:
         return 1.0 - np.tanh(x)**2

"----- Neural Network Classifier -----"
class NeuralNetworkClassifier:

    layerCount = 0
    shape = None
    weights = []
    transferFuncs = []

    "----- Initialization -----"
    def __init__(self, layerSize, layerFuncs = None):

        self.layerCount = len(layerSize) - 1
        self.shape = layerSize

        if layerFuncs is None: #default
            lFuncs = []
            for i in range(self.layerCount):
                if i == self.layerCount - 1:
                    lFuncs.append(linear) #for output layer
                else:
                    lFuncs.append(sigmoid) #for hidden layers
        else:
            if len(layerSize) != len(layerFuncs):
                raise ValueError("Incompatible list of transfer functions.")
            elif layerFuncs[0] is not None:
                raise ValueError("Input layer cannot have a transfer function.")
            else:
                lFuncs = layerFuncs[1:]

        self.transferFuncs = lFuncs

        self._layerInput = []
        self._layerOutput = []
        self._previousWeightDelta = []

        for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):
           self.weights.append(np.random.normal(scale=0.01, size = (l2, l1+1))) #initialize all weights to small random number
           self._previousWeightDelta.append(np.zeros((l2, l1+1)))

    "----- Compute the output of all units in the network -----"
    def FeedForward(self, input):
        nSamples = input.shape[0]

        self._layerInput = []
        self._layerOutput = []

        for index in range(self.layerCount):
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, nSamples])])) #compute w.x for the first hidden layer
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, nSamples])])) #compute w.x for other hidden layers

            self._layerInput.append(layerInput)
            "Compute output using appropriate transfer function"
            self._layerOutput.append(self.transferFuncs[index](layerInput))

        return self._layerOutput[-1].T

    "----- Learning Function: back propagation and gradient descent -----"
    def Learning(self, input, target, alpha = 0.1, beta = 0.5):
        delta = []
        nSamples = input.shape[0]

        self.FeedForward(input)

        "Compute delta updates in decreasing order of the layers"
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta**2)
                delta.append(output_delta * self.transferFuncs[index](self._layerInput[index], False))
            else:
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :] * self.transferFuncs[index](self._layerInput[index], False))

        "Update all the weights in the network"
        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index

            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1, nSamples])])
            else:
                layerOutput = np.vstack([self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1].shape[1]])])

            OxDelta = layerOutput[None, :, :].transpose(2, 0, 1) * delta[delta_index][None, :, :].transpose(2, 1, 0)
            curWeightDelta = np.sum(OxDelta, axis = 0)

            weightDelta = alpha * curWeightDelta + beta * self._previousWeightDelta[index]

            self.weights[index] -= alpha * weightDelta

            self._previousWeightDelta[index] = weightDelta

        return error

if __name__ == "__main__":
    trainingInput = np.load('train_inputs.npy')
    trainingOutput = np.load('train_outputs.npy')

    lFuncs = [None, sigmoid, sigmoid, linear]

    nFeatures = trainingInput.shape[1]

    #test on first 10 samples
    trainingInput = trainingInput[:10,:]
    trainingOutput = trainingOutput[:10]

    nnc = NeuralNetworkClassifier((nFeatures, 50, 50, 1), lFuncs)

    lnMax = 100000
    lnErr = 1e-5
    for i in range(lnMax + 1):
        err = nnc.Learning(trainingInput, trainingOutput)
        if i % 2500 == 0:
            print("Iteration {0}\tError: {1:0.6f}".format(i, err))
            if err < lnErr:
                print("Minimum error reached at iteration {0}".format(i))
                break

    testInput = np.load('test_inputs.npy')
    testInput = testInput[:10,:]
    output = nnc.FeedForward(testInput)
    for i in range(trainingInput.shape[0]):
        print("Output: {0}".format(output[i]))
