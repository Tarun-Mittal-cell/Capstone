from os import name

import numpy as np

class MLP:

    def _init_(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # initiate random weights

        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(w)


    def forward_propogate(self, inputs):
        activations = inputs
        for w in self.weights:
            # calculate the net inputs for a given layer
            net_inputs = np.dot(activations, w)
            #calculate the activations
            activations = self.sigmoid(net_inputs)
        return activations

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

if name == "_main_":

    mlp = MLP()

    inputs = np.random.rand(mlp.num_inputs)

    outputs = mlp.forward_propogate(inputs)

    print("The network output is: {}".format(inputs))
    print("The network output is: {}" .format(outputs))




