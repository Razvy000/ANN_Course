from ann_util import between
from ann_util import make_matrix

use_bias = 1


class ANN:

    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def update_weights(self):
        pass


class Layer:

    def __init__(self, id, layer_size, prev_layer_size):

        self.id = id
        self.n_neurons = layer_size
        self.bias_val = 1

        self.input = [0] * self.n_neurons

        self.output = [0] * (self.n_neurons + use_bias)
        self.output[0] = self.bias_val

        self.error = [0] * self.n_neurons

        self.weight = make_matrix(prev_layer_size + use_bias, self.n_neurons)
        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                self.weight[i][j] = between(-0.2, 0.2)
