import math
from ann_util import between, make_matrix
from ann_util import deriv_logistic, logistic
from ann_util import deriv_hyperbolic_tangent, hyperbolic_tangent

use_bias = 1


class ANN:

    def __init__(self, layer_sizes, activation_fun='tanh'):
        """
        Initialize the network.
        :param activation_fun: tanh or logistic
        """
        self.layers = []
        self.learn_rate = 0.1

        self.squash = None
        self.deriv_squash = None
        if activation_fun == 'tanh':
            self.squash = hyperbolic_tangent
            self.deriv_squash = deriv_hyperbolic_tangent
        elif activation_fun == 'logistic':
            self.squash = logistic
            self.deriv_squash = deriv_logistic

        for l in range(len(layer_sizes)):
            layer_size = layer_sizes[l]
            prev_layer_size = 0 if l == 0 else layer_sizes[l - 1]
            layer = Layer(l, layer_size, prev_layer_size)
            self.layers.append(layer)

    def train(self, inputs, targets, n_epochs):
        """
        Train the network with the labeled inputs for a maximum number of epochs.
        """
        for epoch in range(0, n_epochs):

            epoch_error = 0
            for i in range(0, len(inputs)):

                self.set_input(inputs[i])
                self.forward_propagate()

                sample_error = self.update_error_output(targets[i])
                epoch_error += sample_error

                self.backward_propagate()
                self.update_weights()

            if epoch % 100 == 0:
                print(epoch, epoch_error)

    def predict(self, input):
        """
        Return the network prediction for this input.
        """
        self.set_input(input)
        self.forward_propagate()
        return self.get_output()

    def update_weights(self):
        """
        Update the weights matrix in each layer.
        """
        for l in range(1, len(self.layers)):
            for j in range(0, self.layers[l].n_neurons):
                for i in range(0, self.layers[l-1].n_neurons + use_bias):
                    out = self.layers[l-1].output[i]
                    err = self.layers[l].error[j]
                    self.layers[l].weight[i][j] += self.learn_rate * out * err

    def set_input(self, input_vector):
        input_layer = self.layers[0]

        for i in range(0, input_layer.n_neurons):
            input_layer.output[i + use_bias] = input_vector[i]

    def forward_propagate(self):
        """
        Propagate the input signal forward through the network.
        """
        # exclude the last layer
        for l in range(len(self.layers) - 1):

            src_layer = self.layers[l]
            dst_layer = self.layers[l + 1]

            for j in range(0, dst_layer.n_neurons):

                sum_in = 0

                for i in range(0, src_layer.n_neurons + use_bias):
                    sum_in += dst_layer.weight[i][j] * src_layer.output[i]

                dst_layer.input[j] = sum_in
                dst_layer.output[j + use_bias] = self.squash(sum_in)

    def get_output(self):
        output_layer = self.layers[-1]
        res = [0] * output_layer.n_neurons
        for i in range(0, len(res)):
            res[i] = output_layer.output[i + use_bias]

        return res

    def update_error_output(self, target_vector):
        sample_error = 0
        output_layer = self.layers[-1]
        for i in range(0, output_layer.n_neurons):
            neuron_output = output_layer.output[i + use_bias]
            neuron_error = target_vector[i] - neuron_output
            output_layer.error[i] = self.deriv_squash(output_layer.input[i]) * neuron_error
            sample_error += neuron_error * neuron_error
        sample_error *= 0.5
        return sample_error

    def backward_propagate(self):
        """
        Backprop. Propagate the error from the output layer backwards to the input layer.
        """
        for l in range(len(self.layers) - 1, 0, -1):
            src_layer = self.layers[l]
            dst_layer = self.layers[l - 1]

            for i in range(0, dst_layer.n_neurons):

                error = 0

                for j in range(0, src_layer.n_neurons):
                    error += src_layer.weight[i + use_bias][j] * src_layer.error[j]

                dst_layer.error[i] = self.deriv_squash(dst_layer.input[i]) * error


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
                # adjust initial weights proportional to prev layer size
                good_range = 1.0 / math.sqrt(prev_layer_size + 1)
                self.weight[i][j] = between(-good_range, good_range)

if __name__ == '__main__':

    # a logical function
    logic_ann = ANN([2, 2, 1])
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [[0.0], [1.0], [1.0], [0.0]]

    # train and predict
    logic_ann.train(inputs, targets, 20000)
    print("Predictions after training")
    for i in range(len(targets)):
        print(inputs[i], logic_ann.predict(inputs[i]))
