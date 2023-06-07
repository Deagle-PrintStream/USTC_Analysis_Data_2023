import math, random
import numpy as np
from sklearn.metrics import f1_score

#adjust learning rate dynamically, but not necessary in this project
def eta_propotion_reduce(eta0: float, epoch: int, n_epochs: int) -> float:
    ratio = 5
    new_eta = eta0 / (1 + epoch * ratio / n_epochs) 
    return new_eta

class NN:
    def __init__(
        self,
        input_dim=None,
        output_dim=None,
        hidden_layers=None,
        seed=1,
        weight_init=None,
    ):
        if (input_dim is None) or (output_dim is None) or (hidden_layers is None):
            raise Exception("Invalid arguments given!")
        self.input_dim = input_dim  # number of input nodes
        self.output_dim = output_dim  # number of output nodes
        self.hidden_layers = hidden_layers  # number of hidden nodes @ each layer
        self.network = self._build_network(seed=seed, weight_init=weight_init)

    # Train network
    def train(self, X, y, eta=0.5, n_epochs=200) -> tuple:
        acc_list = list()
        loss_list = list()
        f1_list = list()
        for epoch in range(n_epochs):
            for (x_, y_) in zip(X, y):
                self._forward_pass(x_)  # forward pass (update node["output"])
                yhot_ = self._one_hot_encoding(y_, self.output_dim)  # one-hot target
                self._backward_pass(yhot_)  # backward pass error (update node["delta"])
                eta_k = self._update_eta(eta, epoch, n_epochs)  # update eta dynamically
                self._update_weights(
                    x_, eta_k
                )  # update weights (update node["weight"])

            if epoch % (n_epochs / 10) == 0:
                acc, loss, f1_s = self._test(X, y)
                acc_list.append([epoch, acc])
                loss_list.append([epoch, loss])
                f1_list.append([epoch, f1_s])
                print("epoch : {:d} , Loss :{:.3f} F1:{:.3f}".format(epoch,loss,f1_s))
        return acc_list, loss_list, f1_list

    # Predict using argmax of logits
    def predict(self, X):
        ypred = np.array([np.argmax(self._forward_pass(x_)) for x_ in X], dtype=int)
        return ypred

    def _test(self, X, y) -> tuple:
        y_pred = self.predict(X)
        acc = round(np.sum(y == y_pred) / len(y), 2)
        loss = float(np.dot(y - y_pred, y - y_pred)) / len(y)
        f1_s = f1_score(y_pred, y, average="macro")
        return (acc, loss, f1_s)

    # Build fully-connected neural network (no bias terms)
    def _build_network(self, seed=1, weight_init=None):
        random.seed(seed)

        # Create a single fully-connected layer
        def _layer(input_dim, output_dim, weight_init=None):
            layer = []
            for i in range(output_dim):

                weights = [random.random() for _ in range(input_dim)]  # sample N(0,1)

                if weight_init != None:
                    if len(weight_init) != input_dim:
                        raise ValueError("not enough weight initialized")
                    weights = [i * j for i, j in zip(weights, weight_init)]
                node = {
                    "weights": weights,  # list of weights
                    "output": None,
                    "delta": None,
                }
                layer.append(node)
            return layer

        # Stack layers
        network = []
        if len(self.hidden_layers) == 0:
            network.append(_layer(self.input_dim, self.output_dim))
        else:
            network.append(_layer(self.input_dim, self.hidden_layers[0], weight_init))
            for i in range(1, len(self.hidden_layers)):
                network.append(_layer(self.hidden_layers[i - 1], self.hidden_layers[i]))
            network.append(_layer(self.hidden_layers[-1], self.output_dim))

        return network

    # update eta method
    def _update_eta(self, eta0: float, epoch: int, n_epochs: int) -> float:
        #return eta0  # we use a static constant eta
        return eta_propotion_reduce(eta0,epoch,n_epochs)

    # Forward-pass
    def _forward_pass(self, x):
        transfer = self._sigmoid
        x_in = x
        for layer in self.network:
            x_out = []
            for node in layer:
                node["output"] = transfer(self._dotprod(node["weights"], x_in))
                x_out.append(node["output"])
            x_in = x_out  # set output as next input
        return x_in

    # Backward-pass
    def _backward_pass(self, yhot):
        transfer_derivative = self._sigmoid_derivative
        n_layers = len(self.network)
        for i in reversed(range(n_layers)):  # traverse backwards
            if i == n_layers - 1:
                # Difference between logits and one-hot target
                for j, node in enumerate(self.network[i]):
                    err = node["output"] - yhot[j]
                    node["delta"] = err * transfer_derivative(node["output"])
            else:
                # Weighted sum of deltas from upper layer
                for j, node in enumerate(self.network[i]):
                    err = sum(
                        [
                            node_["weights"][j] * node_["delta"]
                            for node_ in self.network[i + 1]
                        ]
                    )
                    node["delta"] = err * transfer_derivative(node["output"])

    # Update weights
    def _update_weights(self, x, eta):
        for i, layer in enumerate(self.network):
            # Grab input values
            if i == 0:
                inputs = x
            else:
                inputs = [node_["output"] for node_ in self.network[i - 1]]
            # Update weights
            for node in layer:
                for j, input in enumerate(inputs):
                    node["weights"][j] += -eta * node["delta"] * input

    # Dot product
    def _dotprod(self, a, b):
        return sum([a_ * b_ for (a_, b_) in zip(a, b)])

    # Sigmoid
    def _sigmoid(self, x):
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except:
            pass

    # Sigmoid derivative
    def _sigmoid_derivative(self, sigmoid):
        return sigmoid * (1.0 - sigmoid)

    # One-hot encoding
    def _one_hot_encoding(self, idx, output_dim):
        x = np.zeros(output_dim, dtype=int)
        x[idx] = 1
        return x
