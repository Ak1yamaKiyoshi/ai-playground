from logging import LogRecord
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from pprint import pprint

class ActivationFunction:
    def __init__(self, function:str) -> None:
        self.fn, self.df_fn = {
            "sigmoid":(self.sigmoid, self.df_sigmoid)
        }[function]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def df_sigmoid(cls, x):
        return cls.sigmoid(x) * (1 - cls.sigmoid(x))

    def __call__(self, x):
        return self.fn(x)

    def df(self, x):
        return self.df_fn(x)


class LossFunction:
    pass


class MLP:
    def __init__(self) -> None:
        self.activation_function = ActivationFunction('sigmoid')
        self.weights = None
        self.loss_function = lambda y_pred, y_true: np.mean((y_true - y_pred) ** 2)
        self.loss_function_df = lambda y_pred, y_true: 2*(y_pred - y_true)

    def fit(self, X, y, hidden_layers:list=[3, 2, 2]):
        """ Example: [[0, 0], [1, 0], [0, 1]], [[0], [1], [1]] """
        X, y= np.array(X), np.array(y)
        n_features = X.shape[1]
        n_output_nodes = y.shape[1]
        
        self.weights_hidden = []
        self.biases_hidden = []
        prev_features = n_features
        for n_nodes in hidden_layers:
            weight = np.random.rand(prev_features, n_nodes)
            self.weights_hidden.append(weight)
            self.biases_hidden.append(np.zeros(n_nodes))
            prev_features = n_nodes
            
        self.bias_output = np.zeros(n_output_nodes)
        self.weights_outputs = np.random.rand(prev_features, n_output_nodes)
        return self


    def backpropagation(self, X, y, lr=0.3):
        X = np.array(X)
        y = np.array(y)

        outputs = []

        # forward pass 
        prev_inputs = X
        i = 1
        for  layer in self.weights_hidden:
            z = prev_inputs @ layer
            a = self.activation_function(z)
            outputs.append({"neurons":z, "activations":a, "layer": i - 1})
            prev_inputs = a
            i += 1

        z = prev_inputs @ self.weights_outputs
        a = self.activation_function(z)
        outputs.append({"neurons":z, "activations":a, "layer": i-1})

        loss = self.loss_function(a, y)
        dloss = self.loss_function_df(a, y)
        
        unpack_layer = lambda layer: (layer['layer'], layer['activations'], layer['neurons'])

        df = np.dot([dloss],  self.activation_function.df(a))
        wgrad = [df]
        weights = self.weights_outputs
        for layer in outputs[::-1][1:]:
            idx, al, zl = unpack_layer(layer)
            df = np.dot(self.activation_function.df(zl), np.sum(np.dot(df, weights)))
            weights -= lr * np.dot(df, al)
            weights = self.weights_hidden[idx].T
            wgrad.append(df)
        
        return loss

    def predict(self, X):
        X = np.array(X)

        prev_inputs = X
        for layer in self.weights_hidden:
            layer_inputs = prev_inputs @ layer
            layer_outputs = self.activation_function(layer_inputs)
            prev_inputs = layer_outputs

        output_inputs = prev_inputs @ self.weights_outputs
        outputs = self.activation_function(output_inputs)

        return outputs

def get_datasets(path="./data.csv"):
    """ train_X, test_X, train_y, test_y """
    df = pd.read_csv('./data.csv',  )
    X = df.drop("label", axis=1)
    y = df['label']
    return train_test_split(X, y, train_size=0.8)


xor_X = [
    [0, 0], 
    [0, 1],
    [1, 0],
    [1, 1]
]
xor_y = [
    [0],
    [1],
    [1],
    [0]
]


mlp = MLP()
mlp.fit(xor_X, xor_y)

for i in range(4000):
    loss = 0
    lr = 2e-3
    for X, y in zip(xor_X, xor_y):
        loss += mlp.backpropagation(X, y, lr)
    lr *= 0.95
    print(loss)

print(mlp.predict(xor_X[0]))
print(mlp.predict(xor_X[1]))
print(mlp.predict(xor_X[2]))
print(mlp.predict(xor_X[3]))
#print(mlp.predict(i))


#https://medium.com/@tiago.tmleite/neural-networks-multilayer-perceptron-and-the-backpropagation-algorithm-a5cd5b904fde#:~:text=The%20idea%20of%20the%20backpropagation,reaching%20the%20input%20layer%20ofhttps://medium.com/@tiago.tmleite/neural-networks-multilayer-perceptron-and-the-backpropagation-algorithm-a5cd5b904fde#:~:text=The%20idea%20of%20the%20backpropagation,reaching%20the%20input%20layer%20of