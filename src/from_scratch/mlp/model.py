import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split


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


    def fit(self, X, y, hidden_layers:list=[105, 2]):
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


    def backpropagation(self, X, y):
        X = np.array(X)
        y = np.array(y)

        predicted = self.predict(X)
        loss = self.loss_function(y, predicted)

        dL = y - predicted 
        self.activation_function.df(predicted)

        for weight, bias in list(zip(self.weights_hidden, self.biases_hidden))[::-1]:
            # weight -= partial derrivative of loss function in respect of weights * learning rate  
            # bias -= partial derrivative of loss function in respect of bias * learning rate 
            pass
        #print(outputs, loss)


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

print(MLP().fit([[0, 0], [1, 0], [0, 1]], [[0], [1], [1]]).backpropagation([[1, 0]], [[1]]))
