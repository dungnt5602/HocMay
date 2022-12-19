import numpy as np

def unit_step_func(x):
    return np.where(x > 1 , 2, 1)

class PerceptronCustom:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted


import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import Perceptron;
from sklearn import metrics


# Load the data set
data = pd.read_csv('Skin_NonSkin.csv')
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=True)

X_train = dt_Train.iloc[:, :3].values
y_train = dt_Train.iloc[:, 3].values
X_test = dt_Test.iloc[:, :3].values
y_test = dt_Test.iloc[:, 3].values

prcptrn = PerceptronCustom(learning_rate=0.01, n_iters=10)
prcptrn.fit(X_train, y_train)
print(prcptrn.predict(X_test))

prtSk = Perceptron()
prtSk.fit(X_train, y_train)
print(prtSk.predict(X_test))

print(metrics.accuracy_score(y_test, prcptrn.predict(X_test)))
print(metrics.accuracy_score(y_test, prtSk.predict(X_test)))
