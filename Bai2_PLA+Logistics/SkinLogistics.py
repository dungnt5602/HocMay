import numpy as np
import pandas as pd
from sklearn import metrics


class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    # chuan bi du lieu
    data = pd.read_csv('Skin_NonSkin.csv') # doc du lieu tu file excel
    dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=True) # phan chia du lieu test va train

    # chia du lieu thanh cac cot du doan va nhan
    X_train = dt_Train.iloc[:, :3].values
    y_train = dt_Train.iloc[:, 3].values
    X_test = dt_Test.iloc[:, :3].values
    y_test = dt_Test.iloc[:, 3].values

    y_train_cls = [1 if i >= 2 else 0 for i in y_train]
    y_test_cls = [1 if i >= 2 else 0 for i in y_test]

    # Tao object kieu LogisticRegCus
    regressor = LogisticRegressionCustom(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train_cls) # huan luyen mo hinh theo thuat toan
    # kiem thu mo hinh theo thuat toan
    predictions = regressor.predict(X_test)
    print(predictions)
    print("Algorithm:", metrics.accuracy_score(y_test_cls, predictions))

    # Dung thu vien
    model1 = LogisticRegression() # khai bao mo hinh thu vien
    model1.fit(X_train, y_train_cls) # huan luyen mo hinh su dung thu vien

    # tinh toan do chinh xac cua mo hinh su dung thu vien
    y_predL = model1.predict(X_test)
    print(y_predL)
    print("Lib:", metrics.accuracy_score(y_test_cls, y_predL))

