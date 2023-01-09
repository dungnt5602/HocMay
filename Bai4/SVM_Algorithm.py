import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# Testing
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    data = pd.read_csv('Iris.csv')
    dt_Train = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    dt_Test = data[['Species']]
    X_train, X_test, y_train, y_test = train_test_split(dt_Train, dt_Test, test_size=0.3, shuffle=False)
    print(X_train)
    print(y_train)

    clf = SVM()
    clf.fit(X_train, y_train)
    # predictions = clf.predict(X)

    print(clf.w, clf.b)

