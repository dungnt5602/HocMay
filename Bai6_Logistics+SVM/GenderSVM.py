"""
1) Bài toán dự đoán giới tính theo dữ liệu phân tích khuôn mặt
Cong viec chinh:
+ Su dung mo hinh Logistic Regression de du doan gioi tinh
+ Su dung mo hinh SVM (Support Vector Machine) de du doan gioi tinh

2) Mo ta tap du lieu
+ Du lieu gom 300 thuoc tinh
+ Y nghia thuoc tinh: khong biet ( ban quyen)
+ Co 1400 mau du lieu
+ Nhan cua du lieu la mot cot bao gom cac gia tri 1 va 2 => phan lop 1 or 2
"""
import numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
data = pd.read_csv('MyARgender.csv')
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False) # chia du lieu test va train theo ty le 3:7

# chia du lieu thanh cac cot du doan va nhan
X_train = dt_Train.iloc[:, :300].values
y_train = dt_Train.iloc[:, 300]
X_test = dt_Test.iloc[:, :300].values
y_test = dt_Test.iloc[:, 300]

y_train_cls = [1 if i >= 2 else -1 for i in y_train]
y_test_cls = [1 if i >= 2 else -1 for i in y_test]

# su dung thu vien
clfPoly = SVC(kernel='poly', degree = 3, gamma=1, C = 100)
clfPoly.fit(X_train, y_train_cls)
y_predPoly = clfPoly.predict(X_test)
print("Accugracy Poly: %.2f %%" %(100 * metrics.accuracy_score(y_test_cls, y_predPoly)))
print(y_predPoly)
"""
Poly:
+ C=100: 90,71%
+ C=10: 90,24%
+ C=1: 85,48%

Linear:
+ C=100: 89,79%
+ C=10: 87,62%
+ C=1: 84,52%

RBF:
+ C=100: 90,48%
+ C=10: 89,52%
+ C=1: 84,29%

Sigmoid:
+C=100: 85,24%
+C=10: 86,43%
+C=1: 83,33%
"""
# dung thuat toan

class CustomSVM:
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


y_train_cls = numpy.asarray(y_train_cls)
y_test_cls = numpy.asarray(y_test_cls)

clf = CustomSVM()
clf.fit(X_train, y_train_cls)
y_pred = clf.predict(X_test)
# print( metrics.accuracy_score(y_test_cls, y_pred))
print(y_pred)
# print(y_test_cls)