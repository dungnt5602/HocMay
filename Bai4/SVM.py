
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


data = pd.read_csv('Iris.csv')
dt_Train = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
dt_Test = data[['Species']]
X_train, X_test, y_train,  y_test = train_test_split(dt_Train, dt_Test, test_size=0.3, shuffle=False) # chia du lieu test va train theo ty le 3:7

# chia du lieu thanh cac cot du doan va nhan
# su dung thu vien
clfPoly = SVC(kernel='poly', degree = 3, gamma=1, C = 100)
clfPoly.fit(X_train, y_train)
y_predPoly = clfPoly.predict(X_test)
print("Accugracy Poly: %.2f %%" %(100 * metrics.accuracy_score(y_test, y_predPoly)))

