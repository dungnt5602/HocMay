#Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv('iris.data.csv', names=names)
dt_Train = data[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
dt_Test = data[['class']]
X_train, X_test, y_train,  y_test = train_test_split(dt_Train, dt_Test, test_size=0.3, shuffle=False) # chia du lieu test va train theo ty le 3:7

kmeans5 = KMeans(n_clusters=5)
y_kmeans5 = kmeans5.fit_predict(X_train)

