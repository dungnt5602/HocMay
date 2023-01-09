#Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
data = pd.read_csv('iris.data.csv', names=names)

data['Class'] = pd.Categorical(data["Class"])
data["Class"] = data["Class"].cat.codes
X = data.values[:, 0:4]
y = data.values[:, 4]
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)

from sklearn.metrics import classification_report

target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

print(classification_report(data['Class'],kmeans.labels_,target_names=target_names))