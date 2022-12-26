import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('Bai3_car.csv', header=None)
y = data[6]

from sklearn.preprocessing import OrdinalEncoder

buying_price_category = ['low', 'med', 'high', 'vhigh']
maint_cost_category = ['low', 'med', 'high', 'vhigh']
doors_category = ['2', '3', '4', '5more']
person_capacity_category = ['2', '4', 'more']
lug_boot_category = ['small', 'med', 'big']
safety_category = ['low', 'med', 'high']

all_categories = [buying_price_category, maint_cost_category,doors_category,person_capacity_category,lug_boot_category,safety_category]
oe = OrdinalEncoder(categories= all_categories)
X = oe.fit_transform(data[[0, 1, 2, 3, 4, 5]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

DT_classifier = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=10)
DT_classifier.fit(X_train, y_train)

y_pred = DT_classifier.predict(X_test)
print(metrics.accuracy_score(y_pred, y_test))
