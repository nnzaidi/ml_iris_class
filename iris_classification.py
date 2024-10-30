import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = pd.read_csv("iris.csv")
print(iris.head(5))
print(iris.describe())
print('Target labels: ', iris['species'].unique())

# Scatter plot: Iris species according to sepal length and sepal width
# import plotly.express as px
fig = px.scatter(iris,
                 x = 'sepal_width',
                 y = 'sepal_length',
                 color = 'species')
fig.show()
# fig.write_image('Iris species according to sepal length and sepal width.png')

# Train ML model for iris classification task based on species by using KNN algorithm
x = iris.drop('species', axis=1)
y = iris['species']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(x_new)
print('Prediction: {}'.format(prediction))