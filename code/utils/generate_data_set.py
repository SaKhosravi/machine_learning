import numpy as np
from pandas import DataFrame
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class GenerateDataSetBlob():
    def __init__(self, ):
        self.X = None
        self.y = None

    def make_blobs(self, n_samples, n_features, centers, random_state,cluster_std):
        X, y = make_blobs(n_samples=n_samples,
                          n_features=n_features,
                          centers=centers,
                          cluster_std=cluster_std,
                          random_state=random_state)
        self.X = np.array(X)
        self.y = np.array(y)
        return X, y

    def train_test_split(self, X, y, test_size):
        x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=0)

        return x_train, x_test, y_train, y_test

    def getDataFrame(self, columns, target_name):
        df = DataFrame(data=self.X, columns=columns)
        df[target_name] = self.y
        return df

    def plot_dataSet(self,X,y,colors):
        # scatter plot, dots colored by class value
        df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        fig, ax = plt.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key,color=colors[key])
        plt.show()
