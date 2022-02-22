import numpy as np
from utils.distance import Distance

"""
simple knn implementation
"""
class KNN_Classifier(Distance):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        # self.distance = Distance()

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        y_preds = list()
        for sample in x_test:
            k_neighbors_label = self.get_k_neighbors_label(sample)
            # print(distances)
            # nearest_neighbor_ids = distances.argsort()[:self.k]
            # k_neighbors_label = self.y_train[nearest_neighbor_ids]
            y_preds.append(self.voting(k_neighbors_label))
        return y_preds

    def voting(self, labels):
        max_voting = np.bincount(labels).argmax()
        return max_voting

    def get_k_neighbors_label(self, x1):
        distances = list()
        for x2 in self.x_train:
            dist = self.minkowski(x1, x2, 2)
            distances.append(dist)
        distances= np.array(distances)
        nearest_neighbor_ids = distances.argsort()[:self.k]
        k_neighbors_label = self.y_train[nearest_neighbor_ids]
        return k_neighbors_label
