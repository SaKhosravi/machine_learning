import numpy as np
from utils.distance import Distance
from interface import Interface

"""
simple Knn implementation
"""
class KNN_Classifier(Interface,Distance):
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
        print(np.shape(x_test))
        for index, sample in enumerate( x_test):
            k_neighbors_label = self.get_k_neighbors_label(sample)
            # print(distances)
            # nearest_neighbor_ids = distances.argsort()[:self.k]
            # k_neighbors_label = self.y_train[nearest_neighbor_ids]
            y_pred=self.voting(k_neighbors_label)
            y_preds.append(y_pred)

        return np.array(y_preds)

    def loss(self,y_true,y_pred):
        loss_value=self.MAE(y_true,y_pred)
        return loss_value

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


