from abc import abstractmethod

from utils.weights import Weights
from utils.loss import Loss
from utils.activations import Activations

class Interface(Weights,Loss,Activations):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self,x_train, y_train):
        pass
    @abstractmethod
    def predict(self,x_test):
        pass
