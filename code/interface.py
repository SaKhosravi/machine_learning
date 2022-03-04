from abc import abstractmethod

from utils.weights import Weights
from utils.loss import Loss
from utils.activations import Activations
from utils.metric import Metric
class Interface(Weights,Loss,Activations,Metric):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self,x_train, y_train):
        pass
    @abstractmethod
    def predict(self,x_test):
        pass
