"""
activation function list
https://en.wikipedia.org/wiki/Activation_function

"""
import numpy as np
from utils.weights import Weights


class Activations:
    def __init__(self):
        pass

    def Identity(self, x):
        return x

    def sign(self, x):
        if x > 0:
            return 1
        else:
            return 0

    # sign or Binary step
    def binary_step(self, x):
        return np.where(x >= 0, 1, 0)

    # Logistic, sigmoid, or soft step
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Hyperbolic tangent
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    """Rectified linear unit"""

    def ReLU(self, x):
        return x * (x > 0)

    # Gaussian Error Linear Unit
    # def GELU(self,x):
    #     x=np.power(x,2)
    #     return np.exp(-x)

    def softplus(self, x):
        return np.log(1 + np.exp(x))

    # Exponential linear unit (ELU)
    def ELU(self, x, alpha):
        return np.where(x <= 0, alpha * (np.exp(x) - 1), x)

    # Scaled exponential linear unit ,with parameters alpha=1.67326 , lambda=1.0507
    def SELU(self, x, lamda=1.0507, alpha=1.67326):
        return np.where(x < 0, lamda * alpha * (np.exp(x) - 1), lamda * x)

    """
    when alpha == 0.01: Leaky rectified linear unit (Leaky ReLU)
    when alpha != 0.01:Parametric rectified linear unit (PReLU)
    """

    def LeakyRelu(self, x, alpha=0.01):
        return np.where(x < 0, alpha * x, x)

    # Sigmoid linear unit (SiLU)
    def SiLU(self, x):
        return x / (1 + np.exp(-x))

    def softmax(self, x):
        e = np.exp(x)
        return e / np.sum(e)

# x=np.array([-2, -1,0,1,2,3])
# a=Activations()
# print(a.tanh(x))
