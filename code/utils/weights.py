import numpy as np


class Weights:
    def __init__(self):
        pass

    def getRandomWeight(self, length):
        pass

    def getOnesWeight(self, length):
        return np.ones((length,), dtype=int)

    def getZerosVector(self, length):
        return np.zeros((length,), dtype="float32")

    def getOnesMatrix(self,length,width):
        return np.ones((length,width),dtype=int)

    def getOnesMatrix(self,length,width):
        return np.ones((length,width),dtype=int)


# w=Weights()
# a=w.getOnesMatrix(5,5)
# print(a)
# print(a.shape)