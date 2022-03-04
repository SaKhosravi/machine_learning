import numpy as np
from utils.weights import Weights
from utils.activations import Activations
from utils.loss import Loss

class MyPerceptron(Activations, Weights,Loss):
    def __init__(self, iteration, learning_rate):
        self.iteration = iteration
        self.lr=learning_rate
        self.weights = None
        self.bias=0
        self.loss=list()

    def predict(self, x_test):
        x_test=np.array(x_test)
        y_preds=[]
        for x_i in x_test:
            output=np.dot(x_i.T,self.weights)+self.bias
            y_pred=self.sign(output)
            y_preds.append(y_pred)

        return np.array(y_preds)

    def fit(self, x_train, y_train):
        # initial weights
        x_train, y_train = np.array(x_train), np.array(y_train)
        self.weights = self.getZerosVector(np.shape(x_train)[1]).astype(np.float32)

        for iter in range(self.iteration):
            y_preds = []
            for index, x_i in enumerate(x_train):
                # res = np.dot(self.weights, row)
                output = np.dot(x_i.T, self.weights) + self.bias
                y_hat=self.sign(output)
                y_preds.append(y_hat)
                y = y_train[index]

                #update weights
                if y - y_hat !=0:
                    update_rate= self.lr * (y-y_hat)
                    self.weights += update_rate * x_i
                    self.bias += update_rate * 1

            l= self.MAE(y_train,y_preds)
            self.loss.append(l)
            # print(l)
            # print("iteration {} loss:{}".format((iter,l)))



    def getSign(self, x):
        output=np.dot(x.T,self.weights)+self.bias
        if output >  0:
            return 1
        else:
            return 0
