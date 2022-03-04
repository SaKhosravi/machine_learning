import numpy as np
from interface import Interface
from utils.loss import Loss
from utils.metric import Metric


class MyPerceptron(Interface):
    def __init__(self, iteration, learning_rate, loss_function, evaluate_metric):
        # hyper parameter
        self.iteration = iteration
        self.lr = learning_rate

        self.loss = getattr(globals()["Loss"](), loss_function)
        self.evaluate = getattr(globals()["Metric"](), evaluate_metric)

        self._weights = None
        self._bias = 0

        self.training_loss = list()
        self.training_evaluate = list()

    def predict(self, x_test):
        x_test = np.array(x_test)
        y_preds = list()
        for x_i in x_test:
            output = np.dot(x_i.T, self._weights) + self._bias
            y_pred = self.sign(output)
            y_preds.append(y_pred)

        return np.array(y_preds)

    def fit(self, x_train, y_train):
        x_train, y_train = np.array(x_train), np.array(y_train)

        # initial weights
        self._weights = self.getZerosVector(np.shape(x_train)[1]).astype(np.float32)

        for iter in range(self.iteration):
            y_preds = []
            for index, x_i in enumerate(x_train):
                # res = np.dot(self.weights, row)
                output = np.dot(x_i.T, self._weights) + self._bias
                y_hat = self.sign(output)
                y_preds.append(y_hat)
                y = y_train[index]

                # update weights
                if y - y_hat != 0:
                    update_rate = self.lr * (y - y_hat)
                    self._weights += update_rate * x_i
                    self._bias += update_rate * 1

            # loss and evaluation after each iteration
            self.training_loss.append(self.loss(y_train, y_preds))
            self.training_evaluate.append(self.evaluate(y_train, y_preds))

        return np.array(self.training_loss), np.array(self.training_evaluate)

    def getSign(self, x):
        output = np.dot(x.T, self._weights) + self._bias
        if output > 0:
            return 1
        else:
            return 0
