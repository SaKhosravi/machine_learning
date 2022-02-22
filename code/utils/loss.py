import numpy as np
import matplotlib.pyplot as plt


class Loss:
    def __init__(self):
        pass

    # mean absolute error
    def MAE(self, y_true, y_hat):
        y_true, y_hat = np.array(y_true), np.array(y_hat)
        mae = np.sum(np.abs(y_true - y_hat))
        mae = mae / len(y_true)
        return mae

    # mean squared error
    def MSE(self, y_true, y_hat):
        y_true, y_hat = np.array(y_true), np.array(y_hat)
        mse = np.sum(np.power((y_true - y_hat), 2))
        mse = mse / len(y_true)
        return mse

    def plot_loss(self, loss, title, xlabel, ylabel):
        loss = np.array(loss)
        x = np.arange(1, len(loss) + 1)
        plt.plot(x, loss)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
