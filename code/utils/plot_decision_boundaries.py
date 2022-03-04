import numpy as np
from matplotlib import pyplot as plt


class PlotDB:
    def __init__(self, clf,X,y,model_name):
        self.clf=clf
        self.X=X
        self.y=y
        self.model_name=model_name

    def plot(self,):
        fig, ax = plt.subplots()
        # title for the plots
        title = ('Decision surface of '+self.model_name)
        # Set-up grid for plotting.
        X0, X1 = self.X[:, 0], self.X[:, 1]
        xx, yy = self.make_meshgrid(X0, X1)

        self.plot_contours(ax, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=self.y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_ylabel('y label here')
        ax.set_xlabel('x label here')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.legend()
        p=r"C:\Users\sadeg\Documents\my_calss\ml\machine_learning\code\knn\iamges"
        name="{}.png".format(self.model_name)
        # plt.savefig(name)
        plt.show()

    def make_meshgrid(self,x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(self,ax, xx, yy, **params):
        a=np.c_[xx.ravel(), yy.ravel()]
        Z = self.clf.predict(a)
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out





