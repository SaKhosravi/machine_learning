import numpy as np

from Perceptron.perceptron import MyPerceptron
from utils.plot_decision_boundaries import PlotDB
from utils.generate_data_set import GenerateDataSetBlob
from utils.loss import Loss
from sklearn.metrics import accuracy_score

gdb = GenerateDataSetBlob()
X, y = gdb.make_blobs(n_samples=1000, centers=2, n_features=2,
                      random_state=1,cluster_std=2.8)
colors = {0: 'red', 1: 'blue'}
gdb.plot_dataSet(X, y, colors)
x_train, x_test, y_train, y_test = gdb.train_test_split(X,y,test_size=0.2)

l=Loss()

#initialize
clf = MyPerceptron(100, learning_rate=0.001)
#fit
clf.fit(x_train, y_train)
#predict
y_pred=clf.predict(x_test)
print("test loss", l.MAE(y_test, y_pred))
print("train loss", clf.loss[len(clf.loss)-1])
#plot decision boundry
plt = PlotDB(clf, X, y,"Perceptron")
plt.plot()

#plot train loss
l.plot_loss(clf.loss,"Perceptron train loss","iteration","loss")
