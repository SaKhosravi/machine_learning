import numpy as np
from Perceptron.perceptron import MyPerceptron
from utils.plot_decision_boundaries import PlotDB
from utils.generate_data_set import GenerateDataSetBlob
from utils.loss import Loss
l=Loss()

gdb = GenerateDataSetBlob()
colors = {0: 'red', 1: 'blue'}

#make blobs
# X, y = gdb.make_blobs(n_samples=1000, centers=2, n_features=2,
#                       random_state=4,cluster_std=0.8)

#make mons
X, y=gdb.make_mons(500,0.15)
gdb.plot_dataSet(X, y, colors)

x_train, x_test, y_train, y_test = gdb.train_test_split(X,y,test_size=0.2)

# initialize
clf = MyPerceptron(50, learning_rate=0.001,
                   loss_function="MAE", evaluate_metric="accuracy")

# step2: fit
loss, accuracy = clf.fit(x_train, y_train)

# step3: predict
y_pred=clf.predict(x_test)
print("test accuracy",clf.evaluate(y_pred,y_test))


# plot train loss and train accuracy
l.plot_loss(loss,"perceptron,train loss","iteration","loss")
l.plot_loss(accuracy,"perceptron,train accuracy","iteration","accuracy")

# plot decision boundary
name = "perceptron in {} iter".format(50)
plt = PlotDB(clf, X, y, name)
plt.plot()



