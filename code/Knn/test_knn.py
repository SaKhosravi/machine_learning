from sklearn import neighbors
from sklearn.metrics import accuracy_score

from utils.generate_data_set import GenerateDataSetBlob
from utils.plot_decision_boundaries import PlotDB

gdb = GenerateDataSetBlob()
# X, y = gdb.make_blobs(n_samples=150, n_features=2, centers=3,
#                       random_state=0, cluster_std=1.5)
# colors = {0: 'red', 1: 'blue'}


# X, y = gdb.make_circles(n_sample=500,noise=.1,factor=.5)
# colors = {0: 'red', 1: 'blue'}


X, y = gdb.make_mons(n_samples=500, noise=0.18)
colors = {0: 'red', 1: 'blue'}

x_train, x_test, y_train, y_test = gdb.train_test_split(X, y, 0.2)
gdb.plot_dataSet(X, y, colors)

k = 5
clf = neighbors.KNeighborsClassifier(k, weights='uniform')
# clf=KNN_Classifier(k=5,evaluate_metric="accuracy")
clf.fit(x_train, y_train)
y_preds = clf.predict(x_test)
# print(clf.evaluate(y_test,y_preds))
print(accuracy_score(y_preds, y_test))

name = "{}-nn".format(k)
pltdb = PlotDB(clf, X, y, name)
pltdb.plot()
