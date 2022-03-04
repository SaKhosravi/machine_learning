from sklearn import neighbors
from Knn.knn import KNN_Classifier
from utils.generate_data_set import GenerateDataSetBlob
from sklearn.metrics import accuracy_score
from utils.plot_decision_boundaries import PlotDB

gdb = GenerateDataSetBlob()
X, y = gdb.make_blobs(n_samples=150, n_features=2, centers=3,
                      random_state=0, cluster_std=1.5)

x_train, x_test, y_train, y_test = gdb.train_test_split(X, y, 0.2)
colors = {0: 'red', 1: 'blue', 2: 'green'}
gdb.plot_dataSet(X, y, colors)

start, end = 1, 150

k = 1
clf = neighbors.KNeighborsClassifier(k, weights='uniform')
# clf=KNN_Classifier(k=5)
clf.fit(x_train, y_train)
y_preds=clf.predict(x_test)
print(accuracy_score(y_preds,y_test))

name = "{}-nn".format(k)
pltdb = PlotDB(clf, X, y, name)
pltdb.plot()

