from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from knn import KNN_Classifier

#laod data set
data=load_iris()
x=data.data
y=data.target


x_train , x_test , y_train , y_test=train_test_split(x, y, test_size=0.2, shuffle=True)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#initialize
knn=KNN_Classifier(k=3)

#fit
knn.fit(x_train, y_train)

#predict
y_pred= knn.predict(x_test)


acc= accuracy_score(y_test, y_pred)
print(acc)
#
# print("knn with sklearn , ec")
# from sklearn.neighbors import KNeighborsClassifier
# clsi=KNeighborsClassifier(n_neighbors=3)
# clsi.fit(x_train, y_train)
# y_pred= clsi.predict(x_test)
# print(accuracy_score(y_test,y_pred))