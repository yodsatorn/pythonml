from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

clf = svm.SVC()
print(clf.fit(iris.data, iris.target))

x_train,x_test,y_train, y_test = train_test_split( iris.data, iris.target, test_size=0.4 , random_state=0)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
