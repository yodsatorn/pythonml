from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

clf = svm.SVC()
print(clf.fit(iris.data, iris.target))

train_test_split( iris.data, iris.target, test_size=0.4 , random_state=0)
