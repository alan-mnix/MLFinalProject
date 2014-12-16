import sklearn.naive_bayes
import sklearn.svm
import sklearn.metrics
import fusion
import sklearn.datasets
import sklearn.linear_model

data = sklearn.datasets.load_iris()

clf = fusion.FusionClassifier([sklearn.svm.SVC(probability=True), sklearn.svm.SVC(probability=True), sklearn.naive_bayes.MultinomialNB()])

clf2 = sklearn.naive_bayes.MultinomialNB()

clf2.fit(data.data, data.target)
clf.fit(data.data, data.target)

y = clf.predict(data.data)
y2 = clf2.predict(data.data)

print sklearn.metrics.accuracy_score(data.target, y)
print sklearn.metrics.accuracy_score(data.target, y2)

