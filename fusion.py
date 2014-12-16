from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy

class FusionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers=None):
        self.classifiers = classifiers

    def fit(self, X, y):
        self.labels = numpy.unique(y)

        for classifier in self.classifiers:
            classifier.fit(X, y)

    def predict(self, X):
        prediction = self.predict_proba(X)
        idx = numpy.argmax(prediction, axis=1)
        return self.labels[idx]


    def predict_proba(self, X):
        self.predictions_ = list()
        for classifier in self.classifiers:
            self.predictions_.append(classifier.predict_proba(X))
        return numpy.mean(self.predictions_, axis=0)

    def log_proba(self, X):
	return self.predict_proba(X)

class PerClassFusionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers=None):
        self.classifiers = classifiers

    def fit(self, X, y):
	self.labels = numpy.unique(y)

	if len(self.labels)!=len(self.classifiers):
		raise Exception('Number of classifiers should be the same number of ')

        for i in xrange(len(self.labels)):
	    label = self.labels[i]
	    yi = numpy.ones(len(y))
	    yi[y==label] = 1
	    yi[y!=label] = -1
            self.classifiers[i].fit(X, yi)

    def predict(self, X):
	prediction = self.predict_proba(X)
	idx = numpy.argmax(prediction, axis=1)
	return self.labels[idx]
	

    def predict_proba(self, X):
        self.predictions_ = []
        for classifier in self.classifiers:
            self.predictions_.append(classifier.predict_proba(X)[:,1])
	#print self.predictions_
        return numpy.vstack(self.predictions_).T

    def log_proba(self, X):
        return self.predict_proba(X)

