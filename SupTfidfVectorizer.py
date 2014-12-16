import sklearn
import scipy
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class SupTfidfVectorizer(TfidfVectorizer):
	"""docstring for SupervisedTFIDFVectorizer"""
	def __init__(self, **kwargs):
		super(SupTfidfVectorizer, self).__init__(**kwargs)

	def fit_transform(self, X, Y):
		self._tfidf.use_idf=False
		print self._tfidf.__dict__
		Xt = super(SupTfidfVectorizer, self).fit_transform(X)
		Xti = scipy.sparse.coo_matrix(Xt)
		print type(Xt)
		print type(Xt[0])
		print Xt.shape

		labels = numpy.unique(Y)
		clabels = len(labels)

		print 'initializing sidf ...'

		m = Xt.shape[1]
		self._sidf = {}
		for i in labels:
			self._sidf[i] = numpy.zeros(m)

		for i,j,v in zip(Xti.row, Xti.col, Xti.data):
			a = i, j, v
			self._sidf[Y[i]][j]+=v

		print 'counting sidf ...'

		self._midf = numpy.zeros(m)

		print 'average sidf ...'
		for i in labels:
			self._midf += self._sidf[i]
		print 'divide clabels'
		self._midf /= clabels

		print 'sum 1 ...'
		self._midf+=1 #avoid zero division

		print 'invert'
		self._midf = 1.0/self._midf
		diag_midf = scipy.sparse.spdiags(self._midf, 0, m, m)

		print 'Transform the roles ...'
		self._tfidf.use_idf = True
		self._tfidf._idf_diag = diag_midf
		return self._tfidf.transform(Xt)

	def transform(self, X):
		self._tfidf.use_idf=False
		Xt = super(SupTfidfVectorizer, self).transform(X)
		self._tfidf.use_idf=True
		return self._tfidf.transform(Xt)

