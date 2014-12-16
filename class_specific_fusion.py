import sklearn
import sklearn.naive_bayes
import sklearn.svm
import numpy
import fusion
import scipy
import scipy.io
import sys, glob
import os
import os.path
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import grid_search

def cross_val_score(cl, x, y, cv=5):
	score = []
	i=1
	for train, test in sklearn.cross_validation.StratifiedKFold(y, n_folds=cv, shuffle=True):
		print 'Fold ', i
		cl.fit(x[train], y[train])
		yp = cl.predict(x[test])
		acc = sklearn.metrics.accuracy_score(y[test], yp)
		print sklearn.metrics.confusion_matrix(y[test], yp)
		score.append(acc)
		print acc
		i+=1
	return numpy.array(score)

def create_svms(n = 10):
	svms = []
	for i in xrange(10):
		svms.append(sklearn.svm.SVC(C=1, gamma=1e-3, probability=True, class_weight={-1:1, 1:10}))
	return svms


#Somente o nome do arquivo
if __name__=='__main__':

    for file in glob.glob(sys.argv[1]+'.mat'):
        data = scipy.io.loadmat(file)

        #print("\nTreinando SVM multi classe...")
        #print("\n"+file.split("gram_")[1].split("_")[0])
        ytrain = data['Ytrain'].T.reshape(data['Ytrain'].shape[1])

        x_train, x_val, y_train, y_val = cross_validation.train_test_split(data['Xtrain'], ytrain, test_size=0.2, random_state=0)
        tuned_parameters = []

	#cl = fusion.PerClassFusionClassifier([sklearn.svm.SVC(C=1, gamma=1e-3, probability=True), sklearn.svm.SVC(C=1, gamma=1e-3, probability=True), sklearn.svm.SVC(C=1, gamma=1e-3, probability=True), sklearn.svm.SVC(C=1, gamma=1e-3, probability=True), sklearn.svm.SVC(C=1, gamma=1e-3, probability=True), sklearn.svm.SVC(C=1, gamma=1e-3, probability=True), sklearn.svm.SVC(C=1, gamma=1e-3, probability=True), sklearn.svm.SVC(C=1, gamma=1e-3, probability=True),sklearn.svm.SVC(C=1, gamma=1e-3, probability=True),sklearn.svm.SVC(C=1, gamma=1e-3, probability=True)])

	cl = fusion.PerClassFusionClassifier(create_svms())

        #print " -- TRAINNING: grid search with 5 fold cross-validation"

	#cl.fit(x_train, y_train)

	#print cl.predict(x_val)	

	scores = cross_val_score(cl, x_train, y_train, cv=5)

        y_true, y_pred = y_val, cl.predict(x_val)
        score = accuracy_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        total = numpy.sum(cm, axis=1)

        if(cm.shape[0] < 2):
            acc = 1.0
        else:
            acc = []
            for i in range(total.shape[0]):
                if(total[i] > 0):
                    acc.append(float(cm[i, i])/float(total[i]))


        print "score : ", numpy.mean(scores), ' +- ', numpy.std(scores)
        # print metrics.classification_report(y_true, y_pred)
        print "score norm: ", sum(acc)/float(len(acc))
        print "----------------------------"

        print 'Validation Score: ', score

