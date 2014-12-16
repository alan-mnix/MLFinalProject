import sklearn
import sklearn.svm
import numpy
import scipy
import scipy.io
import sys, glob
import os
import os.path
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

#Somente o nome do arquivo
if __name__=='__main__':

    C = range(1, 100)
    for c in C:
        for file in glob.glob(os.path.join(sys.argv[1], '*.mat')):
            data = scipy.io.loadmat(file)

            print("\nTreinando SVM multi classe...")
            clf = sklearn.svm.SVC(kernel="linear", C=float(c))

            ytrain = data['Ytrain'].T.reshape(data['Ytrain'].shape[1])
            clf.fit(data['Xtrain'], ytrain)
            predict = clf.predict(data['Xval'])

            yVal = data['Yval'].T.reshape(data['Yval'].shape[1])
            print "Acuracia: ", sklearn.metrics.accuracy_score(yVal, predict)

            cm = confusion_matrix(yVal, predict)
            total = numpy.sum(cm, axis=1)

            if(cm.shape[0] < 2):
                acc = 1.0
            else:
                acc = []
                for i in range(total.shape[0]):
                    if(total[i] > 0):
                        acc.append(float(cm[i, i])/float(total[i]))

            print "Acuracia norm: ", sum(acc)/float(len(acc))

            X_test, y_test = data["Xtest"], data["Ytest"]

    clf = sklearn.svm.SVC(C=2.0, kernel="linear")
    Xtrain = numpy.concatenate((data["Xtrain"].toarray(), data["Xval"].toarray()), axis=0)
    y_train = numpy.concatenate((ytrain, yVal), axis=0)
    clf.fit(Xtrain, y_train)
    predict = clf.predict(X_test.toarray())
    yTest = y_test.T.reshape(y_test.shape[1])
    cm = confusion_matrix(yTest, predict)
    total = numpy.sum(cm, axis=1)

    if(cm.shape[0] < 2):
        acc = 1.0
    else:
        acc = []
        for i in range(total.shape[0]):
            if(total[i] > 0):
                acc.append(float(cm[i, i])/float(total[i]))
    print "\n---- Teste ----"

    print "Acuracia teste: ", sklearn.metrics.accuracy_score(yTest, predict)
    print "Acuracia teste norm: ", sum(acc)/float(len(acc))

