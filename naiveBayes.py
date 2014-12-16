from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import numpy
import scipy
import scipy.io
import sys, glob
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

#Somente o nome do arquivo
if __name__=='__main__':
    for file in glob.glob(sys.argv[1]+'*.mat'):
        data = scipy.io.loadmat(file)

        #print("\nTreinando Naive Bayes...")
        clf = BernoulliNB(alpha=0.2)

        ytrain = data['Ytrain'].T.reshape(data['Ytrain'].shape[1])
        Xtrain = data['Xtrain']
        Xval = data['Xval']
        clf.fit(Xtrain, ytrain)
        predict = clf.predict(Xval)

        yVal = data['Yval'].T.reshape(data['Yval'].shape[1])
        print "\nAcuracia: ", accuracy_score(yVal, predict)
        X_train = data["Xtrain"]
        X_val = data["Xval"]

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

    clf = BernoulliNB(alpha=0.2)
    Xtrain = numpy.concatenate((Xtrain.toarray(), X_val.toarray()), axis=0)
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

    print "Acuracia teste: ", accuracy_score(yTest, predict)
    print "Acuracia teste norm: ", sum(acc)/float(len(acc))