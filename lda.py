from sklearn.lda import LDA
import numpy
import sklearn
import scipy.io
import sys, glob
from sklearn.metrics import confusion_matrix

#Somente o nome do arquivo
if __name__=='__main__':

    for file in glob.glob(sys.argv[1]+'*.mat'):
        data = scipy.io.loadmat(file)
        X_train = data['Xtrain']
        y_train = data['Ytrain'].T

        print("Treinando LDA...")
        lda = LDA()

        ytrain = data['Ytrain'].T.reshape(data['Ytrain'].shape[1])
        lda.fit(data['Xtrain'].toarray(), ytrain)
        predict = lda.predict(data['Xval'].toarray())

        yVal = data['Yval'].T.reshape(data['Yval'].shape[1])
        print "Acuracia: ", sklearn.metrics.accuracy_score(yVal, predict)
        X_train = data["Xtrain"]
        X_val = data["Xval"]
        X_test, y_test = data["Xtest"], data["Ytest"]

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


    lda = LDA()
    Xtrain = numpy.concatenate((X_train.toarray(), X_val.toarray()), axis=0)
    y_train = numpy.concatenate((ytrain, yVal), axis=0)
    lda.fit(Xtrain, y_train)
    predict = lda.predict(X_test.toarray())
    yTest = y_test.T.reshape(y_test.shape[1])
    print "Acuracia Teste: ", sklearn.metrics.accuracy_score(yTest, predict)