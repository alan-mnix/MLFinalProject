import sklearn
import sklearn.svm
import numpy
import scipy
import scipy.io
import sys, glob
import os
import os.path
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import grid_search

#Somente o nome do arquivo
if __name__=='__main__':

    for file in glob.glob(sys.argv[1]+'.mat'):
        data = scipy.io.loadmat(file)

        #print("\nTreinando SVM multi classe...")
        #print("\n"+file.split("gram_")[1].split("_")[0])
        ytrain = data['Ytrain'].T.reshape(data['Ytrain'].shape[1])

        x_train, x_val, y_train, y_val = cross_validation.train_test_split(data['Xtrain'], ytrain, test_size=0.2, random_state=0)
        tuned_parameters = [
            {'kernel': ['linear'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [1, 10, 100]}
        ]

        print " -- TRAINNING: grid search with 10 fold cross-validation"
        clf = grid_search.GridSearchCV(sklearn.svm.SVC(), tuned_parameters, cv=10, scoring='accuracy')
        clf.fit(x_train, y_train)

        print "score : " + str(clf.best_score_)
        print "params : " + str(clf.best_params_)
        for params, mean_score, scores in clf.grid_scores_:
            print str(mean_score) + " " + str(scores) + " " + str(params)

        y_true, y_pred = y_val, clf.predict(x_val)
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



        print "score : " + str(score)
        # print metrics.classification_report(y_true, y_pred)
        print "score norm: ", sum(acc)/float(len(acc))
        print "----------------------------"

        '''
        yVal = data['Yval'].T.reshape(data['Yval'].shape[1])
        clf = sklearn.svm.SVC(kernel="linear", C=1)

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

    clf = sklearn.svm.SVC(C=1.0, kernel="linear")
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
'''