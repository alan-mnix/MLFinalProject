import sklearn
from sklearn import svm
import numpy
import scipy
import scipy.io
import sys, glob
import os
import os.path
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.metrics import confusion_matrix, accuracy_score

#Somente o nome do arquivo
if __name__=='__main__':

        data = scipy.io.loadmat(sys.argv[1])

        #print("\nTreinando SVM multi classe...")
        #print("\n"+file.split("gram_")[1].split("_")[0])
        ytrain = data['Ytrain'].T.reshape(data['Ytrain'].shape[1])
        classes = numpy.unique(ytrain)
        parametros = []
        total_score = 0
        normalized_score = 0
        for klass in classes:

            indexes_of_class_pos = (ytrain == klass).nonzero()[0]
            indexes_of_class_neg = [x for x in range(0, data['Xtrain'].shape[0]) if x not in indexes_of_class_pos]

            labels = numpy.ones((ytrain.shape[0]))
            for i in range(labels.shape[0]):
                if(indexes_of_class_neg.__contains__(i)):
                    labels[i] = -1
            x_train, x_val, y_train, y_val = cross_validation.train_test_split(data['Xtrain'], labels, test_size=0.2, random_state=0)
            tuned_parameters = [
                {'kernel': ['linear'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]}
                # {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                # {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
            ]

            print "class : " + str(klass)
            print " -- TRAINNING: grid search with 5 fold cross-validation"
            clf = grid_search.GridSearchCV(svm.SVC(class_weight={-1:1,1:10}), tuned_parameters, cv=5, scoring='accuracy')
            clf.fit(x_train, y_train)

            print "score : " + str(clf.best_score_)
            print "params : " + str(clf.best_params_)
            parametros.append(clf.best_params_)
            for params, mean_score, scores in clf.grid_scores_:
			    print '>> ', str(mean_score) + " " + str(scores) + " " + str(params)

            y_true, y_pred = y_val, clf.predict(x_val)
            score = accuracy_score(y_true, y_pred)
            total_score += score

            cm = confusion_matrix(y_true, y_pred)
	    cm = cm.astype(float)
	    print cm
	    cm = (cm.T/cm.sum(axis=1)).T
            #total = numpy.sum(cm, axis=1)

            #if(cm.shape[0] < 2):
            #    acc = 1.0
            #else:
            #    acc = []
            #    for i in range(total.shape[0]):
            #        if(total[i] > 0):
            #            acc.append(float(cm[i, i])/float(total[i]))

            #normalized_score += sum(acc)/float(len(acc))

	    normalized_score += numpy.trace(cm)/cm.shape[0]

            print "score : " + str(score)
            # print metrics.classification_report(y_true, y_pred)
            print "score norm: ", numpy.trace(cm)/cm.shape[0]
            print "----------------------------"
            #clf = sklearn.svm.SVC(kernel="linear", C=1)

        print "total score = " + str(total_score / len(classes))
        print "total norm = " + str(normalized_score / len(classes))
        print parametros
            #clf.fit(data['Xtrain'], ytrain)
            #predict = clf.predict(data['Xval'])
        '''
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
