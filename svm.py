import sklearn
import sklearn.svm
import numpy
import scipy
import scipy.io
import sys, glob
import os
import sklearn.ensemble
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
            {'C': [.001, 0.1, 0.5, 1, 10]}
        ]

	#x_train = x_train[ (y_train == 2) | (y_train == 3) ,:]
        #y_train = y_train[ (y_train == 2)  | (y_train == 3) ]

	#x_val = x_val[ (y_val == 2) | (y_val == 3) ,:]
        #y_val = y_val[ (y_val == 2) | (y_val == 3) ]	

	print len(y_val[y_val==2]), len(y_val[y_val==3]), float(len(y_val[y_val==2]))/ len(y_val[y_val==3])
        print len(y_train[y_train==2]), len(y_train[y_train==3]), float(len(y_train[y_train==2]))/len(y_train[y_train==3])


        print " -- TRAINNING: grid search with 10 fold cross-validation"
        clf = grid_search.GridSearchCV(sklearn.svm.LinearSVC(multi_class='ovr', tol=1e-6, dual=False), tuned_parameters, cv=5, scoring='accuracy')

	import sklearn.naive_bayes
	#import sklearn.neighbors
	#import sklearn.multiclass
	#clf = grid_search.GridSearchCV( sklearn.ensemble.GradientBoostingClassifier(), {}, scoring='accuracy')
        
	clf.fit(x_train, y_train)

        print "score : " + str(clf.best_score_)
        print "params : " + str(clf.best_params_)
        for params, mean_score, scores in clf.grid_scores_:
            print str(mean_score) + " " + str(scores) + " " + str(params)

        y_true, y_pred = y_val, clf.predict(x_val)
        score = accuracy_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        total = numpy.sum(cm, axis=1)
	print cm.astype(float)*100.0/total

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
