from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import numpy
import scipy
import scipy.io
import sys, glob
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn import grid_search

#Somente o nome do arquivo
if __name__=='__main__':
    parametros = []
    total_score = 0
    normalized_score = 0
    for file in glob.glob(sys.argv[1]):
        data = scipy.io.loadmat(file)

        ytrain = data['Ytrain'].T.reshape(data['Ytrain'].shape[1])

        x_train, x_val, y_train, y_val = cross_validation.train_test_split(data['Xtrain'], ytrain, test_size=0.2, random_state=0)
        tuned_parameters = [
            #{'alpha': [0.15]}
            {'alpha': [0.2]}
        ]

        print "-- TRAINNING: grid search with 5 fold cross-validation"
        clf = grid_search.GridSearchCV(BernoulliNB(), tuned_parameters, cv=10, scoring='accuracy')
        clf.fit(x_train, y_train)

        print "score : " + str(clf.best_score_)
        print "params : " + str(clf.best_params_)
        parametros.append(clf.best_params_)
        for params, mean_score, scores in clf.grid_scores_:
            print str(mean_score) + " " + str(scores) + " " + str(params)

        y_true, y_pred = y_val, clf.predict(x_val)
        score = accuracy_score(y_true, y_pred)
        total_score += score

        cm = confusion_matrix(y_true, y_pred)
        total = numpy.sum(cm, axis=1)

        if(cm.shape[0] < 2):
            acc = 1.0
        else:
            acc = []
            for i in range(total.shape[0]):
                if(total[i] > 0):
                    acc.append(float(cm[i, i])/float(total[i]))

        normalized_score += sum(acc)/float(len(acc))

        print "score : " + str(score)
        # print metrics.classification_report(y_true, y_pred)
        print "score norm: ", sum(acc)/float(len(acc))
        print "----------------------------"

    '''
    clf = MultinomialNB(alpha=1)
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
    '''