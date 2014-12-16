import scipy.io
import sys, numpy
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn import cross_validation

def lsa(X_train, X_val, components):
    print "Performing dimensionality reduction using LSA: ", components
    svd = TruncatedSVD(n_components=components)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    lsa.fit(X_train)
    X_train = lsa.transform(X_train)
    X_val = lsa.transform(X_val)
    return  X_train, X_val

def lda(X_train, X_val, y_train):
    print("Performing dimensionality reduction using LDA...")
    lda = LDA()
    try:
        lda.fit(X_train, y_train)
    except TypeError:
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        lda.fit(X_train, y_train)
    X_train = lda.transform(X_train)
    X_val = lda.transform(X_val)
    return  X_train, X_val

def pca(X_train, X_val, components):
    print("Performing dimensionality reduction using PCA...")
    pca = PCA(n_components=components)
    try:
        pca.fit(X_train)
    except TypeError:
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)
    return  X_train, X_val

#ex: python validation.py tfidf.mat tfidf_red lsa 1000
if __name__=='__main__':

    if(len(sys.argv) < 3):
        print "Usage: <input> <output_pattern> OR <input> <output_pattern> <reduction_method> <n_components>"
    data_name = sys.argv[1]
    data = scipy.io.loadmat(data_name)
    dataset_file = sys.argv[2]

    if(len(sys.argv) > 3):
        f = sys.argv[3]
        components = int(sys.argv[4])

    X_train = data['Xtrain']
    y_train = data['Ytrain'].T
    y_train =  y_train.reshape(y_train.shape[0])

    XTrain, XTest = lsa(X_train, data['Xtest'], components)
    scipy.io.savemat(dataset_file+".mat", dict(Xtrain = XTrain, Ytrain = data['Ytrain'], Xtest = XTest, Ytest = data['Ytest']))
'''
    skf = cross_validation.StratifiedKFold(y_train, n_folds=10)
    cont = 1

    for train_index, val_index in skf:
        XTrain, XVal = X_train[train_index], X_train[val_index]
        yTrain, yVal = y_train[train_index], y_train[val_index]

        if(len(sys.argv) > 3):
            dictF = {"lsa":lsa, "lda":lda, "pca":pca}
            if(f=="lda"):
                XTrain, XVal = dictF[f](XTrain, XVal, yTrain)
            else:
                XTrain, XVal = dictF[f](XTrain, XVal, components)

        scipy.io.savemat(dataset_file+"_"+str(cont)+".mat", dict(Xtrain = XTrain, Ytrain = yTrain, Xtest = data['Xtest'], Ytest = data['Ytest'] , Xval = XVal, Yval = yVal))
        cont += 1
'''