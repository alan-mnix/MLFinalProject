# -*- coding: utf-8 -*-
from read import read
from time import time
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import tfidf, numpy
from sklearn.lda import LDA
import pickle
from scipy.sparse import csr_matrix
from sklearn.random_projection import sparse_random_matrix
from sklearn.svm import SVC
from scipy import spatial
from mlp import MLPClassifier
from sklearn.decomposition import PCA

def classify(train_samples_pred, test_point, k):

    #print("classify")
    distances=spatial.distance.cdist(train_samples_pred, test_point)[:,0]
    nns = distances.argsort()
    return nns[:k]

def topTerms(tfidf, tfs):
    terms = tfidf.get_feature_names()
    for i in range(tfs.shape[0]):
        row = tfs[i, :]
        row = row.tolist()
        indexes = []
        copyRow = row[:]
        for i in range(10):
            maximal = max(copyRow)
            idx = row.index(maximal)
            indexes.append(idx)
            copyRow.remove(maximal)
        topTerms = []
        for i in indexes:
            topTerms.append(terms[i])
        print topTerms

m, files1 = read("training.json")
t, files2 = read("input00.txt")
files = files1+files2
y = [str(file["topic"]) for file in files1]
map = []
for i in range(len(y)):
    if(len(map) == 0 or not map.__contains__(y[i])):
        map.append(y[i])
y_map = numpy.array([map.index(y[i]) for i in range(len(y))])

print("Construindo TF-IDF...")
#tfidf.return_features(files)
X, vectorizer = tfidf.vectorizeTFIDF(files)
X_train = X[:m,:]
X_test = X[m:, :]
print X_train.shape, X_test.shape


'''
a = open("features", "wb")
for feature in vectorizer.get_feature_names():
    a.write(feature)
    a.write("\n")

a.close()
'''

'''
print("Performing dimensionality reduction using LSA...")
svd = TruncatedSVD(n_components=2, algorithm='arpack')
lsa = make_pipeline(svd, Normalizer(copy=False))
lsa.fit(X_train)
X_train = lsa.transform(X_train)
X_test = lsa.transform(X_test)
'''

'''
print("Performing dimensionality reduction using PCA...")
pca = PCA(n_components=0.9)
X_train = X_train.toarray()
X_test = X_test.toarray()
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
'''
'''
print("Performing dimensionality reduction using LDA...")
lda = LDA(n_components=9)
X_train = X_train.toarray()
X_test = X_test.toarray()
lda.fit(X_train, y_map)
X_train = lda.transform(X_train)
X_test = lda.transform(X_test)
'''


print("Treinando SVM multi classe...")
clf = SVC(kernel="linear")
clf.fit(X_train, y_map)

print("Calculando predições...")
predict = clf.predict(X_test)
predict = list(predict)


'''
print("Treinando MLP...")
mlp = MLPClassifier()
mlp.fit(X_train, y_map)
training_score = mlp.score(X_train, y_map)
print training_score
'''

'''
X_train = X_train.toarray()
predict = []
print("Calculando kNN...")
for i in range(X_test.shape[0]):
    print(str(i)+" de "+str(X_test.shape[0]))
    #point_test = numpy.array([X_test[i, :].T])
    point_test = X_test.getrow(i)
    point_test = point_test.toarray()
    nns = classify(X_train, point_test, 10)
    nns = list(nns)
    labels = [y_map[n] for n in nns]
    total = []
    for k in range(len(map)):
        total.append(labels.count(k))
    label = total.index(max(total))
    predict.append(label)
'''


#Contar a qtd de acertos
arquivoOut = open("output00.txt")
output = arquivoOut.readlines()
output = [item.replace("\n", "") for item in output]

print sklearn.metrics.accuracy_score(data['Ytest'].T, predict)
print("Predicões:\n")
count = 0
for i in range(len(predict)):
    if(map[predict[i]] == output[i]):
        count += 1
    print map[predict[i]], output[i]

print("Total: "+str(count)+" de "+str(len(predict)))
