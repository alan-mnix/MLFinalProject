import nltk, re, numpy
import json
import re
import sklearn
import sklearn.metrics
import sklearn.svm
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import time
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
import sys

#import psyco
#psyco.full()


pattern=re.compile("[^\w']")
path = 'dataset/'
token_dict = {}

p = re.compile(r"[\w']+")

#buffer = [None]*1000

class StemmingTokenizer(object):

        def __init__(self):
                #self.stemmer = nltk.stem.RSLPStemmer()
		self.stemmer = nltk.stem.RegexpStemmer('ing$|s$|e$|able$|or$|er$|ness$|ional$|ful$|ment$', min = 4)
		#self.stemmer = nltk.stem.porter.PorterStemmer()

        def __call__(self, doc):
            return [self.stemmer.stem(t.lower()) for t in p.findall(doc)]
	        #self.stemmer.stem(t.lower())
		


def read(file):
    data = map(json.loads, open(file).readlines())
    m = data[0]
    data.pop(0)
    return m, data

def _toAll(files, tfidf):
    t= time.time()
    feats = tfidf.get_feature_names()
    list = [None]*len(files)
    for i in range(len(files)):
        file = files[i]
        text = file['question'] + ' ' + file['excerpt']

	text = text.split()
	t = []
	for i in text:
		if i in feats:
			t.append(i)
	text = string.join(' ', t)
        
        list[i] = text
    
    return list


def _toText(files):
    t= time.time()
    list = [""]*len(files)
    for i in range(len(files)):
        file = files[i]
        
	text = file['question'] + ' ' + file['excerpt']
        list[i] = text
    return list

def vectorizeTFIDF(files):
    t = time.time()
    list = _toText(files)
    tfidf = TfidfVectorizer(tokenizer = StemmingTokenizer(), token_pattern="word", stop_words="english")
    tfs = tfidf.fit_transform(list)
    #print 'TFIDF: ', time.time() - t

    t = time.time()
    token = StemmingTokenizer()
    for i in list:
	token(i)
    #print 'Time stemmer: ', time.time() - t
    return tfs, tfidf

def numericLabels(labels):
    d =  {'mathematica':0, 'photo':1, 'apple':2, 'unix':3, 'android':4, 'security':5, 'wordpress':6, 'gis':7, 'scifi':8, 'electronics':9 }
    return [d[x] for x in labels]

def main():

	raw_train_file = 'training.json'

	mtrain, train = read(raw_train_file)

	train_label = numpy.array([x['topic'] for x in train])

    #Chama a funcao que faz o n-gram, na qual constroi o dicionario e entao chama a funcao do TF-IDF
	train_tfs, tfidf = vectorizeTFIDF(train)

	clf = sklearn.svm.LinearSVC(multi_class='ovr', dual=False)
	#lfp = sklearn.svm.LinearSVC(multi_class='ovr', tol=1e-6)

	clf.fit(train_tfs, train_label)

	data = sys.stdin.readlines()
	m = int(data[0])

	inputs = [None]*m
	for i in xrange(m):
		t = data[i+1]
		inputs[i] = json.loads(t)

	t = time.time()
	X = tfidf.transform(_toText(inputs))
	
	t = time.time()
	yp = clf.predict(X)

	sys.stdout.write('\n'.join(yp))

main()
