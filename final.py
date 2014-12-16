import nltk, stop_words, re, numpy
from nltk.stem.snowball import EnglishStemmer
import json, unicodedata
import pynlpl.textprocessors as nlp
import unicodedata
import re
import json
import sklearn
import sklearn.metrics
import sklearn.svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


stemmer = EnglishStemmer(ignore_stopwords=True)
pattern=re.compile("[^\w']")
path = 'dataset/'
token_dict = {}

def read(file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))
    m = data[0]
    data.pop(0)
    return m, data

def stem_tokens(tokens, stemmer):
    stemmed = []
    stopWords = stop_words.get_stop_words("English")
    for item in tokens:
        if(item not in stopWords and len(item) > 3 and item.isalpha()):
            formated = pattern.sub('',str(item))
            steam  = stemmer.stem(formated)
            if(len(formated) > 0 and not stemmed.__contains__(steam)):
                stemmed.append(steam)
    return stemmed


def tokenize(text):
    #tokens = nltk.word_tokenize(text)
    tokens = nlp.tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def _toText(files):
    list = [""]*len(files)
    for i in range(len(files)):
        file = files[i]
        file["excerpt"] = unicodedata.normalize('NFKD', file["excerpt"]).encode('ascii', 'ignore')
        file["question"]  = unicodedata.normalize('NFKD', file["question"]).encode('ascii', 'ignore')
        text = str(re.sub(r'[^\x00-\x7F]+',' ', file["question"])+" "+re.sub(r'[^\x00-\x7F]+',' ', file["excerpt"]))
        caracters = ["\'", "\n", "\r", "\t"]
        for c in caracters:
            text = text.replace(c, "")
        list[i] = text
    return list

def vectorizeTFIDF(files):
    list = _toText(files)
    #this can take some time
    tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern="word", stop_words="english", strip_accents="unicode")
    tfs = tfidf.fit_transform(list)
    features = tfidf.get_feature_names()
    return tfs, tfidf

def numericLabels(labels):
    d =  {'mathematica':0, 'photo':1, 'apple':2, 'unix':3, 'android':4, 'security':5, 'wordpress':6, 'gis':7, 'scifi':8, 'electronics':9 }
    return [d[x] for x in labels]

raw_train_file = 'training.json'

mtrain, train = read(raw_train_file)

train_label = [x['topic'] for x in train]

    #Chama a funcao que faz o n-gram, na qual constroi o dicionario e entao chama a funcao do TF-IDF
train_tfs, tfidf = vectorizeTFIDF(train)

clf = sklearn.svm.LinearSVC(multi_class='ovr', tol=1e-3, dual=False)

clf.fit(train_tfs, train_label)

#print 'Manda os paranaue'

m = input()

inputs = [None]*m
for i in xrange(m):
	t = raw_input()
	inputs[i] = json.loads(t)

X = tfidf.transform(_toText(inputs))
yp = clf.predict(X)

for i in xrange(m):
	print yp[i]
