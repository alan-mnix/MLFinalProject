import nltk, stop_words, re, numpy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import EnglishStemmer
import json, unicodedata
import pynlpl.textprocessors as nlp
import scipy
import scipy.io
import sys
import numpy

pattern=re.compile("[^\w']")
path = 'dataset/'
token_dict = {}
stemmer = EnglishStemmer(ignore_stopwords=True)


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

def tfidf_grams(files):
    files = _toText(files)
    '''
    print "1-gram"
    cv = CountVectorizer(tokenizer=tokenize, token_pattern="word", min_df=2, strip_accents='ascii', ngram_range=(1,1), stop_words="english")
    cv.fit(files)
    features1 = cv.get_feature_names()
    print "2-gram"
    cv = CountVectorizer(tokenizer=tokenize, token_pattern="word", min_df=2, strip_accents='ascii', ngram_range=(2,2), stop_words="english")
    cv.fit(files)
    features2 = cv.get_feature_names()
    print "3-gram"
    cv = CountVectorizer(tokenizer=tokenize, token_pattern="word", min_df=2, strip_accents='ascii', ngram_range=(3,3), stop_words="english")
    cv.fit(files)
    features3 = cv.get_feature_names()
    print "4-gram"
    cv = CountVectorizer(tokenizer=tokenize, token_pattern="word", min_df=2, strip_accents='ascii', ngram_range=(4,4), stop_words="english")
    cv.fit(files)
    features4 = cv.get_feature_names()
    feat_total = numpy.concatenate((features1, features2, features3, features4), axis=0)
    '''
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words="english", token_pattern="word", min_df=2, strip_accents="ascii", ngram_range=(1,4))
    tfs = tfidf.fit_transform(files)
    features = tfidf.get_feature_names()
    arq = open("vocabulary", "w")
    for item in features:
        arq.write("%s\n" % item)
    arq.close()
    exit(0)
    return tfs, tfidf

def vectorizeTFIDF(files):
    list = _toText(files)
    #this can take some time
    tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern="word", stop_words="english", strip_accents="unicode", ngram_range=(4, 4))
    tfs = tfidf.fit_transform(list)
    features = tfidf.get_feature_names()
    return tfs, tfidf

def numericLabels(labels):
    d =  {'mathematica':0, 'photo':1, 'apple':2, 'unix':3, 'android':4, 'security':5, 'wordpress':6, 'gis':7, 'scifi':8, 'electronics':9 }
    return [d[x] for x in labels]


#example: python tfidf.py training.json input00.txt output00.txt tfidf.mat
if __name__=='__main__':

    if len(sys.argv)<5:
        print 'Usage: <train> <test> <test_label> <output>'
        exit(1)

    raw_train_file = sys.argv[1]
    raw_test_file = sys.argv[2]

    labels_test_file = sys.argv[3]
    dataset_file = sys.argv[4]


    mtrain, train = read(raw_train_file)
    mtest, test = read(raw_test_file)

    train_label = [x['topic'] for x in train]

    arquivoOut = open(labels_test_file)
    test_label = [x.strip() for x in arquivoOut.readlines()]

    # print test_label

    #Chama a funcao que faz o n-gram, na qual constroi o dicionario e entao chama a funcao do TF-IDF
    train_tfs, tfidf = tfidf_grams(train)
    test_tfs = tfidf.transform(_toText(test))


    scipy.io.savemat(dataset_file, dict(Xtrain = train_tfs, Ytrain = numericLabels(train_label), Xtest = test_tfs, Ytest = numericLabels(test_label), dictionary = tfidf.get_feature_names()))
