import nltk, stop_words, re, numpy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import EnglishStemmer
import json, unicodedata
import pynlpl.textprocessors as nlp
import scipy
import scipy.io
from gensim import corpora, models, similarities, matutils
import sys

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
    print 'to_text data', files[0]
    for i in range(len(files)):
	file = files[i]
        try:
		file["excerpt"] = unicodedata.normalize('NFKD', file["excerpt"]).encode('ascii', 'ignore')
        	file["question"]  = unicodedata.normalize('NFKD', file["question"]).encode('ascii', 'ignore')
	except:
		e = ''
		#print '>>> ', file['excerpt']
        text = str(re.sub(r'[^\x00-\x7F]+',' ', file["question"])+" "+re.sub(r'[^\x00-\x7F]+',' ', file["excerpt"]))
        caracters = ["\'", "\n", "\r", "\t"]
        for c in caracters:
            text = text.replace(c, "")
        list[i] = text
    return list

def get_corpus(data):
	texts = _toText(data)
	texts = [tokenize(x) for x in texts]
	print type(texts)
	dictionary = corpora.Dictionary(texts)
	#print dictionary
	dictionary.save('temp.dict')
	corpus = [dictionary.doc2bow(text) for text in texts]
	return corpus, dictionary

def to_corpus(data, dic):
	print 'Before data ', data[0]
	texts = _toText(data)
        texts = [tokenize(x) for x in texts]
	corpus = [dic.doc2bow(text) for text in texts]
	return corpus


def create_LDA(corpus, dictionary, components):
        lda = models.ldamulticore.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=components)
	for i in lda.show_topics():
		print i
	return lda

def LDA(data, components):
	corpus, dic = get_corpus(data)
	lda = create_LDA(corpus, dic, components)
	print lda
	return lda, dic

def LDA_transform(lda, dictionary, data):
	print 'Transforming ...'
	corpus = to_corpus(data, dictionary)
	print len(corpus)
	f = [lda[c] for c in corpus]
	return matutils.corpus2csc(f).T

def numericLabels(labels):
    d =  {'mathematica':0, 'photo':1, 'apple':2, 'unix':3, 'android':4, 'security':5, 'wordpress':6, 'gis':7, 'scifi':8, 'electronics':9 }
    return [d[x] for x in labels]


#example: python tfidf.py training.json input00.txt output00.txt tfidf.mat
if __name__=='__main__':

    if len(sys.argv)<6:
        print 'Usage: <train> <test> <test_label> <output> <# of topics>'
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

    components = int(sys.argv[5])

    print train[0]
    print test[0]

    # print test_label

    #Chama a funcao que faz o n-gram, na qual constroi o dicionario e entao chama a funcao do TF-IDF
    print 'Computing LDA ...'
    corpus, dic =  get_corpus(train)
    #print 'Extracting Train ...'
    #train_tfs = to_corpus(dic, train)
    print 'Extracting Test'
    print test[0]
    test_tfs = to_corpus(test, dic)
    train_tfs = corpus

    #print train_tfs.shape()

    train_tfs = matutils.corpus2csc(train_tfs).T
    test_tfs = matutils.corpus2csc(test_tfs).T

    print 'Saving ...'
    scipy.io.savemat(dataset_file, dict(Xtrain = train_tfs, Ytrain = numericLabels(train_label), Xtest = test_tfs, Ytest = numericLabels(test_label)))
