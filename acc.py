import numpy
import sklearn
import sklearn.metrics

def f2array(label_file):
	f = open(label_file)
    	return [x.strip() for x in f.readlines()]


def main():
	f1name = 'myout.txt'
	f2name = 'output00.txt'
	yp = f2array(f1name)
	yt = f2array(f2name)
	print 'Acc: ',  sklearn.metrics.accuracy_score(yp, yt)

main()	
