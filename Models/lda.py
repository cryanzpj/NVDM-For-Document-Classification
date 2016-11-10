import numpy as np
import sys
sys.path.insert(0, '/Users/cryan/Desktop/Project/NVDM-For-Document-Classification/Models')
from make_corp import *
import gensim
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics

from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier


def train_lsi(corp,id2word,n_topic =100,test = None):
    lsi = gensim.models.lsimodel.LsiModel(corpus=np.array(corp),id2word=id2word,num_topics = n_topic)
    #lda = gensim.models.hdpmodel.HdpModel(corp, id2word=id2word)

    train_feature = np.zeros((len(corp),n_topic))
    for i,j in enumerate(corp):
        temp = np.asarray(lsi[j])
        train_feature[i,temp[:,0].astype(int)] = temp[:,1]

    if test:
        test_feature = np.zeros((len(test),n_topic))
        for i,j in enumerate(test):
            temp = np.asarray(lsi[j])
            test_feature[i,temp[:,0].astype(int)] = temp[:,1]

    return lsi,train_feature,test_feature

if __name__ == "__main__":
    train_dir = "../data/RCV1/token/lyrl2004_tokens_train.dat"
    vocab_dir =  "../data/RCV1/token/lyrl2004_tokens_train.txt"
    test_dir = "../data/RCV1/token/lyrl2004_tokens_test_pt0.dat"
    labels = "../data/RCV1/token/rcv1-v2.topics.qrels"


    #load data and convert to corpus
    vocab = vocab_to_dict(vocab_dir)
    corp = make_corp(train_dir,vocab)
    id2word = dict(np.array(vocab.items(),dtype=object)[:,::-1])

    test = make_corp(test_dir,vocab)

    #train lsi model and get document representations
    lsi,data_train,data_test = train_lsi(corp,id2word,2000,test)

    label = fetch_rcv1(data_home="../data/RCV1/vec",download_if_missing=False)

    model = OneVsRestClassifier(PassiveAggressiveClassifier(n_iter = 50),n_jobs=-1)
    model.fit(data_train, label.target[:data_train.shape[0]])
    res1  =model.predict(data_test)

    metrics.f1_score(label.target[data_train.shape[0]:data_train.shape[0]+data_test.shape[0]], res1 , average="micro")


