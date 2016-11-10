import numpy as np
from make_corp import *
import gensim
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics

from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.datasets import  make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier



def train_lsi(corp,id2word,n_topic =100):
    lsi = gensim.models.lsimodel.LsiModel(corpus=np.array(corp),id2word=id2word,num_topics = n_topic)
    #lda = gensim.models.hdpmodel.HdpModel(corp, id2word=id2word)

    res = np.zeros((len(corp),n_topic))
    for i,j in enumerate(corp):
        temp = np.asarray(lsi[j])
        res[i,temp[:,0].astype(int)] = temp[:,1]
    return res

if __name__ == "__main__":
    file_dir = "../data/RCV1/token/lyrl2004_tokens_train.dat"
    vocab_dir =  "../data/RCV1/token/lyrl2004_tokens_train.txt"

    #load data and convert to corpus
    vocab = vocab_to_dict(vocab_dir)
    corp = make_corp(file_dir,vocab)
    id2word = dict(np.array(vocab.items(),dtype=object)[:,::-1])

    #train lda model and get document representations
    lsi = train_lda(corp,id2word,200)
    # lda.get_document_topics(corp,minimum_probability= 0.0)


    res1 =  OneVsRestClassifier(LinearSVC()).fit(lsi, data_train.target).predict(lsi)
    metrics.f1_score(data_train.target, res1 , average="micro")

    vector_model = LsiModel(corpus=np.array(corp), num_topics=100, id2word=id2word)

    data_train.target[2001]