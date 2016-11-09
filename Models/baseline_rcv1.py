"""
======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how scikit-learn can be used to classify documents
by topics using a bag-of-words approach. This example uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

The dataset used in this example is the 20 newsgroups dataset. It will be
automatically downloaded, then cached.

The bar plot indicates the accuracy, training time (normalized) and test time
(normalized) of each classifier.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import pickle

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


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--data_home",
              action="store_true", dest="data_home",
              help="The directory that contain RCV1 dataset. default is the current dir.")

op.add_option("--macro",
               dest="macro", default=True,
               help="Use macro f1 or micro f1. default is True, use macro f1.")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


###############################################################################
if opts.macro == True:
    average_option = "macro"
else:
    average_option = "micro"
if opts.data_home:
    data_home = opts.data_home
else:
    data_home = "."
data_train = fetch_rcv1(data_home = data_home, subset = "train")
data_test = fetch_rcv1(data_home = data_home, subset = "test")
print('data loaded')



# split a training set and a test set
X_train, X_test = data_train.data, data_test.data
y_train, y_test = data_train.target, data_test.target
###############################################################################
# Benchmark classifiers
def benchmark(clf, name = ""):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    if name == "kNN":
        pred = clf.predict(X_test[:200][:])
        score = metrics.f1_score(y_test[:200][:], pred, average=average_option)
    else:
        pred = clf.predict(X_test)
        score = metrics.f1_score(y_test, pred, average=average_option)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
        print()

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))
    print()
    clf_descr = str(clf).split('(')[1].split("=")[-1]
    return clf_descr, score, train_time, test_time

results = []
for clf, name in (
        (OneVsRestClassifier(Perceptron(n_iter=50)), 
            "Perceptron"),
        (OneVsRestClassifier(PassiveAggressiveClassifier(n_iter=50)), 
            "Passive-Aggressive"),
        (OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10)), 
            "kNN")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf, name))
for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(OneVsRestClassifier(LinearSVC(C=1.0,
                          loss='squared_hinge', 
                          penalty=penalty, dual=False, tol=1e-3))))

    # Train SGD model
    results.append(benchmark(OneVsRestClassifier(SGDClassifier(alpha=.0001, 
                          n_iter=50, penalty=penalty))))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(OneVsRestClassifier(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet"))))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
#results.append(benchmark(OneVsRestClassifier(NearestCentroid())))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(OneVsRestClassifier(MultinomialNB(alpha=.01))))
results.append(benchmark(OneVsRestClassifier(BernoulliNB(alpha=.01))))




# save the result to .pkl

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)
print("Save results to result_rcv1_" + average_option + ".pkl.")
file = open("result_rcv1_"+average_option+".pkl", "wb")
pickle.dump(results, file)
file.close()

