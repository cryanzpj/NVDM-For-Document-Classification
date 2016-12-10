import numpy as np
import sys
import tensorflow as tf
import os

from sklearn.datasets import fetch_rcv1
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier

def data_prepare(name):
    print("Loading data")
    if os.path.isfile(name):
        f = np.load(name)
        data,label = f['data'],f['label']
    data = data.tolist()
    label = label.tolist()
    n_data = data.shape[0]
    
    train_size = 0.9
    #train/val split
    index = np.array(xrange(n_data))
    data_train,label_train = data.toarray()[index[:int(n_data*train_size)]], label.toarray()[index[:int(n_data*train_size)]]
    data_val,label_val = data.toarray()[index[int(n_data*train_size):]],label.toarray()[index[int(n_data*train_size):]]
    return data_train, label_train.astype(np.float32), data_val, label_val.astype(np.float32)



def f1(pred,label):
    tp = tf.cast(tf.reduce_sum(tf.mul(tf.cast(tf.equal(pred,1),tf.int32) ,tf.cast(tf.equal(label,1),tf.int32)),0),tf.float32)
    fn = tf.cast(tf.reduce_sum(tf.mul(tf.cast(tf.equal(pred,0),tf.int32) ,tf.cast(tf.equal(label,1),tf.int32)),0),tf.float32)
    fp = tf.cast(tf.reduce_sum(tf.mul(tf.cast(tf.equal(pred,1),tf.int32) ,tf.cast(tf.equal(label,0),tf.int32)),0),tf.float32)
    recall= tp/(tp+fn)
    prec = tp/(tp+fp)
    f1 = 2 * (prec*recall)/(prec + recall)
    return f1


def thres_search(data,label,n):
    res = []
    for i in range(n):
        n_label = tf.cast(tf.reduce_sum(label[i]),tf.int32)
        temp = tf.mul(data[i],label[i])
        temp = tf.reshape(tf.nn.top_k(temp,n_label +1).values,[1,1,-1,1])
        thres = tf.reshape(tf.contrib.layers.avg_pool2d(temp,[1,2],[1,1]),[-1,1])
        predicts = tf.map_fn(lambda x: tf.cast(tf.greater_equal(data[i],x),tf.float32),thres)
        f1_scores = tf.map_fn(lambda x: f1(x,label[i]),predicts)
        thres_opt = thres[tf.cast(tf.arg_max(f1_scores,0),tf.int32)]
        res.append(thres_opt)
        # R = tf.map_fn(lambda x: tf.contrib.metrics.streaming_recall(x,label[i])[0],predicts)
        # P = tf.map_fn(lambda x: tf.contrib.metrics.streaming_precision(x,label[i])[0],predicts)
        #thres_opt = thres[np.argsort(map(lambda x:  metrics.f1_score(x,sess.run(label[i]),average = "macro") ,predicts))[-1]]

    return tf.reshape(res,[-1])




def SVMbanchmark(X_train, y_train, X_test, y_test):
    # optimial c is 10.0, f1 = 0.52
    print("Training LinearSVC with l1-based feature selection")
    X_valid, y_valid = X_test[:10000], y_test[:10000]
    score_list = []
    CList = [0.1, 0.5, 1, 10, 50, 100]
    for c in CList:
        clf = OneVsRestClassifier(LinearSVC(C=c, penalty='l1', dual=False))
        clf.fit(X_train, y_train)
         
        pred = clf.predict(X_valid)
        score = metrics.f1_score(y_valid, pred, average="macro")
        score_list.append(score)
        print("f1-score: {:f}, c is {:f}".format(score, c))
    clf = OneVsRestClassifier(LinearSVC(penality="l1", dual=False, C=CList[np.argmax(score_list)]))
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = metrics.f1_score(y_test, pred, average="micro")
    print("f1-score for test set: {:f}".format(score))

def Linear(x, out_shape, name="linear", l2 = True):
    '''
    input:
    out_shape: single number, indicates the shape after linear transformation
    '''
    with tf.variable_scope(name):
        w = tf.get_variable(name="w", shape=[x.get_shape()[-1], out_shape[0]],\
                initializer=tf.truncated_normal_initializer(stddev=0.05))
        b = tf.get_variable(name="b", shape=[out_shape[0]],\
                    initializer=tf.truncated_normal_initializer(stddev=0.05))
    l2_loss = tf.reduce_sum(tf.square(w)) + tf.reduce_sum(tf.square(b))
    if l2:
        return tf.matmul(x,w) + b, l2_loss    
    else: 
        return tf.matmul(x,w) + b


class mlp():
    def __init__(self):
        '''hyperparameters'''
        self.learning_rate = 0.0001
        self.hidden_size = 2000
        self.num_class = 103
        self.emb_size = 200 # or 200 # latent dim
        self.l2_lambda = 0.1
        self.feature_size = 10000 
        self.batch_size = 32
        self.epoch = 8
        self.display_score = 10
        self.sess = tf.Session()

        self.input_x = tf.placeholder(tf.float32, [None,self.feature_size], name="input_x") # 1*10000, 1doc
        self.input_y = tf.placeholder(tf.float32, [None,self.num_class], name="input_y") # 1*103, 1 doc
        #self.threshold_t = tf.Variable(tf.constant(0.3, shape=[1]))

        with tf.name_scope("classification"):
            self.h1, self.h1_l2 = Linear(self.input_x, [self.hidden_size], name="projection_1")
            dh1 = tf.nn.relu(self.h1)
            self.h2, self.h2_l2 = Linear(dh1, [103], name="projection_2")
            self.dh2 = self.h2
            # multilabel priedictor
            self.mp, hmp_l2 = Linear(self.h1, [1], name="multilabel_predictor")
            #self.threshold_t = tf.Variable(tf.constant(0.3, shape=[1]))

            # threshold
            self.softmax_dh2 = tf.nn.softmax(self.dh2)
            self.threshold_t = thres_search(self.softmax_dh2,self.input_y,self.batch_size)
            self.t_loss= tf.reduce_mean(tf.square(self.mp - self.threshold_t), 0)
            self.l2_loss = self.l2_lambda*(hmp_l2)
            self.threshold_loss = self.t_loss + self.l2_loss

            self.t_tiled = tf.transpose(tf.reshape(tf.tile(self.threshold_t, tf.pack([self.num_class])),[-1,self.batch_size]))

            #cond = tf.less(self.softmax_dh2, tf.reshape(self.t_tiled, [-1, self.num_class]))
            cond = tf.less(self.softmax_dh2, self.t_tiled)
            Ones = tf.ones([self.batch_size, self.num_class])
            Zeros = tf.zeros([self.batch_size, self.num_class])
            self.predict = tf.select(cond, Zeros, Ones)

            self.score = tf.nn.softmax_cross_entropy_with_logits(self.dh2, self.input_y) #64*103

        with tf.name_scope('loss'):
            self.classification_loss = tf.reduce_mean(self.score)
            self.loss = 0.1 * tf.reduce_mean(self.classification_loss)
            #self.loss = self.l2_lambda * self.l2_loss_all + tf.reduce_mean(self.classification_loss)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1=0.8).minimize(self.loss)
        self.optimizer_threshold = tf.train.AdamOptimizer(self.learning_rate,beta1=0.8).minimize(self.threshold_loss)# + self.loss)

        self.init = tf.initialize_all_variables()
        #self.sess.run(tf.initialize_local_variables())
        self.sess.run(self.init)

    def train(self, X_train, y_train):
        self.sess.run(self.init)
        total_batch = X_train.shape[0] // self.batch_size 
        for e in range(self.epoch):

            for i in range(total_batch):
                X_batch = X_train[i*self.batch_size:(i+1)*self.batch_size]
                y_batch = y_train[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {
                        self.input_x : X_batch,
                        self.input_y : y_batch
                        }
                _,_,  loss, t_loss, l2_loss, predict =  self.sess.run([
                            self.optimizer, 
                            self.optimizer_threshold,
                            self.loss,
                            self.t_loss,
                            self.l2_loss,
                            self.predict], feed_dict)
                if i == 30:
                    # import pdb
                    # pdb.set_trace()
                    h1, h2, mp, softmax_dh2, t = self.sess.run([
                        self.h1,
                        self.h2,
                        self.mp,
                        self.softmax_dh2,
                        self.threshold_t], feed_dict)
                if i % self.display_score == 0:
                    score = metrics.f1_score(np.array(predict).reshape(-1,103), y_batch.reshape(-1,103), average="micro")
                    score_macro = metrics.f1_score(np.array(predict).reshape(-1,103), y_batch.reshape(-1,103), average="macro")
                    print("batch: {}, loss: {:f}, t_loss: {:.4f}, l2_loss: {:.4f}, mi: {:.4f}, ma: {:.4f}".format(e, i, loss,t_loss[0], l2_loss, score, score_macro))


    def test(self, X_test, y_test):
        print("Testing")        
        total_batch = X_test.shape[0] // self.batch_size
        test_pred = []
        for i in range(total_batch):
            X_batch = X_test[i*self.batch_size:(i+1)*self.batch_size]
            y_batch = y_test[i*self.batch_size:(i+1)*self.batch_size]
            feed_dict = {
                    self.input_x : X_batch,
                    self.input_y : y_batch
                    }
                
            predict =  self.sess.run([
                            self.predict], feed_dict)
            if len(test_pred) == 0:
                test_pred = predict[0].tolist()
            else:
                test_pred.extend(predict[0].tolist())

            if i % 100 == 0:

                score = metrics.f1_score(np.array(predict).reshape(-1,103), 
                    y_batch, average="macro")
                print("Test: batch: {}, score: {:f}".format(i, score))


        score_test = metrics.f1_score(np.array(test_pred).reshape(-1,103),
                y_test[:total_batch*self.batch_size].reshape(-1,103), average="micro")
        score_test_ma = metrics.f1_score(np.array(test_pred).reshape(-1,103),
                y_test[:total_batch*self.batch_size].reshape(-1,103), average="macro")
        print("mi score for test is {:.4f}, ma score is {:4f}".format(score_test, score_test_ma))



def main():
    X_train, y_train, X_test, y_test = data_prepare("../data_train.npz")
    #SVMbanchmark(X_train, y_train, X_test, y_test)
    nn = mlp()
    nn.train(X_train, y_train)
    nn.test(X_test, y_test)

if __name__ == "__main__":
    main()

