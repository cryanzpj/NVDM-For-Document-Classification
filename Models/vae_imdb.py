import numpy as np
import sys
import tensorflow as tf
import os
import itertools

from scipy.sparse import vstack
from sklearn.datasets import fetch_rcv1
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

Relu = tf.nn.relu
Dropout = tf.nn.dropout
def imdb_load_data():
    X_train_path = "/scratch/yg1281/inf_proj/data/imdb_train.npy"
    y_train_path = "/scratch/yg1281/inf_proj/data/imdb_train_label.npy"
    X_test_path = "/scratch/yg1281/inf_proj/data/imdb_test.npy"
    y_test_path = "/scratch/yg1281/inf_proj/data/imdb_test_label.npy"
    X = np.load(X_train_path)
    y = np.load(y_train_path)
    X = vstack(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y,test_size = 0.3, random_state = 1)
    print("Data loaded")
    return  X_train, X_val, y_train, y_val 


def SVMbanchmark(X_train, X_test, y_train, y_test):
    # optimial c is 10.0, f1 = 0.52
    print("Training LinearSVC with l1-based feature selection")
    import pdb
    pdb.set_trace()
    X_valid, y_valid = X_test[:10000], y_test[:10000]
    score_list = []
    CList = [0.1, 0.5, 1, 10, 50, 100]
    for c in CList:
        clf = LinearSVC(C=c, penalty='l1', dual=False)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_valid)
        score = metrics.accuracy_score(y_valid, pred)
        score_list.append(score)
        print("f1-score: {:f}, c is {:f}".format(score, c))
    clf = LinearSVC(penality="l1", dual=False, C=CList[np.argmax(score_list)])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
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
        self.learning_rate = 0.001
        self.hidden_size = 300
        self.num_class = 1 
        self.emb_size = 200 # or 200 # latent dim
        self.l2_lambda = 0.0001
        self.feature_size = 89527 
        self.batch_size = 1000
        self.epoch = 2
        self.display_score = 10
        
        self.sess = tf.Session()
        self.encode()
        self.decode()
        self.build_model()

        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def encode(self):
        self.input_x = tf.placeholder(tf.float32, [None,self.feature_size], name="input_x") # 1*89527
        self.x_id = tf.placeholder(tf.int32, [None], name='x_id')

        with tf.name_scope("encoding"):
            h1_, h1_l2 = Linear(self.input_x, [self.hidden_size], name="h1")
            h1 = Relu(h1_)
            h2_, h2_l2 = Linear(h1, [self.emb_size], name="h2")
            h2 = Relu(h2_)
            self.hmean, hmean_l2 = Linear(h2, [self.emb_size], name="hmean")
            self.hlogvar, hlogvar_l2 = Linear(h2, [self.emb_size], name="hlogvar")
            epsilon = tf.random_normal([self.emb_size])
            self.z = self.hmean + tf.sqrt(tf.exp(self.hlogvar)) * epsilon
            self.l2_loss_all = self.l2_lambda * (h1_l2 + h2_l2 + hmean_l2 + hlogvar_l2)

    def decode(self):
        with tf.name_scope("decoding"):
            self.e, _ = Linear(self.z, [self.feature_size])
            self.p_xi_h = tf.squeeze(tf.nn.softmax(self.e))


    def build_model(self):

        self.input_y = tf.placeholder(tf.float32, [None,self.num_class], name="input_y") # 1*1, 1doc
        self.one_hot = tf.reshape(tf.cast(tf.one_hot(tf.cast(self.input_y, tf.int32), 2,0,1), tf.float32), [-1,2])

        
        self.recon_loss = -tf.reduce_sum(tf.log(0.0001 + tf.gather(self.p_xi_h, self.x_id)))
        self.KL = -0.5 * tf.reduce_sum(1.0 + self.hlogvar - tf.pow(self.hmean, 2)\
                  - tf.exp(self.hlogvar), reduction_indices = 1)
        self.loss = tf.reduce_mean(0.0001 * self.KL + self.recon_loss)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,0.9)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in self.grads_and_vars] 
        self.train_op = self.optimizer.apply_gradients(self.capped_gvs)
        #self.optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1=0.9).minimize(self.loss)

        self.init = tf.initialize_all_variables()
        self.sess.run(self.init)

    def train(self, X_train, y_train):
        #self.saver.restore(self.sess, "./imdbmodel/model.ckpt")
        total_batch = X_train.shape[0] // self.batch_size 
        for e in range(self.epoch):
            perplist = []
            for i in range(total_batch):
                X_batch = X_train[i*self.batch_size:(i+1)*self.batch_size]
                y_batch = y_train[i*self.batch_size:(i+1)*self.batch_size]
                x_batch_id = [_ for _ in itertools.compress(range(self.feature_size), map(lambda x : x>0, X_batch[0].toarray()[0]))]
                feed_dict = {
                        self.input_x : X_batch.toarray(),
                        self.input_y : np.reshape(y_batch, [-1,1]),
                        self.x_id : x_batch_id
                        }
                _, loss =  self.sess.run([
                            self.train_op, 
                            self.loss], feed_dict)
                if np.isnan(loss):
                    import pdb
                    pdb.set_trace()
                if i % self.display_score == 0:
                    p_xi_h = self.sess.run([self.p_xi_h], feed_dict)
                    valid_p = np.mean(np.log(p_xi_h[0][x_batch_id]))
                    perplist.append(valid_p)
                    print("step: {}, perp: {:f}".format(i, np.exp(-np.mean(perplist))))
            # save model every epoch
                if i > 0 and i % 2000 == 0:
                    self.savemodel()

    

    def test(self, X_test, y_test):
        print("Testing")        
        y_all = []
        total_batch = X_test.shape[0] // self.batch_size
        test_pred = []
        for i in range(total_batch):
            X_batch = X_test[i*self.batch_size:(i+1)*self.batch_size]
            y_batch = y_test[i*self.batch_size:(i+1)*self.batch_size]
            x_batch_id = [_ for _ in itertools.compress(range(self.feature_size), map(lambda x : x>0, x_batch[0]))]
            feed_dict = {
                    self.input_x : X_batch.toarray(),
                    self.input_y : np.reshape(y_batch, [-1,1]),
                    self.x_id : x_batch_id 
                    }
                
            predict =  self.sess.run([self.logits], feed_dict)
            if len(test_pred) == 0:
                test_pred = np.argmin(predict[0], axis=1).tolist()
                y_all = y_batch.tolist()
            else:
                test_pred.extend(np.argmin(predict[0], axis=1).tolist())
                y_all.extend(y_batch.tolist())

            if i % 100 == 0:
                score = metrics.accuracy_score(np.argmin(predict[0], axis=1), y_batch)
                print("Test: batch: {}, score: {:f}".format(i, score))


        score_test = metrics.accuracy_score(test_pred, y_all)
        print("acc score for test is {:.4f}".format(score_test))


    def savemodel(self):
        save_path = self.saver.save(self.sess, "./imdbmodel/model.ckpt")


    def applymodel(self, X_train, X_test, y_train, y_test):
        self.saver.restore(self.sess, "./imdbmodel/model.ckpt")
        total_batch = X_train.shape[0] // self.batch_size
        X_train_emb = []; y_train_emb = [] 
        for i in range(total_batch):
            print i
            X_batch = X_train[i*self.batch_size:(i+1)*self.batch_size]
            
            y_batch = y_train[i*self.batch_size:(i+1)*self.batch_size]
            feed_dict = {
                    self.input_x : X_batch.toarray()
                    }
            emb = self.sess.run([self.z], feed_dict)
            if len(X_train_emb) == 0:
                X_train_emb = emb[0].tolist()
                y_train_emb = y_batch.tolist()
            else:
                X_train_emb.extend(emb[0].tolist())
                y_train_emb.extend(y_batch.tolist())

        total_batch = X_test.shape[0] // self.batch_size
        X_test_emb = []; y_test_emb = []
        for i in range(total_batch):
            print i
            X_batch = X_test[i*self.batch_size:(i+1)*self.batch_size]
            y_batch = y_test[i*self.batch_size:(i+1)*self.batch_size]
            feed_dict = {
                    self.input_x : X_batch.toarray()
                    }
            emb = self.sess.run([self.z], feed_dict)
            print(emb[0].shape)
            if len(X_test_emb) == 0:
                X_test_emb = emb[0].tolist()
                y_test_emb = y_batch.tolist()
            else:
                X_test_emb.extend(emb[0].tolist())
                y_test_emb.extend(y_batch.tolist())
        return X_train_emb, X_test_emb, y_train_emb, y_test_emb

def main():
    X_train, X_test, y_train, y_test = imdb_load_data()
    #SVMbanchmark(X_train, y_train, X_test, y_test)
    nn = mlp()
    #nn.train(X_train, y_train)
    #nn.test(X_test, y_test)
    X_train_emb, X_test_emb, y_train_emb, y_test_emb = nn.applymodel(X_train, X_test, y_train, y_test)
    SVMbanchmark(X_train_emb, X_test_emb, y_train_emb, y_test_emb)

if __name__ == "__main__":
    main()

