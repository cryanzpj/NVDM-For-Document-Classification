import tensorflow as tf
import numpy as np
import os,time, datetime
import scipy.io as io
from tensorflow.contrib import learn
from scipy.sparse import vstack
import itertools
import glob
from sklearn.metrics import f1_score



###
tf.flags.DEFINE_string("data_path",'../data_train.npz',"path for training data")
tf.flags.DEFINE_integer("vocab_size",10000,"Vocanulary size")
tf.flags.DEFINE_integer("hid_size", 500, "Dimensionality hidden layer size (default: 500)")
tf.flags.DEFINE_integer("lat_size", 200, "Dimensionality latent layer size (default: 200, alter: 50)")
tf.flags.DEFINE_integer("num_class", 103, "Number of class (default: 2)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularizaion lambda (default: 0.5)")
tf.flags.DEFINE_float("train_size", 0.9, "training data size")
tf.flags.DEFINE_float("learning_rate", 5e-2/10000, "Learning Rate default: 5e-2 ")
tf.flags.DEFINE_float("momentum", 0.9, "SGD Momentum ")

# Training parameters
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("train_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 1000)")
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps (default: 10000)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

Relu = tf.nn.relu

def display_para():
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

def load_data():
    #data loding
    if os.path.isfile(FLAGS.data_path):
        f = np.load(FLAGS.data_path)
        data,label = f['data'],f['label']
    data = data.tolist()
    label = label.tolist()
    n_data = data.shape[0]

    #train/val split
    index = np.random.permutation(xrange(n_data))
    data_train,label_train = data.toarray()[index[:int(n_data*FLAGS.train_size)]], label.toarray()[index[:int(n_data*FLAGS.train_size)]]
    data_val,label_val = data.toarray()[index[int(n_data*FLAGS.train_size):]],label.toarray()[index[int(n_data*FLAGS.train_size):]]
    return data_train, label_train.astype(np.float32), data_val, label_val.astype(np.float32)

def f1_score_multiclass(pred,label):
    if len(pred) != len(label):
        print('inconsistant shape' )
        return
    else:
        score = []
        for i in range(pred.shape[1]):
            score.append(f1_score(label[:,i],pred[:,i]))
        return np.mean(score)



def Linear(x, out_shape, name="linear"):
    '''
    input:
    out_shape: single number, indicates the shape after linear transformation
    '''
    with tf.variable_scope(name):
        w = tf.get_variable(name="w", shape=[x.get_shape()[-1], out_shape[0]],\
                    initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable(name="b", shape=[out_shape[0]],\
                    initializer=tf.truncated_normal_initializer(stddev=0.02))
    l2_loss = tf.reduce_sum(tf.square(w)) + tf.reduce_sum(tf.square(b))
    return tf.matmul(x,w) + b, l2_loss



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



class mlp():
    def __init__(self):
        '''hyperparameters'''
        self.learning_rate = FLAGS.learning_rate
        self.hidden_size = FLAGS.hid_size
        self.vocab_size = FLAGS.vocab_size
        self.num_class = FLAGS.num_class
        self.emb_size = FLAGS.lat_size # or 200 # latent dim
        self.l2_lambda = FLAGS.l2_reg_lambda

        self.input_x = tf.placeholder(tf.float32, [None,self.vocab_size], name="input_x") # 1*10000, 1doc
        self.input_y = tf.placeholder(tf.float32, [None,self.num_class], name="input_y") # 1*103, 1 doc
        #self.input_y = tf.placeholder(tf.float32, [self.num_class], name="input_y")
        self.one_hot = tf.cast(tf.one_hot(tf.cast(self.input_y,tf.int32),2,0,1),tf.float32 )# 103*2


        with tf.name_scope('classification'):
            h1, h1_l2 = Linear(self.input_x,[1000],name = 'projection_1')
            h2, h2_l2 = Linear(h1,[300],name = 'projection_2')
            loss_pool = []
            pool = [] #construct 103 binary classifier
            # self.scores, h3_l2 = Linear(h2,[self.num_class],name = 'projection_3')
            for i in range(self.num_class):
                t1, t2 = Linear(h2,[2],name = "class_"+str(i))
                pool.append(t1)
                loss_pool.append(t2)

            self.pool_flat = tf.reshape(pool, [FLAGS.batch_size,self.num_class, -1])
            self.l2_loss_all = (h1_l2 +h2_l2 + tf.reduce_sum(loss_pool))
            self.scores = tf.nn.softmax_cross_entropy_with_logits(self.pool_flat, self.one_hot)
            self.predicts = tf.argmax(tf.nn.sigmoid_cross_entropy_with_logits(self.pool_flat, self.one_hot),2)
            # self.predictions = tf.cast(tf.reshape(tf.nn.top_k(self.scores, 4 )[1],[1,-1]),tf.int32)
            # set_y = tf.cast(tf.reshape(tf.where(tf.not_equal(self.input_y,tf.constant(0, dtype=tf.float32))),[1,-1]),tf.int32)
            # intersection = tf.contrib.metrics.set_intersection(self.predictions,set_y)
            # union = tf.contrib.metrics.set_union(self.predictions,set_y)
            # self.accuracy = tf.convert_to_tensor(tf.cast(tf.contrib.metrics.set_size(intersection),tf.float32) /tf.cast(tf.contrib.metrics.set_size(union),tf.float32) )
            #
        with tf.name_scope('loss'):
            self.classification_loss = tf.reduce_mean(self.scores)
            self.loss =  self.l2_lambda * self.l2_loss_all + tf.reduce_mean(self.classification_loss)
            #self.loss = self.l2_lambda * self.l2_loss_all + tf.reduce_mean(self.classification_loss)

        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate,beta1=0.8)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in self.grads_and_vars]
        self.train_op = self.optimizer.apply_gradients(self.capped_gvs, global_step=self.global_step)


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        data_train, label_train, data_val, label_val = load_data()
        sess.run(tf.initialize_local_variables())

        nvdm  = mlp()
        # Define Training procedure
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
        print("Writing to {}\n".format(out_dir))
        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", nvdm.loss)
        # Train Summaries
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
        # Dev summaries
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch,epoch,predicts,labels):
            """
            A single training step
            """
            feed_dict = {
                  nvdm.input_x: x_batch,
                  nvdm.input_y:y_batch,
                }

            _, step,  loss,predict = sess.run([nvdm.train_op, nvdm.global_step, nvdm.loss,nvdm.predicts], feed_dict)


            time_str = datetime.datetime.now().isoformat()
            if step % FLAGS.train_every == 0:
                # import pdb
                # pdb.set_trace()

                score = f1_score_multiclass(np.array(predicts).reshape((-1,103)),np.array(labels).reshape(-1,103))
                print("time: {},  epoch: {}, step: {}, loss: {:g}, score: {:g}".format(time_str,epoch, step, loss,score))

                return [],[]


            predicts.append(predict)
            labels.append(y_batch.astype(int))
            return predicts,labels


        def dev_step(x_batch, y_batch):
            """
            Evaluates model on a dev set
            """
            feed_dict = {nvdm.input_x: x_batch,nvdm.input_y:y_batch}

            step, loss= sess.run(
                [nvdm.global_step, nvdm.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("time: {}, step: {}, loss {:g},".format(time_str, step, loss))



        # Generate batches
        batches = batch_iter(
            list(zip(data_train, label_train)), FLAGS.batch_size, FLAGS.num_epochs)
        num_batches_per_epoch = int(label_train.shape[0]/FLAGS.batch_size) + 1

        current_step = 1


        predicts,labels = [],[]

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch = np.array(x_batch) #64 * 10000
            y_batch = np.array(y_batch) #64 * 103
            predicts, labels = train_step(x_batch, y_batch,1+ int(current_step//num_batches_per_epoch),predicts,labels)
            current_step = tf.train.global_step(sess, nvdm.global_step)
            # if current_step % FLAGS.evaluate_every == 0:
            #     print("\nEvaluation:")
            #     dev_step(x_val, y_val, writer=dev_summary_writer)
            #     print("")
            # if current_step % FLAGS.checkpoint_every == 0:
            #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #     print("Saved model checkpoint to {}\n".format(path))








