'''
This script implement NVDM
'''
import tensorflow as tf
import numpy as np
import os,time, datetime
import scipy.io as io
from tensorflow.contrib import learn
from scipy.sparse import vstack

###
tf.flags.DEFINE_string("data_path",'./data/data_train.npz',"path for training data")
tf.flags.DEFINE_integer("vocab_size",10000,"Vocanulary size")
tf.flags.DEFINE_integer("embedding_dim", 500, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("num_class", 103, "Number of class (default: 2)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("train_size", 0.9, "training data size")
tf.flags.DEFINE_float("learning_rate", 5e-2/10000, "Learning Rate default: 5e-2 ")
tf.flags.DEFINE_float("momentum", 0.9, "SGD Momentum ")
tf.flags.DEFINE_float("drop_out", 0.5, "drop out keep rate ")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 250, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("saving_local", False, "Whether to  save data in local for easy access")
tf.flags.DEFINE_boolean("pretrain", False, "Whether to use pre-trained word2vec to initialize embeeding matrix")

###

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


#data loding
if os.path.isfile(FLAGS.data_path):
    f = np.load(FLAGS.data_path)
    data, label = f['data'],f['label']
data = data.tolist()
label = label.tolist()
n_data = data.shape[0]

Relu = tf.nn.relu

#train/val split
index = np.random.permutation(xrange(n_data))
data_train,label_train = data[index[:int(n_data*FLAGS.train_size)].tolist()], label[index[:int(n_data*FLAGS.train_size)].tolist()]
data_val,label_val = data[index[int(n_data*FLAGS.train_size):]],label[index[int(n_data*FLAGS.train_size):]]

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
        #XXX b initializer might change
    return tf.matmul(x,w) + b

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




class nvdm():
    def __init__(self,lr = 0.001,hs = 500,vocab_size = 10000,emb_size = 50,num_class = 103, n_epoch = 10, N = 10000):
        '''hyperparameters'''
        self.epochs = n_epoch
        self.batch_size = 128
        self.learning_rate = lr
        self.hidden_size = hs
        self.vocab_size = vocab_size
        self.num_class = num_class
        self.emb_size = emb_size # or 200 # latent dim
        self.N = N #10000
        self.R = tf.Variable(tf.truncated_normal([self.emb_size, self.vocab_size],\
                 stddev=0.002, name="r"))
        self.b_r = tf.Variable(tf.truncated_normal([self.vocab_size], \
                 stddev=0.002, name="br"))
        
        '''encode'''
        self.input_x = tf.placeholder(tf.float32, [None, self.vocab_size], name="input_x") # 1*10000, 1doc
        #self.input_y = tf.placeholder(tf.float32, [None, self.num_class], name="input_y")

        with tf.name_scope("encoding"):

            h1 = Relu(Linear(self.input_x, [self.hidden_size], name="h1"))
            h2 = Relu(Linear(h1, [self.hidden_size], name="h2"))  #1*hidden_Size
            self.hmean = Linear(h2, [self.emb_size], name="hmean")
            self.hlogvar = Linear(h2, [self.emb_size], name="hlogvar") #1*emb_size = 50 # XXX

            epsilon = tf.Variable(tf.random_normal([self.emb_size]))
            z = self.hmean + tf.sqrt(tf.exp(self.hlogvar))*epsilon  # 1*emb_size
    
        '''decode'''
        with tf.name_scope("decoding"):  # FUCK FUCK FUCKING
            #self.x_i = tf.placeholder(tf.float32, [self.vocab_size, None]) #10000*N
            #R:50*10000
            x_i = tf.Variable(initial_value=np.identity(self.vocab_size), dtype='float32')
            e = tf.matmul(tf.matmul(-z, self.R), x_i) + self.b_r #e:vocab_size*N
            p_xi_h = tf.nn.log_softmax(e, dim=0) * self.input_x # vocab_size*N


        with tf.name_scope('loss'):
            self.recon_loss = tf.reduce_sum(p_xi_h)
            self.KL = -0.5 * tf.reduce_sum(1.0 + self.hlogvar-tf.pow(self.hmean, 2) \
                      - tf.exp(self.hlogvar), reduction_indices=1)
            self.loss = tf.reduce_mean(0.0001 * self.KL + self.recon_loss)


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        nvdm  = nvdm()
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #self.optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1=0.8).minimize(self.loss)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate,beta1=0.8)
        grads_and_vars = optimizer.compute_gradients(nvdm.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
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

        def train_step(x_batch, y_batch,epoch):
            """
            A single training step
            """
            feed_dict = {
                  nvdm.input_x: x_batch,
                }

            _, step, summaries, loss,  = sess.run(
                [train_op, global_step, loss_summary, nvdm.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("time: {},  epoch: {}, step: {}, loss: {:g}".format(time_str,epoch, step, loss))
            if step % 50 == 0:
                train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                  nvdm.input_x: x_batch,
                }

            step, summaries, loss= sess.run(
                [global_step, dev_summary, nvdm.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, global {:g}".format(time_str, step, loss))
            if writer:
                writer.add_summary(summaries, step)


        # Generate batches
        batches = batch_iter(
            list(zip(data_train, label_train)), FLAGS.batch_size, FLAGS.num_epochs)
        num_batches_per_epoch = int(label_train.shape[0]/FLAGS.batch_size) + 1

        current_step = 1
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch = vstack(x_batch).toarray() #64 * 10000
            y_batch = vstack(y_batch).toarray() #64 * 103
            train_step(x_batch, y_batch,1+ int(current_step//num_batches_per_epoch))
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_val, y_val, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))





        
    
