'''
This script implement NVDM
'''
import tensorflow as tf
import numpy as np
import os,time, datetime
import scipy.io as io
from tensorflow.contrib import learn
from scipy.sparse import vstack
import itertools
import glob
###
tf.flags.DEFINE_string("data_path",'./data/data_train.npz',"path for training data")
tf.flags.DEFINE_integer("vocab_size",10000,"Vocanulary size")
tf.flags.DEFINE_integer("hid_size", 500, "Dimensionality hidden layer size (default: 500)")
tf.flags.DEFINE_integer("lat_size", 200, "Dimensionality latent layer size (default: 200, alter: 50)")
tf.flags.DEFINE_integer("num_class", 103, "Number of class (default: 2)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularizaion lambda (default: 0.5)")
tf.flags.DEFINE_float("train_size", 0.9, "training data size")
tf.flags.DEFINE_float("learning_rate", 5e-2/1000, "Learning Rate default: 5e-2 ")
tf.flags.DEFINE_float("momentum", 0.9, "SGD Momentum ")

# Training parameters
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
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
    return data_train, label_train, data_val, label_val


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




class nvdm():
    def __init__(self):
        '''hyperparameters'''
        self.learning_rate = FLAGS.learning_rate
        self.hidden_size = FLAGS.hid_size
        self.vocab_size = FLAGS.vocab_size 
        self.num_class = FLAGS.num_class
        self.emb_size = FLAGS.lat_size # or 200 # latent dim
        self.l2_lambda = FLAGS.l2_reg_lambda
        self.R = tf.Variable(tf.truncated_normal([self.emb_size, self.vocab_size],\
                 stddev=0.002, name="r"))
        self.b_r = tf.Variable(tf.truncated_normal([self.vocab_size], \
                 stddev=0.002, name="br"))
        self.encode()
        self.decode()
        self.build_model()


    def encode(self):
        '''encode'''
        self.input_x = tf.placeholder(tf.float32, [None,self.vocab_size], name="input_x") # 1*10000, 1doc
        self.x_id = tf.placeholder(tf.int32, [None], name = "x_id")
        #self.input_y = tf.placeholder(tf.float32, [self.num_class], name="input_y")

        with tf.name_scope("encoding"):
            h1_, h1_l2 = Linear(self.input_x, [self.hidden_size], name="h1")
            h1 = Relu(h1_)
            h2_, h2_l2 = Linear(h1, [self.hidden_size], name="h2")  #1*hidden_Size
            h2 = Relu(h2_)
            self.hmean, hmean_l2 = Linear(h2, [self.emb_size], name="hmean")
            self.hlogvar, hlogvar_l2 = Linear(h2, [self.emb_size], name="hlogvar") #1*emb_size = 50 # XXX
            epsilon = tf.Variable(tf.random_normal([self.emb_size]))
            self.z = self.hmean + tf.sqrt(tf.exp(self.hlogvar))*epsilon  # 1*emb_size
            self.l2_loss_all = self.l2_lambda * (h1_l2 + h2_l2 + hmean_l2 + hlogvar_l2)



    def decode(self):
        '''decode'''
        with tf.name_scope("decoding"):   
            #self.x_i = tf.placeholder(tf.float32, [self.vocab_size, None]) #10000*N
            #R:50*10000
            self.e = -tf.matmul(self.z, self.R) + self.b_r     
            self.p_xi_h = tf.squeeze(tf.nn.softmax(self.e))


    def build_model(self):
        with tf.name_scope('loss'):
            self.recon_loss = -tf.reduce_sum(tf.log(tf.gather(self.p_xi_h, self.x_id)))
            self.KL = -0.5 * tf.reduce_sum(1.0 + self.hlogvar-tf.pow(self.hmean, 2) \
                      - tf.exp(self.hlogvar), reduction_indices=1)
            self.loss = tf.reduce_mean(.001 * self.KL + self.recon_loss) + self.l2_loss_all

        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate,beta1=0.8)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in self.grads_and_vars]
        self.train_op = self.optimizer.apply_gradients(self.capped_gvs, global_step=self.global_step)



with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        display_para()
        data_train, label_train, data_val, label_val = load_data()
        nvdm  = nvdm()

        # Output directory for models and summaries
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
        saver = tf.train.Saver(tf.all_variables(), write_version=tf.train.SaverDef.V2)
        sess.run(tf.initialize_all_variables())
        try:
            newest = max(glob.iglob(checkpoint_dir + '/*model*0'), key=os.path.getctime)
            saver.restore(sess, newest)
            print("load checkpoint: {}".format(newest))
        except:
            pass
        def train_step(x_batch, y_batch, epoch):
            """
            A single training step
            """
            x_batch_id = [ _ for _ in itertools.compress(range(10000), map(lambda x: x>0,x_batch[0]))]
            feed_dict = {nvdm.input_x: x_batch, nvdm.x_id: x_batch_id}
            '''
            h1b = [v for v in tf.all_variables() if v.name == "h1/b:0"][0]
            h1w = [v for v in tf.all_variables() if v.name == "h1/w:0"][0]
            _, step, summaries, loss, kl, rc, p_xi_h, R, hb, hw, e  = sess.run(
                [nvdm.train_op, global_step, loss_summary, nvdm.loss, nvdm.KL, nvdm.recon_loss, nvdm.p_xi_h, nvdm.R, h1b, h1w, nvdm.e], feed_dict)
            '''
            _, step,  loss = sess.run([nvdm.train_op, nvdm.global_step, nvdm.loss], feed_dict)

            time_str = datetime.datetime.now().isoformat()
            if step % FLAGS.train_every == 0:
                print("time: {},  epoch: {}, step: {}, loss: {:g}".format(time_str,epoch, step, loss))
            if np.isnan(loss):
                import pdb
                pdb.set_trace()
            #train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch):
            """
            Evaluates model on a dev set
            """
            feed_dict = {nvdm.input_x: x_batch}

            step, loss= sess.run(
                [nvdm.global_step, nvdm.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("time: {}, step: {}, loss {:g},".format(time_str, step, loss))

        
        def prediction(x_sample, y_sample): # sample has size 20
            '''
            Get the perplexity of the test set
            '''

            perplist = []
            for i in range(20):
                x_batch_id = [ _ for _ in itertools.compress(range(10000), map(lambda x: x>0,x_sample[0]))]
                feed_dict = {nvdm.input_x: x_sample[i].reshape(1,10000)}
                step, p_xi_h = sess.run([nvdm.global_step, nvdm.p_xi_h], feed_dict)

                valid_p = np.mean(np.log(p_xi_h[x_batch_id]))
                perplist.append(valid_p)
            print("perplexity: {}".format(np.exp(-np.mean(perplist))))


        
        current_step = 1
        for epo in range(FLAGS.num_epochs):
            # Training loop. For each doc...
            for doc, y in zip(data_train, label_train):
                doc = doc.reshape(1, 10000)
                train_step(doc, y, 1 + epo)
                current_step = tf.train.global_step(sess, nvdm.global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    prediction(data_val[:20], label_val)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))




        
    
