'''
This script implement NVDM
'''
import tensorflow as tf

Relu = tf.nn.relu

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



class nvdm():
    def __init__(self):
        '''hyperparameters'''
        self.epochs = 10
        self.batch_size = 128
        self.learning_rate = 0.001
        self.hidden_size = 500
        self.vocab_size = 10000
        self.emb_size = 50 # or 200 # latent dim
        self.R = tf.Varibale(tf.truncated_normal([self.hidden, self.vocab_size],\
                 stddev=0.002, name="r"))
        self.b_r = tf.Variable(tf.truncated_normal([self.vocab_size], \
                 stddev=0.002, name="br"))
        self.model()
        
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()

    def model(self):
        '''encode'''
        self.x = tf.placeholder(tf.float32, [None, self.vocab_size]) # 1*10000, 1doc
        h1 = Relu(Linear(self.x, [self.hidden_size], name="h1"))
        h2 = Relu(Linear(h1, [self.hidden_size], name="h2")) #1*hidden_Size
        self.hmean = Linear(self.h2, [self.emb_size], name="hmean") 
        self.hlogvar = Linear(self.h2, [self.emb_size], name="hlogvar")
    
        '''decode'''
        self.x_i = tf.placeholder(tf.float32, [self.vocab_size, None]) #10000*N

        e = tf.matmul(tf.matmul(-h2, self.R), self.x_i) + self.b_r #e:vocab_size*N
        p_xi_h = tf.nn.logsoftmax(e, dim=0) # vocab_size*N
        self.recon_loss = tf.reduce_sum(p_xi_h)
        self.KL = -0.5 * tf.reduce_sum(1.0 +  self.hlogvar - tf.pow(self.hmean, 2) \
                  - tf.exp(self.hlogvar), reduction_indices = 1) 
        self.loss = tf.reduce_mean(self.KL + self.reconstruct_loss)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1=0.8).minimize(self.loss)

    def train(self):
        

        
    
