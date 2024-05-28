import tensorflow as tf
from utils import *


class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1-self.dropout)

            x = tf.matmul(x, self.vars['weights'])

            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)

        return outputs


class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)

            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])

            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)

        return outputs
class HGCN1(tf.Module):
    def __init__(self, input_dim, output_dim, activation=tf.nn.elu):
        super(HGCN1, self).__init__()
        
        self.weights = tf.Variable(tf.random.normal([input_dim, input_dim], dtype=tf.float32), name='weights')
        self.weights1 = tf.Variable(tf.random.normal([input_dim, output_dim], dtype=tf.float32), name='weights1')
        self.bias = tf.Variable(tf.zeros([input_dim],  dtype=tf.float32), name='bias')
        self.bias1 = tf.Variable(tf.zeros([output_dim],  dtype=tf.float32), name='bias1')
        
        self.linear_m = tf.Variable(tf.random.normal([757, input_dim], dtype=tf.float32), name='linear_m', trainable=False)
        self.linear_d = tf.Variable(tf.random.normal([435, input_dim], dtype=tf.float32), name='linear_d', trainable=False)
        
        self.weights_m = tf.Variable(tf.random.normal([128, 757], dtype=tf.float32), name='weights_m')
        self.weights_d = tf.Variable(tf.random.normal([128, 435], dtype=tf.float32), name='weights_d')
        
        self.activation = activation
        self.dropout_rate = 0

    def __call__(self,inputs,mir,dis, wl):

        one_m = tf.matmul(mir, self.linear_m)
        one_d = tf.matmul(dis, self.linear_d)
        X_0 = tf.concat([one_m, one_d], axis=0)
        x = tf.matmul(X_0 , self.weights)
        output1 = tf.matmul(inputs, x) + self.bias
        output = tf.matmul(output1, self.weights1) + self.bias1
        if self.activation is not None:
            output = self.activation(output)
            
        output1 = output
        mg = output1[0:757,:]
        dg = output1[757: 757+435, :]

        mg = tf.matmul(mg, self.weights_m)
        dg = tf.matmul(dg, self.weights_d)

        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            dg = dg.eval()
            mg = mg.eval()
            wl = wl.eval()
        
        lap_matx_m, lap_Dm = get_lapl_matrix(mg)
        lap_matx_d, lap_Dd = get_lapl_matrix(dg)
        
        lap_matx_m_1 = tf.constant(lap_matx_m, tf.float32)
        lap_matx_d_1 = tf.constant(lap_matx_d, tf.float32)
        
        weight_emb_m = self.weights1[0:757,:]
        weight_emb_d = self.weights1[757: 757+435, :]
        
        weight_emb_m = tf.matmul(weight_emb_m, wl)
        weight_emb_d = tf.matmul(weight_emb_d, wl)
        
        
        self.m_lap = tf.matmul(tf.matmul(tf.transpose(weight_emb_m), lap_matx_m_1), weight_emb_m)
        self.d_lap = tf.matmul(tf.matmul(tf.transpose(weight_emb_d), lap_matx_d_1), weight_emb_d)
        self.m_loss = tf.trace(self.m_lap)
        self.d_loss = tf.trace(self.d_lap)
        self.l_loss = tf.add(tf.multiply(self.m_loss, tf.constant([0.01], tf.float32)), tf.multiply(self.d_loss, tf.constant([0.01], tf.float32)))

                           
        return self.l_loss     
        
def get_lapl_matrix(sim):
    m,n = sim.shape
    lap_matrix_tep = np.zeros([m,m])
    for i in range(m):
        lap_matrix_tep[i,i] = np.sum(sim[i,:])
    lap_matrix = lap_matrix_tep - sim
    return lap_matrix, lap_matrix_tep

def Product(x):
    miR = x[0:757, :]
    dis = x[757:, :]
    return miR,dis

class InnerProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, name, num_r, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, input_dim, name='weights')

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1-self.dropout)
            print('self.num_r',self.num_r)

            R = inputs[0:self.num_r, :]
            D = inputs[self.num_r:, :]

            R = tf.matmul(R, self.vars['weights'])
            D = tf.transpose(D)
            x = tf.matmul(R, D)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)

        return outputs,self.vars['weights']
