import tensorflow as tf
from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, HGCN1,Product
from utils import *
import numpy as np

def normalized(x):
    node_degrees = np.sum(x, axis=1)

    D_minus_half = np.diag(1.0 / np.sqrt(node_degrees))

    normalized = np.dot(np.dot(np.dot(D_minus_half, x), x.T), D_minus_half)

    return normalized.astype(np.float32)

class GCNModel():

    def __init__(self, placeholders, num_features, emb_dim, features_nonzero, adj_nonzero, nhete_hyper_matrix,mir_mkl, dis_matrix, num_r, name, act=tf.nn.elu):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.hgnn = nhete_hyper_matrix
        self.mir = tf.cast(mir_mkl, tf.float32)
        self.dis = tf.cast(dis_matrix, tf.float32)
        self.act = act
        self.att = tf.Variable(tf.constant([0.5, 0.3, 0.2]))
        self.num_r = num_r
        with tf.variable_scope(self.name):
            self.build()

        
        
    def build(self):
        print(self.adj_nonzero)
        self.adj = dropout_sparse(self.adj, 1-self.adjdp, self.adj_nonzero)# dropout 操作
        print(self.adj)
        
        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)
        
        self.hidden2 = GraphConvolution(
            name='gcn_dense_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden1)

        
        self.emb = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden2)
        

        self.embedding_md = self.hidden1*self.att[0]+self.hidden2*self.att[1]+self.emb*self.att[2]
        

        self.reconstructions,wl = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim, num_r=self.num_r, act=lambda x: x)(self.embedding_md)



        self.hgcn1 = HGCN1(input_dim=self.input_dim, output_dim=self.emb_dim, activation=tf.nn.relu)

        input2 = self.hgnn
        normalized_inputs = normalized(input2)
        self.l_loss = self.hgcn1(normalized_inputs, self.mir, self.dis,wl)

