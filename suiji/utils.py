import numpy as np
import tensorflow as tf
import scipy.sparse as sp


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )

    return tf.Variable(initial, name=name)


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out*(1./keep_prob)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_nomalized = adj_.dot(
        degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_nomalized = adj_nomalized.tocoo()
    

    return sparse_to_tuple(adj_nomalized)

def Product(x):
    miR = x[0:757, :]
    dis = x[757:, :]
    return miR,dis

def preprocess_hypergraph(H, variable_weight=False):
    H = np.array(H)
    n_edge = H.shape[1]

    W = np.ones(n_edge)

    DV = np.sum(H * W, axis=1)

    DE = np.sum(H, axis=0)

    a = np.where(DE == 0, 0, np.power(DE, -1))
    invDE = np.diag(a)

    b = np.where(DV == 0, 0, np.power(DV, -0.5))
    DV2 = np.diag(b)

    W = np.diag(W)
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2.dot(H)
        invDE_HT_DV2 = invDE.dot(HT).dot(DV2)
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2.dot(H).dot(W).dot(invDE).dot(HT).dot(DV2)
    return G


def constructNet(mir_dis_matrix):
    mir_matrix = np.matrix(
        np.zeros((mir_dis_matrix.shape[0], mir_dis_matrix.shape[0]), dtype=np.int8)) #757*757全0矩阵
    dis_matrix = np.matrix(
        np.zeros((mir_dis_matrix.shape[1], mir_dis_matrix.shape[1]), dtype=np.int8)) #435*435全0矩阵
    mat1 = np.hstack((mir_matrix, mir_dis_matrix))
    mat2 = np.hstack((mir_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj


def constructHNet(mir_dis_matrix, mir_matrix, dis_matrix):
    mat1 = np.hstack((mir_matrix, mir_dis_matrix))
    mat2 = np.hstack((mir_dis_matrix.T, dis_matrix))
    
    return np.vstack((mat1, mat2))
