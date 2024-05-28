import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import gc
import os
import random
import time
import h5py
from clac_metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer
from scipy.io import loadmat

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def PredictScore(train_matrix, mir_mkl, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp, hete_hyper_matrix):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_matrix, mir_mkl, dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_matrix.sum()
    X = constructNet(train_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    adj_orig = train_matrix.copy()

    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=()),
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, hete_hyper_matrix,mir_mkl, dis_matrix, train_matrix.shape[0], name='HHMDA')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            l_loss_value = model.l_loss,
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_matrix.shape[0], num_v=train_matrix.shape[1], association_nam=association_nam)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", avg_cost)
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)


    return res

def cos_dis(x):

    x = np.mat(x)
    norm = np.linalg.norm(x, axis=1)
    norm[norm == 0] = 1
    x = np.divide(x, norm.reshape(-1, 1))
    dis_mat = 1 - np.matmul(x, x.T)
    return dis_mat


def cross_validation_experiment(mir_dis_matrix, mir_mkl, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp, hete_hyper_matrix):
    index_matrix = np.mat(np.where(mir_dis_matrix == 1))
    
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()

    random.seed(seed)
    random.shuffle(random_index)

    k_folds = 5

    mir_len = mir_dis_matrix.shape[0]
    dis_len = mir_dis_matrix.shape[1]
    
    metric = np.zeros((1, 5))

    neg = ([], [])
    neg_nam = association_nam
    while len(neg[0]) < neg_nam:
        i, j = random.randint(0, mir_len-1), random.randint(0, dis_len-1)
        if mir_dis_matrix[i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    neg_row = np.array(neg[0])
    neg_col = np.array(neg[1])
    neg_index = np.stack([neg_row, neg_col])
    random_neg_index = neg_index.T.tolist()
    CV_size = int(association_nam / k_folds)
    CV_neg_size = int(neg_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]
    random_index = temp

    temp_neg = np.array(random_neg_index[:neg_nam - neg_nam %
                                 k_folds]).reshape(k_folds, CV_neg_size,  -1).tolist()
    temp_neg[k_folds - 1] = temp_neg[k_folds - 1] + \
        random_neg_index[neg_nam - neg_nam % k_folds:]
    random_neg_index = temp_neg

    print("seed=%d, evaluating mir-disease...." % (seed))

    for k in range(k_folds):

        pos_index = random_index[k]
        neg_index = random_neg_index[k]
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(mir_dis_matrix, copy=True)
        test_matrix = train_matrix[tuple(np.array(random_index[k]).T)]
        train_matrix[tuple(np.array(random_index[k]).T)] = 0

        mir_cos = cosine_similarity(train_matrix)

        mir_cos = np.triu(mir_cos) + np.triu(mir_cos, 1).T

        np.fill_diagonal(mir_cos, 1)

        dis_cos = cosine_similarity(train_matrix.T)

        dis_cos = np.triu(dis_cos) + np.triu(dis_cos, 1).T

        np.fill_diagonal(dis_cos, 1)



        mir_mkl = (mir_cos + mir_mkl)/2

        dis_matrix = (dis_cos + dis_matrix)/2


        mir_disease_res = PredictScore(
            train_matrix, mir_mkl, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp, hete_hyper_matrix)


        predict_y_proba = mir_disease_res.reshape(mir_len, dis_len)
        predict_y = predict_y_proba[tuple(np.array(random_index[k]).T)]


        metric_tmp = cv_model_evaluate(mir_dis_matrix, predict_y_proba, neg_index, pos_index,k)

        print(metric_tmp)

        metric += metric_tmp
        del train_matrix
        del test_matrix
        gc.collect()
    print(metric / k_folds)
    metric = np.array(metric / k_folds)
    
    return metric

def hete_gene_hyper(mir,dis):

    hete_hyper_matrix = np.vstack((mir, dis))
    
    return hete_hyper_matrix

if __name__ == "__main__":

    
    mir_mkl = np.loadtxt("data/mm_delete.txt")
    dis_mkl = np.loadtxt("data/dd_delete.txt")
    mir_dis_matrix = np.loadtxt("data/md_delete.txt")
    mg_mkl = np.loadtxt("data/mg_delete.txt")
    dg_mkl = np.loadtxt("data/dg_delete.txt")

    epoch = 4000

    emb_dim = 128
    
    hete_hyper_matrix = hete_gene_hyper(mg_mkl,dg_mkl)
    
    lr = 0.01
    adjdp = 0.6
    dp = 0.4
    result = np.zeros((1, 5), float)
    average_result = np.zeros((1, 5), float)
    circle_time = 10
    for i in range(circle_time):
        random_integer = random.randint(1, 1000000)
        result += cross_validation_experiment(mir_dis_matrix, mir_mkl, dis_mkl, random_integer, epoch, emb_dim, dp, lr, adjdp, hete_hyper_matrix)

    average_result = result / circle_time
    print(average_result)


