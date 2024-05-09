import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import gc
import random

import torch

from clac_metric import cv_model_evaluate
from utils import *
from model_RN import GCNModel
from opt import Optimizer


def PredictScore(train_drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_drug_dis_matrix, drug_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_drug_dis_matrix.sum()
    X = constructNet(train_drug_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_drug_dis_matrix.shape[0], name='LAGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_drug_dis_matrix.shape[0], num_v=train_drug_dis_matrix.shape[1], association_nam=association_nam)
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
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res
# 这是一个使用 TensorFlow 实现的 GCN 模型，用于预测药物和疾病之间的关联得分。下面是该函数中各参数的含义：
# train_drug_dis_matrix: 训练集中已知的药物-疾病关联矩阵
# drug_matrix: 药物的特征矩阵，每行表示一个药物的特征向量
# dis_matrix: 疾病的特征矩阵，每行表示一个疾病的特征向量
# seed: 随机数种子，用于实验的可重复性
# epochs: 训练轮数
# emb_dim: GCN 模型中节点嵌入向量的维度
# dp: dropout 概率，防止过拟合
# lr: 学习率
# adjdp: 邻接矩阵的 dropout 概率，防止过拟合
# 该函数主要执行以下步骤：
# 根据训练集中的药物-疾病关联矩阵构建一个混合网络邻接矩阵 adj。
# 构建一个药物-疾病关联网络邻接矩阵 adj_orig。
# 对 adj 进行预处理（即规范化），并将其转换为稀疏矩阵形式。
# 将药物和疾病的特征矩阵转换为稀疏矩阵形式。
# 定义 GCN 模型，包括节点嵌入向量、重构矩阵等。
# 定义优化器，计算损失函数，并进行反向传播更新模型参数。
# 在每轮训练结束后输出损失函数值。
# 训练结束后使用训练好的模型对关联得分进行预测，并返回预测结果。

def cross_validation_experiment(drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp):

    # drug_dis_matrix: 药物 - 疾病关联矩阵，其维度为(m, n)，m表示药物数量，n表示疾病数量，元素为0或1，其中1表示药物和疾病之间存在关联；
    # drug_matrix: 药物相似性矩阵，其维度为(m, m)，m表示药物数量，元素表示药物之间的相似度；
    # dis_matrix: 疾病相似性矩阵，其维度为(n, n)，n表示疾病数量，元素表示疾病之间的相似度；
    # seed: 随机数种子，用于控制随机数生成过程的输出，以保证结果的可重复性；
    # epochs: 模型训练的迭代次数；
    # emb_dim: 药物和疾病的向量维度；
    # dp: Dropout概率，用于控制模型过拟合程度，防止过度拟合；
    # lr: 模型学习率；
    # adjdp: 邻域Dropout概率，用于控制邻域节点的变化程度。
    index_matrix = np.mat(np.where(drug_dis_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating drug-disease...." % (seed))
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(drug_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        drug_len = drug_dis_matrix.shape[0]
        dis_len = drug_dis_matrix.shape[1]
        drug_disease_res = PredictScore(
            train_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp)
        predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)
        metric_tmp = cv_model_evaluate(
            drug_dis_matrix, predict_y_proba, train_matrix)
        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / k_folds)
    metric = np.array(metric / k_folds)
    return metric


if __name__ == "__main__":
    drug_sim = np.loadtxt('../data/drug_sim.csv', delimiter=',')
    dis_sim = np.loadtxt('../data/dis_sim.csv', delimiter=',')
    drug_dis_matrix = np.loadtxt('../data/drug_dis.csv', delimiter=',')
    # Features = torch.tensor(Features.toarray(), dtype=torch.float32)
    epoch = 4000
    emb_dim = 64
    lr = 0.01
    adjdp = 0.6  #规则
    dp = 0.4  #dp: Dropout概率，用于控制模型过拟合程度，防止过度拟合；节点
    simw = 6
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    for i in range(circle_time):
        result += cross_validation_experiment(
            drug_dis_matrix, drug_sim*simw, dis_sim*simw, i, epoch, emb_dim, dp, lr, adjdp)
    average_result = result / circle_time
    print("accuracy（准确率）  precision（精确度） recall（召回率） F1-score（F1指标） AUPR（平均准确率） AUC（ROC曲线下面积） MCC（Matthews相关系数）")
    print(average_result)
