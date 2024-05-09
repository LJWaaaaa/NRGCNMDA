import argparse

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import gc
import random

import torch
from matplotlib import pyplot
from numpy import interp
from sklearn.metrics import roc_curve, auc

from clac_metric import cv_model_evaluate
from utils import *
from model_end import GCNModel
from opt import Optimizer
from sklearn.decomposition import PCA

def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)
    return feat_norm

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
    features_nonzero = features[1].shape[0]#稀疏特征矩阵的非零元素的数量
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
    print(model)
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
    #首先使用where函数找出 drug_dis_matrix 中的值为 1 的位置，得到一个形状为 (2, n) 的矩阵 index_matrix，其中第一行表示所有值为1的元素在
    # drug_dis_matrix 中的行索引，第二行表示这些元素的列索引然后使用shape属性中的第1个元素对列数进行计数，得到 association_nam 变量，即 drug_dis_matrix
    # 中值为 1 的元素的个数，由于 index_matrix.shape[1] 即为 association_nam，因此 index_matrix 的列数即表示 drug_dis_matrix 中值为 1 的元素的个数。
    # print(index_matrix)
    # print(association_nam)
    # [[0   1   1... 625 625 626]
    #  [124 101 138...  18  21 116]]
    # 1152
    random_index = index_matrix.T.tolist()#random_index 是将 index_matrix 进行转置操作后，将其转换为Python列表的结果。这意味着 random_index 将包含
    # 原始矩阵 index_matrix 中的数据，但其行和列的顺序将发生改变。原本是行的数据现在变成了列，原本是列的数据现在变成了行。
    # print(random_index)
    #[[0, 124], [1, 101], [1, 138], [2, 101], [3, 101], [4, 4], [4, 124], [4, 128], [5, 57], [5, 101],......
    random.seed(seed)
    random.shuffle(random_index)
    all_predict_y_proba = []
    k_folds = 5

    CV_size = int(association_nam / k_folds)#CV_size 是一个变量，它用于计算每个交叉验证折叠（fold）的大小。通常情况下，交叉验证
    # 的目标是将数据均匀地分成 k 个部分，所以 CV_size 就是将总的数据点数量 association_nam 除以折数 k_folds 得到的结果。这个值表示每个验证折叠中的数据点数量。
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    # print(temp)

    temp[k_folds - 1] = temp[k_folds - 1] + random_index[association_nam - association_nam % k_folds:]
    # print(temp[k_folds - 1])
    random_index = temp
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating drug-disease...." % (seed))

    # Initialize variables for ROC curve
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(drug_dis_matrix, copy=True)
        # print(train_matrix.shape)
        # print(drug_dis_matrix)
        # print(train_matrix)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        # print(train_matrix)
        drug_len = drug_dis_matrix.shape[0]
        dis_len = drug_dis_matrix.shape[1]
        drug_disease_res = PredictScore(
            train_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp)
        predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)# 将模型预测的得分重新整形为一个矩阵，
        # 使其维度与原始 drug_dis_matrix 相匹配。这将创建一个矩阵，其中每个元素表示药物和疾病之间的预测得分。
        metric_tmp = cv_model_evaluate(
            drug_dis_matrix, predict_y_proba, train_matrix)

        # Compute ROC curve and AUC
        fpr, tpr, auc_thresholds = roc_curve(drug_dis_matrix.flatten(), predict_y_proba.flatten())

        auc_score=auc(fpr, tpr)
        pyplot.plot(fpr, tpr, label='ROC fold %d (AUC = %0.4f)' % (k+1,auc_score),linewidth=0.3)

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        pyplot.xlabel('False positive rate, (1-Specificity)')
        pyplot.ylabel('True positive rate,(Sensitivity)')
        pyplot.title('Receiver Operating Characteristic curve: 5-Fold CV')
        pyplot.legend()

        print("------这是第%d次交叉验证结果..------\nAUPR, AUC, f1_score, accuracy, recall, specificity, precision" %(k+1))
        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()

    mean_tpr /= k_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    pyplot.plot(mean_fpr, mean_tpr, '--', linewidth=0.5, label='Mean ROC (AUC = %0.2f)' % mean_auc)

    # Plot random guessing line
    pyplot.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    pyplot.legend()
    pyplot.show()
    metric = np.array(metric / k_folds)
    return metric


if __name__ == "__main__":
    drug_sim = np.loadtxt('../MIC-DRUG/smiles gip.csv', delimiter=',', dtype="float32")
    dis_sim = np.loadtxt('../MIC-DRUG/microbe gip cosine.csv', delimiter=',', dtype="float32")
    drug_dis_matrix = np.loadtxt('../MIC-DRUG/interaction.csv', delimiter=',', dtype="int")

    print(drug_sim.shape)
    print(dis_sim.shape)
    print(drug_dis_matrix.shape)

    # attributes_list = []
    # similarity = np.vstack((np.hstack((drug_sim, np.zeros(shape=(drug_sim.shape[0], dis_sim.shape[1]), dtype=int))),
    #                         np.hstack((np.zeros(shape=(dis_sim.shape[0], drug_sim.shape[0]), dtype=int), dis_sim))))
    # print(similarity)
    # attributes_list.append(similarity)
    # # print(attributes_list)
    # features = np.hstack(attributes_list)
    # features = normalize_features(features)
    # features = sp.csr_matrix(features)
    # print(features)
    # Features = torch.tensor(features.toarray(), dtype=torch.float32)
    # print(Features)


    epoch = 4000
    emb_dim = 64
    lr = 0.01
    adjdp = 0.6
    dp = 0.4  #dp: Dropout概率，用于控制模型过拟合程度，防止过度拟合；
    simw = 6
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    for i in range(circle_time):
        result += cross_validation_experiment(
            drug_dis_matrix, drug_sim*simw, dis_sim*simw, i, epoch, emb_dim, dp, lr, adjdp)
    average_result = result / circle_time
    print("aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision")
    print(result)
