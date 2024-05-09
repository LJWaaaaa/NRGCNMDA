import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import gc
import random
from clac_metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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
    # 创建一个空列表来存储每一折的预测结果
    all_predict_y_probas = []
    index_matrix = np.mat(np.where(drug_dis_matrix == 1))
    association_nam = index_matrix.shape[1] #药物-疾病关联的总数量
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)#每折中的关联数量
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    #将关联索引按折数分配到临时列表中。
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
        #调用 PredictScore 函数进行模型训练和预测，得到预测概率矩阵 predict_y_proba。
        predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)
        all_predict_y_probas.append(predict_y_proba)
        metric_tmp = cv_model_evaluate(
            drug_dis_matrix, predict_y_proba, train_matrix)
        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / k_folds)
    metric = np.array(metric / k_folds)
    return metric,all_predict_y_probas


if __name__ == "__main__":
    drug_sim = np.loadtxt('../data/drug_sim.csv', delimiter=',')
    dis_sim = np.loadtxt('../data/dis_sim.csv', delimiter=',')
    drug_dis_matrix = np.loadtxt('../data/drug_dis.csv', delimiter=',')
    epoch = 4000
    emb_dim = 64
    lr = 0.01
    adjdp = 0.6
    dp = 0.4  #dp: Dropout概率，用于控制模型过拟合程度，防止过度拟合；
    simw = 6
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)

    circle_time = 5  # 进行五交叉验证
    all_predict_y_probas = []  # 创建一个空列表来存储每一折的预测结果

    for i in range(circle_time):
        metric, predict_y_probas= cross_validation_experiment(
            drug_dis_matrix, drug_sim*simw, dis_sim*simw, i, epoch, emb_dim, dp, lr, adjdp)
        result += metric
        all_predict_y_probas.extend(predict_y_probas)  # 添加每一折的预测结果
    average_result = result / circle_time
    print("accuracy（准确率）  precision（精确度） recall（召回率） F1-score（F1指标） AUPR（平均准确率） AUC（ROC曲线下面积） MCC（Matthews相关系数）")
    print(average_result)
    # 注意：确保已经定义了 average_result，即每一折交叉验证的评价结果

    # 计算平均 ROC 曲线和 AUC
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    # mean_auc = np.mean(average_result[:, 6])  # 平均 AUC

    plt.figure(figsize=(8, 6))

    # 绘制每一折的 ROC 曲线
    for i in range(circle_time):
        fpr, tpr, _ = roc_curve(drug_dis_matrix.flatten(), all_predict_y_probas[i].flatten())
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='Fold %d (AUC = %0.2f)' % (i + 1, auc(fpr, tpr)))

    # 绘制平均 ROC 曲线
    mean_tpr /= circle_time
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # 平均 AUC
    plt.plot(mean_fpr, mean_tpr, color='b', linestyle='--', label='Mean ROC (AUC = %0.2f)' % mean_auc)

    # 设置图形标题、坐标轴标签和图例
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    # Print the mean AUC value
    print("Mean AUC:", mean_auc)

    # 显示图形
    plt.show()
