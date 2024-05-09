# 这段代码导入了 TensorFlow 所需的库和模块，以及从 layers 模块中导入自定义层
# （GraphConvolution、GraphConvolutionSparse 和 ETG）。
# 还从 utils 模块导入了实用函数。
import tensorflow as tf
from layers import GraphConvolution, GraphConvolutionSparse,InnerProductDecoder

from utils import *
# from layers import GraphAttention

class GCNModel():
    # 构造函数代码

    def __init__(self, placeholders, num_features, emb_dim, features_nonzero, adj_nonzero, num_r, name, act=tf.nn.elu):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.act = act
        self.att = tf.Variable(tf.constant([0.5, 0.33, 0.25]))
        self.num_r = num_r
        with tf.variable_scope(self.name):
            self.build()

    #     # 模型构建代码
    def build(self):

        # 在构建阶段对邻接矩阵进行dropout操作
        self.adj = dropout_sparse(self.adj, 1-self.adjdp, self.adj_nonzero)

        # 第一个图卷积层（稀疏特征输入）
        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)
        # 添加密集连接和残差连接
        # self.dense1 = tf.keras.layers.Dense(units=self.emb_dim, activation=self.act)(self.hidden1)
        # self.hidden1 = self.hidden1 + self.dense1
        # 第二个图卷积层（密集特征输入）
        self.hidden2 = GraphConvolution(
            name='gcn_dense_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden1)+self.hidden1

        # 添加密集连接和残差连接
        # self.dense2 = tf.keras.layers.Dense(units=self.emb_dim, activation=self.act)(self.hidden2)
        # self.hidden2 = self.hidden2 + self.dense2

        # 第三个图卷积层
        self.hidden3 = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden2)+ self.hidden2
        self.hidden4 = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden3) + self.hidden3
        self.hidden5 = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden4) + self.hidden4

        self.emb = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden5)+self.hidden5

        # # 添加残差连接
        # self.residual1 = self.hidden1
        # self.residual2 = self.hidden2
        # self.residual3 = self.hidden3
        # self.hidden1 = self.hidden1 + self.residual1
        # self.hidden2 = self.hidden2 + self.residual2
        # self.hidden3 = self.hidden3 + self.residual3

            # 密集残差连接


        # self.embeddings = self.hidden1 * \
        #     self.att[0]+self.hidden2*self.att[1]+self.emb*self.att[2]

        # 使用内积解码器重构图数据
        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim, num_r=self.num_r, act=tf.nn.sigmoid)(self.emb)

#变分自编码