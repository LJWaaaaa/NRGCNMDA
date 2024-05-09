import tensorflow as tf
from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
from utils import *

class ResidualBlock2(tf.keras.layers.Layer):
    def __init__(self, emb_dim,adj,dropout,act):
        super(ResidualBlock2, self).__init__()
        self.emb_dim = emb_dim
        self.adj=adj
        self.dropout =dropout
        self.act = act

        self.conv1 = GraphConvolution(
            name='gcn_residual_layer1',
            input_dim=emb_dim,
            output_dim=emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)

        self.conv2 = GraphConvolution(
            name='gcn_residual_layer2',
            input_dim=emb_dim,
            output_dim=emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)

        # 添加归一化操作
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = inputs
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)  # 在第一个图卷积层后添加归一化

        x = self.conv2(x)
        x = self.batch_norm2(x)  # 在第二个图卷积层后添加归一化

        x +=residual   # 添加残差连接
        return x

class GCNModel():
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
    def build(self):
        self.adj = dropout_sparse(self.adj, 1-self.adjdp, self.adj_nonzero)
        # 第一个GraphConvolution层
        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)

        self.residual_block2 = ResidualBlock2(emb_dim=self.emb_dim,adj=self.adj,dropout=self.dropout,act=self.act)
        self.residual2 = self.residual_block2(self.hidden1)



        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim, num_r=self.num_r, act=tf.nn.sigmoid)(self.residual3)