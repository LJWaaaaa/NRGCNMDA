import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from utils import *
import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.nn.functional as F
from torch import nn

#用于创建图卷积层的对象
class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels.无边标记无向图的基本图卷积层"""
    # 类构造方法，用于初始化图卷积层的参数和变量。以下是方法的参数
    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        #input_dim: 输入特征的维度，表示每个节点的特征向量的大小。
        #output_dim: 输出特征的维度，表示每个节点的特征向量的目标大小。
        #adj: 表示图的邻接矩阵，用于描述图中节点之间的连接关系。
        #name: 图卷积层的名称
        #dropout: 用于指定dropout概率，以减少过拟合
        #act: 激活函数，用于在特征传播过程中对节点特征进行非线性变换，默认为ReLU函数。
        self.name = name                                #将图卷积层的名称存储在实例变量 self.name 中。
        self.vars = {}                                  #创建一个字典来存储图卷积层的变量（在这里主要是权重）
        self.dropout = dropout                          #存储 dropout 概率。防止过拟合
        self.adj = adj                                  #存储图的邻接矩阵。
        self.act = act                                  #存储激活函数
        self.issparse = False                           #表示图卷积层的输入是否是稀疏张量，默认为False。
        with tf.variable_scope(self.name + '_vars'):    #创建一个 TensorFlow 变量作用域，用于组织权重变量。
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        #self.vars['weights']：创建了一个权重矩阵，通过 weight_variable_glorot 函数初始化。
        # 这个权重矩阵将用于将输入特征映射到输出特征。

    def __call__(self, inputs):                         #定义了当调用图卷积层对象时要执行的操作
        with tf.name_scope(self.name):
            x = inputs                                  #x = inputs：将输入特征存储在变量 x 中
            x = tf.nn.dropout(x, 1-self.dropout)        #应用 dropout 操作以减少过拟合。
            # 计算注意力分数
            attention_scores = tf.matmul(tf.matmul(x, self.vars['weights']), tf.transpose(x))

            # 计算注意力权重
            attention_weights = tf.nn.softmax(attention_scores)

            # 应用注意力权重到邻居节点的特征
            neighbor_aggregation = tf.matmul(attention_weights, x)

            x = tf.matmul(neighbor_aggregation, self.vars['weights'])  # 使用权重矩阵进行线性变换
            # x = tf.matmul(x, self.vars['weights'])      #将输入特征与权重矩阵相乘，进行线性变换。
            # x = tf.sparse_tensor_dense_matmul(self.adj, x)#通过邻接矩阵与线性变换的结果进行稀疏矩阵乘法.用于在图上传播特征。
            outputs = self.act(x)                       #应用激活函数 self.act，将传播后的特征进行非线性变换，得到最终的输出特征。
        return outputs

class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs.图卷积层用于稀疏输入"""
     #用于处理稀疏输入的图卷积层（Graph Convolution Layer）的实现
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
        self.features_nonzero = features_nonzero  #输入特征中非零元素的数量。
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs

class InnerProductDecoder():
    def __init__(self, input_dim, name, num_r, dropout=0., act=tf.nn.sigmoid, hidden_units=64):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r
        self.hidden_units = hidden_units
        with tf.variable_scope(self.name + '_w'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, input_dim, name='w')
            self.vars['hidden_weights'] = weight_variable_glorot(
                input_dim, self.hidden_units, name='hidden_weights')
            self.vars['output_weights'] = weight_variable_glorot(
                self.hidden_units, 1, name='output_weights')
        self.layer1 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.layer3 = tf.keras.layers.Dense(units=input_dim, activation=tf.nn.relu)
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            R = inputs[0:self.num_r, :]
            D = inputs[self.num_r:, :]
            R = tf.matmul(R, self.vars['weights'])
            D = tf.transpose(D)
            x = tf.matmul(R, D)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
        return outputs

# 这段代码实现了一个基本的图卷积神经网络（GCN）和内积解码器模型，用于无边标签的无向图上的链接预测任务。具体来说：
# GraphConvolution 类实现了一层基本的图卷积层，其输入为节点特征矩阵和邻接矩阵，输出为经过非线性激活函数后的节点特征矩阵。
# GraphConvolutionSparse 类实现了针对稀疏输入的图卷积层，其输入为稀疏节点特征矩阵和邻接矩阵，输出为经过非线性激活函数后的节点特征矩阵。
# InnerProductDecoder 类实现了内积解码器，其输入为节点特征矩阵中的前num r行表示关系矩阵（即边的类型），后面的部分则表示节点特征，输出为链接预测得分。
# 这些类都包括一个 __call__ 方法，L用于在计算图中进行前向传播。同时，它们都包含变量参数（例如权重），可以在训练过程中进行更新。整个模型是由这些层构成的管道，在执行前向传播时将从上一层传递的张量作为输入传递到下一层。
# outputs = self.act(x)  # 使用指定的激活函数进行输出
# outputs_normalized = tf.nn.l2_normalize(outputs, axis=0)  # 归一化处理
# return outputs_normalized  # 返回归一化后的输出

