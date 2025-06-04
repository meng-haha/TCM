from layers import GraphConvolutionSparse, GraphConvolution, BilinearDecoder
import tensorflow as tf
import os

class GCNModel(tf.keras.Model):
    def __init__(self, input_dim, emb_dim, adj,num_r, act=tf.nn.elu, dropout=0.5):
        super(GCNModel, self).__init__()
        self.gc1 = GraphConvolutionSparse(input_dim, emb_dim, activation=act, dropout=dropout)
        self.gc2 = GraphConvolution(emb_dim, emb_dim,adj=adj,activation=act, dropout=dropout)
        self.gc3 = GraphConvolution(emb_dim, emb_dim,adj=adj,activation=act, dropout=dropout)
        # self.att = tf.Variable(tf.random.normal([3]), trainable=True, dtype=tf.float32)
        self.att = tf.Variable([0.5, 0.33, 0.05], trainable=True, dtype=tf.float32,name="att")
        # self.decoder = BilinearDecoder(emb_dim, num_r)

    def call(self, inputs, training=False):
        hidden1 = self.gc1(inputs,training=training)
        hidden2 = self.gc2(hidden1,training=training)
        hidden3 = self.gc3(hidden2,training=training)

        # Embedding加权融合（多层嵌入融合的可学习加权平均）
        alpha = tf.nn.softmax(self.att)
        # embeddings = alpha[0] * hidden1 + alpha[1] * hidden2 + alpha[2] * hidden3
        embeddings = self.att[0] * hidden1 + self.att[1] * hidden2 + self.att[2] * hidden3
        return embeddings
        # output = self.decoder(embeddings)
        # return output


class FusionModel(tf.keras.Model):
    def __init__(self, input_dim1,input_dim2, emb_dim, adj, hyper_adj, num_r, dropout=0.5):
        super(FusionModel, self).__init__()
        # GCN 分支
        self.gcn_branch = GCNModel(input_dim1, emb_dim, adj, num_r, dropout=dropout)
        # HyperGCN 分支
        self.hyper_branch = HyperGCNBranch(input_dim2, emb_dim, hyper_adj, dropout=dropout)
        # 可学习融合权重（也可以改成拼接）
        self.alpha = self.add_weight(
            name="alpha",
            shape=(2,),
            initializer=tf.constant_initializer([0.5, 0.5]),
            trainable=True,
            dtype=tf.float32
        )

        # 解码器用于关联预测（也可以继续用 BilinearDecoder）
        self.decoder = BilinearDecoder(emb_dim, num_r)

    def call(self, inputs, training=False):
        inputs_data=inputs['inputs_data']
        inputs_hyper=inputs['inputs_hyper']
        # 分支输出
        emb_gcn = self.gcn_branch(inputs_data, training=training)
        emb_hyper = self.hyper_branch(inputs_hyper, training=training)

        # softmax归一化权重
        alpha = tf.nn.softmax(self.alpha)
        emb_fused = alpha[0] * emb_gcn + alpha[1] *  emb_hyper

        self.emb_gcn=emb_gcn
        self.emb_hyper=emb_hyper
        # 解码
        output = self.decoder(emb_fused)
        return output


class HyperGCNBranch(tf.keras.Model):
    def __init__(self, input_dim, emb_dim, hyper_adj, dropout=0.5):
        super(HyperGCNBranch, self).__init__()
        self.gc1 = GraphConvolutionSparse(input_dim, emb_dim, activation=tf.nn.relu, dropout=dropout)
        self.hgcn = GraphConvolution(emb_dim, emb_dim, adj=hyper_adj, activation=tf.nn.relu, dropout=dropout)

    def call(self, inputs, training=False):
        x = self.gc1(inputs, training=training)# 输入是 [num_herbs, dim]
        x = self.hgcn(x, training=training)
        return x
