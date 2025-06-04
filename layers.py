import tensorflow as tf

class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim,adj,activation=tf.nn.relu, dropout=0.0):
        super(GraphConvolution, self).__init__()
        self.weight = self.add_weight(shape=(input_dim, output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True,
                                      name="aa")
        self.activation = activation
        self.dropout = dropout
        self.adj=adj


    def call(self, inputs,training=False):
        x = inputs# 拿到 SparseTensor
        # adj = inputs['adj']  # 可能用于 matmul，或在模型外部处理
        # x = tf.sparse.to_dense(x)  # 将稀疏矩阵转换为稠密矩阵
        x = tf.nn.dropout(x, rate=self.dropout) if training else x
        x = tf.matmul(x, self.weight)
        x = tf.sparse.sparse_dense_matmul(self.adj, x)
        return self.activation(x)
class GraphConvolutionSparse(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, activation=tf.nn.relu, dropout=0.0):
        super(GraphConvolutionSparse, self).__init__()
        self.weight = self.add_weight(shape=(input_dim, output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True,name='gc_weight')
        self.activation = tf.nn.relu
        self.dropout = dropout


    def call(self, inputs, training=False):
        x = inputs['features']  # 拿到 SparseTensor
        adj = inputs['adj']  # 可能用于 matmul，或在模型外部处理


        if training and isinstance(x, tf.SparseTensor):
            num_values = tf.shape(x.values)[0]
            dropout_mask = tf.random.uniform([num_values]) < (1 - self.dropout)
            x = tf.sparse.retain(x, dropout_mask)
        if isinstance(x, tf.SparseTensor):
            x_dense = tf.sparse.to_dense(x)
        else:
            x_dense = x

        x = tf.sparse.sparse_dense_matmul(adj, x_dense)
        x = tf.matmul(x, self.weight)

        # 输出的样式是什么？
        return self.activation(x)


class BilinearDecoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_r, activation=tf.nn.sigmoid):
        super(BilinearDecoder, self).__init__()
        self.weight = self.add_weight(shape=(input_dim, input_dim),
                                      initializer='glorot_uniform',
                                      trainable=True,name="bb")
        self.activation = activation
        self.num_r = num_r

    def call(self, embeddings):
        R = embeddings[:self.num_r, :]
        D = embeddings[self.num_r:, :]
        x = tf.matmul(R, self.weight)
        x = tf.matmul(x, tf.transpose(D))
        # 展平成一维
        logits=tf.reshape(x,[-1])
        return logits



