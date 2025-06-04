import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from scipy.sparse import isspmatrix_coo, coo_matrix, csr_matrix
from sklearn.model_selection import KFold
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.python.training.training_util import global_step

from clr import cyclic_learning_rate
from model import HyperGCNBranch, FusionModel
from model import GCNModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

def train_step(model, train_drug_dis_matrix,inputs_data, adj, labels, optimizer, pos_weight, norm):
    with tf.GradientTape() as tape:
        preds = model(inputs_data, training=True)
        preds = tf.reshape(preds, tf.shape(train_drug_dis_matrix))  # 确保 preds 与 labels 形状一致
        loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            logits=preds, labels=labels, pos_weight=pos_weight))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def build_hyper_adj(H_matrix):
    H = tf.constant(H_matrix, dtype=tf.float32)  # [num_herbs, num_chems]
    # 计算超边权重矩阵 W（对角线）
    d = tf.reduce_sum(H, axis=0)  # 每个成分连接了多少草药 [num_chems]
    d_inv = tf.where(d != 0, 1.0 / d, tf.zeros_like(d))  # 避免除零
    W = tf.linalg.diag(d_inv)  # [num_chems, num_chems]
    # 构造 A_hyper = H * W * H^T
    A_hyper = tf.matmul(H, tf.matmul(W, tf.transpose(H)))  # [num_herbs, num_herbs]
    return A_hyper
def constructNet(drug_dis_matrix):
    drug_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj

def constructHNet(drug_dis_matrix, drug_matrix, dis_matrix):
    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))

def cosine_similarity(z1, z2):
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)
    return tf.matmul(z1, tf.transpose(z2))  # [N, N]

def contrastive_loss(z1, z2, temperature=0.5):
    sim_matrix = cosine_similarity(z1, z2) / temperature  # [N, N]
    batch_size = tf.shape(sim_matrix)[0]
    labels = tf.range(batch_size)

    # 两个方向的交叉熵损失（对称 InfoNCE）
    loss1 = tf.keras.losses.sparse_categorical_crossentropy(labels, sim_matrix, from_logits=True)
    loss2 = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(sim_matrix), from_logits=True)

    return tf.reduce_mean(loss1 + loss2) / 2.0

def pred_metrics(y_true, y_pred):
    y_pred_class = tf.round(tf.nn.sigmoid(y_pred))
    # 计算准确率
    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(y_true, y_pred_class)
    accuracy_value = accuracy.result().numpy()

    # 计算精确率
    precision = tf.keras.metrics.Precision()
    precision.update_state(y_true, y_pred_class)
    precision_value = precision.result().numpy()

    # 计算召回率
    recall = tf.keras.metrics.Recall()
    recall.update_state(y_true, y_pred_class)
    recall_value = recall.result().numpy()

    # 计算 F1 分数
    f1 = 2 * precision_value * recall_value / (precision_value + recall_value) if (precision_value + recall_value) > 0 else 0

    # 计算 AUC
    auc = tf.keras.metrics.AUC()
    auc.update_state(y_true, y_pred)
    auc_value = auc.result().numpy()

    # 计算对数损失
    log_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred).numpy()


    return accuracy_value,precision_value,recall_value,f1,auc_value,log_loss

def pad_matrix(X, target_shape):
    padded = np.zeros(target_shape, dtype=np.float32)
    padded[:X.shape[0], :X.shape[1]] = X
    return padded
def pad_matrix_bottom(X, target_rows):
    """
    只在下方填充零，补到 target_rows 行，列数不变。
    """
    assert target_rows >= X.shape[0], "目标行数不能小于原始行数"
    pad_rows = target_rows - X.shape[0]
    padding = np.zeros((pad_rows, X.shape[1]), dtype=X.dtype)
    return np.vstack([X, padding])
def PredictScore(train_drug_dis_matrix, drug_matrix, dis_matrix, herb_chemical,seed, epochs, emb_dim, dp, lr, adjdp):
    # 初始化设置
    tf.random.set_seed(seed)
    tdm32 = tf.cast(train_drug_dis_matrix, tf.float32)
    hc32=tf.cast(herb_chemical, tf.float32)
    # 构造网络结构
    adj = constructHNet(train_drug_dis_matrix, drug_matrix, dis_matrix)
    adj = tf.cast(adj, tf.float32)  # 先把稠密矩阵转换成 float32
    adj = tf.sparse.from_dense(adj)  # 再转为 SparseTensor
    X = constructNet(train_drug_dis_matrix)
    X = csr_matrix(X)
    features = sparse_to_tuple(X)

    A_hyper_adj=build_hyper_adj(herb_chemical)# [num_herbs, num_herbs]
    A_hyper_adj = pad_matrix(A_hyper_adj, (2033, 2033))# 填充为维度为2033的矩阵
    A_hyper_adj=tf.sparse.from_dense(tf.cast(A_hyper_adj,tf.float32))#转为稀疏矩阵
    X1=pad_matrix_bottom(herb_chemical,2033) #填充为行为2033的矩阵
    X1=csr_matrix(X1)# [num_herbs, num_chemical]
    features_hyper=sparse_to_tuple(X1)
    # 构建模型输入
    features_tensor = tf.sparse.SparseTensor(
        indices=features[0],
        values=tf.cast(features[1],tf.float32),
        dense_shape=features[2]
    )
    features_tensor_hyper=tf.sparse.SparseTensor(
        indices=features_hyper[0],
        values=tf.cast(features_hyper[1], tf.float32),
        dense_shape=features_hyper[2]
    )
    inputs_hyper = {
        'features': features_tensor_hyper,
        'adj': A_hyper_adj
    }
    inputs_data={
        'features': features_tensor,
        'adj': adj
    }
    inputs={
        'inputs_hyper':inputs_hyper,
        'inputs_data':inputs_data
    }
    # 使用Keras构建模型
    # model = GCNModel(input_dim=features[2][1], emb_dim=emb_dim, adj=adj, num_r=train_drug_dis_matrix.shape[0])
    # model_hyper=HyperGCNBranch(input_dim=features_hyper[2][1], emb_dim=emb_dim, hyper_adj=A_hyper_adj)
    model_fusion=FusionModel(input_dim1=features[2][1],input_dim2=features_hyper[2][1],emb_dim=emb_dim,adj=adj,hyper_adj=A_hyper_adj, num_r=train_drug_dis_matrix.shape[0])
    # 计算 pos_weight（正负样本不均衡时）
    num_positives = tf.reduce_sum(tf.cast(train_drug_dis_matrix,tf.int32))
    num_total = tf.size(tf.cast(train_drug_dis_matrix,tf.int32))
    pos_weight = tf.cast((num_total - num_positives) / num_positives, tf.float32)
    print("Positive weight:", pos_weight.numpy())

    # 归一化因子，确保损失值不因稀疏而过小
    num_total=tf.cast(num_total,tf.float32)
    norm = tf.cast(num_total, tf.float32) / (2.0 * tf.cast(num_positives, tf.float32))
    global_step = tf.Variable(0, trainable=False)
    # 定义优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=cyclic_learning_rate(learning_rate=lr*0.1,max_lr=lr,global_step=global_step,gamma=.995,mode='exp_range'))
    # 初始化准确率指标
    accuracy = tf.keras.metrics.Accuracy()


    # 训练循环Hyper
    loss_curve = []
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            preds = model_fusion(inputs, training=True)
            preds = tf.reshape(preds, tf.shape(train_drug_dis_matrix))  # 确保 preds 与 labels 形状一致
            bce_loss = tf.nn.weighted_cross_entropy_with_logits(
                logits=preds, labels=tdm32, pos_weight=pos_weight)
            loss_value = tf.reduce_mean(bce_loss)
            emb_gcn=model_fusion.emb_gcn
            emb_hyper=model_fusion.emb_hyper
            # 计算模态对比损失
            loss_contrast = contrastive_loss(emb_gcn, emb_hyper)
            # 总损失(超参)
            lambda_contrast = 0.01
            loss_total = loss_value + lambda_contrast * loss_contrast
            # 计算准确率
        preds_class = tf.round(tf.nn.sigmoid(preds))  # 将 logits 转换为 0 或 1
        metrics=pred_metrics(tdm32,preds)
        accuracy.update_state(tf.cast(tdm32, tf.int32), tf.cast(preds_class, tf.int32))
        current_accuracy = accuracy.result().numpy()
        # 重置准确率指标
        accuracy.reset_states()
        grads = tape.gradient(loss_total, model_fusion.trainable_variables)
        # grads = tape.gradient(loss_value, model_fusion.trainable_variables)

        # 梯度裁剪
        grads, _ = tf.clip_by_global_norm(grads, 6.0)
        grads_vars = [(g, v) for g, v in zip(grads, model_fusion.trainable_variables) if g is not None]
        optimizer.apply_gradients(grads_vars)
        loss_curve.append(loss_total.numpy())
        # loss_curve.append(loss_value.numpy())
        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss_total.numpy():.5f} | Accuracy: {current_accuracy:.5f}")
            print(f"Accuracy: {metrics[0]:.4f}|Precision: {metrics[1]:.4f}|Recall: {metrics[2]:.4f}|F1 Score: {metrics[3]:.4f}|AUC: {metrics[4]:.4f}|Log Loss: {metrics[5]:.4f}")
    drug_disease_res = model_fusion(inputs, training=False)
    drug_disease_res = tf.sigmoid(drug_disease_res).numpy()  # 转为概率预测结果

    return drug_disease_res, loss_curve


def cross_validateion(drug_dis_matrix, drug_matrix, dis_matrix,herb_chemical, seed, epochs, emb_dim, dp, lr, adjdp):
    # 1. 获取所有正样本索引（二维坐标）
    positive_pairs = np.argwhere(drug_dis_matrix == 1)  # shape=(N, 2)
    # 2. 初始化交叉验证器
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    # 3. 初始化评估指标与曲线记录
    metric = np.zeros((1, 8))
    all_loss_curves = []

    print("seed=%d, evaluating drug-disease...." % seed)
    # 4. 遍历每一折
    for k, (train_index, test_index) in enumerate(kf.split(positive_pairs)):
        print("------this is %dth cross validation------" % (k + 1))
        # 提取验证集中正样本索引
        val_pos = positive_pairs[test_index]

        # 构造训练用矩阵（将 val_pos 对应的值置零）
        train_matrix = np.array(drug_dis_matrix, copy=True)
        train_matrix[val_pos[:, 0], val_pos[:, 1]] = 0

        # drug_len = drug_dis_matrix.shape[0]
        # dis_len = drug_dis_matrix.shape[1]
        drug_len = train_matrix.shape[0]
        dis_len = train_matrix.shape[1]
        # 5. 模型训练 & 获取预测结果
        drug_disease_res, loss_curve = PredictScore(
            train_matrix, drug_matrix, dis_matrix,herb_chemical,
            seed=seed,
            epochs=epochs,
            emb_dim=emb_dim,
            dp=dp,
            lr=lr,
            adjdp=adjdp
        )
        tdm32 = tf.cast(train_matrix, tf.float32)
        num_positives = tf.reduce_sum(tf.cast(drug_dis_matrix, tf.int32))
        num_total = tf.size(tf.cast(drug_dis_matrix, tf.int32))
        pos_weight = tf.cast((num_total - num_positives) / num_positives, tf.float32)
        # 归一化因子，确保损失值不因稀疏而过小
        num_total = tf.cast(num_total, tf.float32)
        predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)

        # 计算测试集的交叉熵损失
        test_loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=predict_y_proba, labels=tdm32,pos_weight=pos_weight))
        # 6. 保存 loss 曲线
        all_loss_curves.append(loss_curve)

    plt.figure(figsize=(10, 6))
    for i, curve in enumerate(all_loss_curves):
        plt.plot(curve, label=f'Fold {i + 1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curves Across Folds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 可选：你可以在这里计算当前折的 AUC、AUPR 等，并更新 `metric`


def sparse_to_tuple(sparse_mx):
    if not isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
        # 提取行和列索引
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    # 提取非零值
    values = sparse_mx.data
    # 提取矩阵形状
    shape = sparse_mx.shape
    return coords, values, shape
if __name__=="__main__":
    epoch = 5000
    emb_dim = 128
    lr = 0.01
    adjdp = 0.7
    dp = 0.5
    simw = 6
    seed=42
    drug_sim = pd.read_excel('../data/final_final_simherb.xlsx', engine='openpyxl', header=None).values
    dis_sim = pd.read_excel('../data/final_final_simdisease.xlsx', engine='openpyxl', header=None).values
    drug_dis_matrix = pd.read_excel('../data/final_final_dis_herb.xlsx', engine='openpyxl', header=None).values
    herb_chemical=pd.read_excel('../data/final_final_chemical_herb.xlsx', engine='openpyxl', header=None).values
    # 转置 drug_dis_matrix(自己数据集需要转置)转置草药化学
    drug_dis_matrix = drug_dis_matrix.T
    herb_chemical=herb_chemical.T
    all_loss_curves, performance = cross_validateion(
        drug_dis_matrix, drug_sim * simw, dis_sim * simw, herb_chemical,seed, epoch, emb_dim, dp, lr, adjdp)
