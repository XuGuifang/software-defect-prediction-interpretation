import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Permute, Conv2D, MaxPooling2D,
                                     Dropout, Flatten, Dense)
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
# from keras import regularizers
# from keras.regularizers import l2

from tqdm import tqdm 

temperature = 100 

def train(f, g, sess, x_train, x_train_flat, y_train, optimizer, epochs, batch_size):
    num_samples = len(x_train) 
    steps_per_epoch = np.ceil(num_samples / batch_size).astype(int) 
    print(f"Train on {num_samples} samples") 

    # 定义损失函数
    categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy()

    for epoch in range(epochs): 

        # 根据轮次动态调整损失函数的系数
        lmbd_acc = 1.0 
        # lmbd_hard = 1.0 
        # lmbd_fed = 1.0 
        lmbd_hard = 0.8  if epoch >= 2 else 0 
        lmbd_fed = 0.5 if epoch >= 2 else 0


        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{epochs}", unit="step") as pbar:
            for step in range(steps_per_epoch):
                start_idx = step * batch_size
                end_idx = start_idx + batch_size
                data1 = x_train[start_idx:end_idx]
                data2 = x_train_flat[start_idx:end_idx]
                target = y_train[start_idx:end_idx]

                with tf.GradientTape() as tape:
                    f_pred = f.model(data1, training=True)
                    g_pred = g.model(data2, training=True)

                    loss_acc = categorical_crossentropy(target, f_pred)  # 预测器损失函数
                    loss_hard = L_hard(target, g_pred, f_pred)  # 硬损失函数
                    loss_fed = L_fed(f, g_pred, data2, temperature) # 保真度损失函数

                    loss = lmbd_acc * loss_acc + lmbd_hard * loss_hard + lmbd_fed * loss_fed

                gradients = tape.gradient(loss, f.model.trainable_variables + g.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, f.model.trainable_variables + g.model.trainable_variables))

                pbar.update(1)
                
    return f, g

def train2(f, g, sess, x_train, x_train_flat, y_train, optimizer, epochs, batch_size):
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train_flat, y_train))
    dataset = dataset.shuffle(buffer_size=len(x_train)).batch(batch_size)
    # 创建迭代器
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    next_batch = iterator.get_next()

    # 定义损失函数
    categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy()


    for epoch in range(epochs):
        # 初始化迭代器
        sess.run(iterator.initializer)
        
        # 根据轮次动态调整损失函数的系数
        lmbd_acc = 1.0 
        lmbd_hard = 0.8 if epoch >= 2 else 0
        lmbd_fed = 0.8 if epoch >= 2 else 0
        with tqdm(total=len(x_train), desc=f"Epoch {epoch + 1}/{epochs}", unit="step") as pbar:
            while True:
                try:
                    # 获取下一个批次的数据
                    data1, data2, target = sess.run(next_batch)

                    with tf.GradientTape() as tape:
                        f_pred = f.model(data1, training=True)
                        g_pred = g.model(data2, training=True)

                        loss_acc = categorical_crossentropy(target, f_pred)
                        loss_hard = L_hard(target, g_pred, f_pred)
                        loss_fed = L_fed(f, g_pred, data2, temperature)

                        loss = lmbd_acc * loss_acc + lmbd_hard * loss_hard + lmbd_fed * loss_fed

                    gradients = tape.gradient(loss, f.model.trainable_variables + g.model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, f.model.trainable_variables + g.model.trainable_variables))

                    pbar.update(len(data1))

                except tf.errors.OutOfRangeError:
                    # 数据集结束
                    break

    return f, g


def L_hard(c, q_i, y_i):
    
    loss_interpreter = tf.keras.losses.categorical_crossentropy(c, q_i, from_logits=False)
    loss_predictor = tf.keras.losses.categorical_crossentropy(c, y_i, from_logits=False)
    
    L_hard = tf.reduce_mean(loss_interpreter + loss_predictor)
    
    return L_hard

# 使用均方误差（MSE）计算重构损失
def rec_loss(input, reconstructed_output):
    return tf.reduce_mean(tf.square(input - reconstructed_output))

def L_fed(f, q_i_T, input_data, T):
    # 获取预测器的 logits
    v_i_logits = f.build_model()

    # 调整预测器的 logits
    v_i_logits_scaled = v_i_logits / T
    # 根据温度调整后的 logits 计算概率分布
    p_i_T = tf.nn.softmax(v_i_logits_scaled)

    # 计算损失
    loss_fed = -tf.reduce_sum(p_i_T * tf.math.log(q_i_T))

    return loss_fed

def f_model_evaluate(f_model, x, y):
        if f_model.model:
            score = f_model.model.evaluate(x, y)
            print('accuracy: {:.2f}% | loss: {}'.format(100*score[1], score[0]))
        else:
            print('Missing model instance.')

def g_model_evaluate(g_model, x, y, batch_size):
        if g_model.model:
            score = g_model.model.evaluate(x, y, batch_size)
            print('accuracy: {:.2f}% | loss: {}'.format(100*score[1], score[0]))
        else:
            print('Missing initialized model instance.')



def analyze(f, g, x_test, x_test_flat, y_test):
    # 初始化混淆矩阵
    conf_matx_fy = np.zeros([2, 2])  
    conf_matx_fg = np.zeros([2, 2])  
    conf_matx_gy = np.zeros([2, 2])  

    # 对整个数据集进行预测
    output_f = f.predict(x_test)
    output_g = g.predict(x_test_flat)
    y = np.argmax(y_test, axis=1)
    pred_f = np.argmax(output_f, axis=1)
    pred_g = np.argmax(output_g, axis=1)

    # 更新混淆矩阵
    for i in range(len(y)):
        conf_matx_fy[y[i], pred_f[i]] += 1
        conf_matx_fg[pred_f[i], pred_g[i]] += 1
        conf_matx_gy[y[i], pred_g[i]] += 1

    # 计算准确度和保真度
    accuracy_f = np.diag(conf_matx_fy).sum() / len(y)
    fidelity = np.diag(conf_matx_fg).sum() / len(y)  # 保真度计算
    accuracy_g = np.diag(conf_matx_gy).sum() / len(y)  # g模型的准确度计算

    return accuracy_f, fidelity, accuracy_g

