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


class ConvNet(object):
    def __init__(self, img_rows, img_cols, img_chans, n_classes,
                 optimizer=Adam, loss='categorical_crossentropy',  #加入惩罚项
                 metrics=['acc'], learning_rate=3e-04):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_chans = img_chans
        self.n_classes = n_classes

        self.optimizer = Adam(lr=learning_rate)
        self.loss = loss
        self.metrics = metrics

        self.model = None
        # self.logits_model = None

    def build_model(self):
        input_layer = Input(shape=(self.img_rows, self.img_cols, self.img_chans))

        # handle image dimensions ordering
        if tf.keras.backend.image_data_format() == 'channels_first':
            #如果data_format是“ channels_first”，则具有形状：（批，通道，行，列）的4D张量
            latent = Permute((3, 1, 2))(input_layer)
        else:
            latent = input_layer

        # define the network architecture

        latent = Conv2D(filters=256, kernel_size=1,activation='relu')(latent)
        latent = MaxPooling2D(pool_size=1)(latent)
        latent = Conv2D(filters=128, kernel_size=1,activation='relu')(latent)
        latent = Conv2D(filters=64, kernel_size=1, activation='relu')(latent)
        latent = MaxPooling2D(pool_size=1)(latent)
        latent = Conv2D(filters=32, kernel_size=1, activation='relu')(latent)
        latent = Dropout(rate=0.5)(latent)
        latent = Flatten()(latent)
        latent = Dense(units=8, activation='relu')(latent)
        '''
        latent = Conv2D(filters=256, kernel_size=1, activation='relu')(latent)
        latent = Conv2D(filters=128, kernel_size=1, activation='relu')(latent)
        latent = Conv2D(filters=64, kernel_size=1, activation='relu')(latent)
        #latent = Dropout(rate=0.25)(latent)
        #latent = MaxPooling2D(pool_size=(2, 2))(latent)
        latent = Conv2D(filters=64, kernel_size=1, activation='relu')(latent)
        latent = Conv2D(filters=128, kernel_size=1, activation='relu')(latent)
        #latent = MaxPooling2D(pool_size=(2, 2))(latent)
        latent = Conv2D(filters=32, kernel_size=1, activation='relu')(latent)
        #latent = Dropout(rate=0.5)(latent)
        latent = Flatten()(latent)
        latent = Dense(units=128, activation='relu')(latent)
        '''
        logits_layer = Dense(units=self.n_classes)(latent)  # 这里不使用激活函数，直接得到logits
        #output_layer = Dense(units=self.n_classes, activation='softmax', kernel_regularizer=l2(0.0003))(latent)
        output_layer = Dense(units=self.n_classes, activation='softmax')(latent)

        # self.logits_model = Model(inputs=input_layer, outputs=logits_layer)  # 新增模型输出logits
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=self.optimizer, loss=self.loss,  metrics=self.metrics)
        
        return logits_layer

    def get_logits(self, x):
        # 新增函数使用logits模型进行预测
        return self.logits_model.predict(x)
    
    def maybe_train(self, data_train, data_valid, batch_size, epochs):

        DIR_ASSETS = 'assets/'
        PATH_MODEL = DIR_ASSETS + 'teacher_model.hdf5'

        if os.path.exists(PATH_MODEL):
            print('Loading trained model from {}.'.format(PATH_MODEL))
            self.model = load_model(PATH_MODEL)
        else:
            print('No checkpoint found on {}. Training from scratch.'.format(
                PATH_MODEL))
            self.build_model()
            x_train, y_train = data_train
            self.model.fit(x_train, y_train, validation_data=data_valid,
                           batch_size=batch_size, epochs=epochs)
            print('Saving trained model to {}.'.format(PATH_MODEL))
            if not os.path.isdir(DIR_ASSETS):
                os.mkdir(DIR_ASSETS)
            self.model.save(PATH_MODEL)

    def evaluate(self, x, y):
        if self.model:
            score = self.model.evaluate(x, y)
            print('accuracy: {:.2f}% | loss: {}'.format(100*score[1], score[0]))
        else:
            print('Missing model instance.')

    def predict(self, x):
        if self.model:
            return self.model.predict(x, verbose=1)
        else:
            print('Missing model instance.')

    def predict_on_batch(self,x):
        if self.model:
            return self.model.predict_on_batch(x)
        else:
            print('Missing initialized model instance.')

    def predict_prob(self, x):
        if self.model:
            return self.model.predict(x.reshape(-1,13,2,1 ), verbose=1) # -1,7,3,1    (-1,13,2,1)
        else:
            print('Missing model instance.')   