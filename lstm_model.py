# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 15:18:07 2022

@author: analysis
"""

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from random import shuffle
# import time
import numpy as np
# import torch.autograd as autograd
# from ERGO_models import DoubleLSTMClassifier
# from sklearn.metrics import roc_auc_score, roc_curve
# from keras.models import Model
# from keras.layers import Input, LSTM, Dense
import tensorflow as tf


def train_model(train_seq_TCR, train_seq_epitope, test_seq_TCR, test_seq_epitope):
           
    batch_size = 10
    epoch_size = 500
    
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(train_seq_TCR.shape[1],train_seq_TCR.shape[2])))
    model.add(tf.keras.layers.Normalization(axis = 2, mean = 0.5, variance = 0.5))
    # model.add(tf.keras.layers.LSTM(36, activation = 'relu', return_sequences = True, kernel_initializer = tf.keras.initializers.random_normal(stddev=0.01)))
    model.add(tf.keras.layers.LSTM(16, activation = 'relu', return_sequences = True))
    # model.add(tf.keras.layers.LSTM(64, return_sequences = True))
    # model.add(tf.keras.layers.LSTM(32, return_sequences = True))
    # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.LSTM(16, return_sequences = True))
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(8, activation = 'relu', return_sequences = True))
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(4, activation = 'relu'))
    model.add(tf.keras.layers.Dense(train_seq_epitope.shape[1]*train_seq_epitope.shape[2], activation = 'softmax'))
    model.add(tf.keras.layers.Reshape((train_seq_epitope.shape[1], train_seq_epitope.shape[2])))
    
    # model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.compile(
              # optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
              optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum = 0.2), 
              # optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-2, rho = 0.9),
              # loss=tf.keras.losses.BinaryCrossentropy(),
              # loss=tf.keras.losses.CategoricalCrossentropy(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.Accuracy(),
                       tf.keras.metrics.AUC(),
                       # tf.keras.metrics.FalseNegatives()
                       ])
    print(model)
    print('Train...')
    model.fit(train_seq_TCR, train_seq_epitope,
              batch_size = batch_size,
              epochs = epoch_size,
              shuffle = True, 
              validation_split = 0.1,
              # validation_data=[test_seq_TCR, test_seq_epitope]
              )
    # model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
    
    return model
    
    # encoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    # encoder_embedding = embed_layer(encoder_inputs)
    # encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    # encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    
    # decoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    # decoder_embedding = embed_layer(decoder_inputs)
    # decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    # decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
    
    # # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
    # outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    # model = Model([encoder_inputs, decoder_inputs], outputs)