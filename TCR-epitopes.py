# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 21:00:23 2022

@author: analysis
"""

# import packages
from __future__ import print_function

# from keras.models import Model
# from keras.layers import Input, LSTM, Dense
import numpy as np
import sklearn.preprocessing as skp
import pandas as pd
import lstm_model as lstm 
import argparse
import tensorflow as tf
import pickle

# load data
def load_data(filename):
    # df = pd.read_csv(filename)
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    seq_TCR, seq_epitope = [], []    
    for line in lines[1:]:
        TCR, epitope = line.split(',')
        seq_TCR.append(TCR)
        seq_epitope.append(epitope)
    
    return seq_TCR, seq_epitope

# pad sequences into fixed length
def pad_seq(seq_TCR, seq_epitope):
    max_TCR_seq_length = max([len(seq) for seq in seq_TCR])
    max_epitope_seq_length = max([len(seq) for seq in seq_epitope])
    seq_TCR = [seq + (max_TCR_seq_length - len(seq)) * '.' for seq in seq_TCR]
    seq_epitope = [seq + (max_epitope_seq_length - len(seq)) * '.' for seq in seq_epitope]
    
    return seq_TCR, seq_epitope
   
# transform TCR and epitopes into one-hot encoding
def one_hot_encode(seq_TCR, seq_epitope):
    seq_all = []
    [seq_all.extend(ele) for ele in seq_TCR]
    [seq_all.extend(ele) for ele in seq_epitope] 

    amino = np.reshape(pd.unique(seq_all), (-1, 1))
    enc = skp.OneHotEncoder().fit(amino).transform(amino).toarray()
    maps = {}
    for a, b in zip(amino.squeeze(1), enc):
        maps[a] = b
    
    seq_TCR_array = np.zeros((len(seq_TCR), len(seq_TCR[0]), len(maps)))
    for i in range(len(seq_TCR)):
        seq_TCR_array[i, :, :] = np.asarray([*map(maps.get, seq_TCR[i])])
    
    seq_epitope_array = np.zeros((len(seq_epitope), len(seq_epitope[0]), len(maps)))
    for i in range(len(seq_epitope)):
        seq_epitope_array[i, :, :] = np.asarray([*map(maps.get, seq_epitope[i])])
        
    # for i in range(len(seq_TCR)):
    #     seq_TCR[i] = np.concatenate([*map(maps.get, seq_TCR[i])], 0)
    # seq_TCR = np.asarray(seq_TCR)
    # for i in range(len(seq_epitope)):
    #     seq_epitope[i] = np.concatenate([*map(maps.get, seq_epitope[i])], 0)
    # seq_epitope = np.asarray(seq_epitope)
    
    return seq_TCR_array, seq_epitope_array, maps

# transform TCR and epitopes into label encoding
def label_encode(seq_TCR, seq_epitope):
    seq_all = []
    [seq_all.extend(ele) for ele in seq_TCR]
    [seq_all.extend(ele) for ele in seq_epitope] 
    amino = pd.unique(seq_all)
    enc = skp.LabelEncoder().fit(amino).transform(amino)
    maps = {}
    for a, b in zip(amino, enc):
        maps[a] = b
    
    seq_TCR_array = np.zeros((len(seq_TCR), 1, len(seq_TCR[0])))
    for i in range(len(seq_TCR)):
        seq_TCR_array[i, :, :] = np.reshape(np.asarray([*map(maps.get, seq_TCR[i])]), (1, -1))
    
    seq_epitope_array = np.zeros((len(seq_epitope), 1, len(seq_epitope[0])))
    for i in range(len(seq_epitope)):
        seq_epitope_array[i, :, :] = np.reshape(np.asarray([*map(maps.get, seq_epitope[i])]), (1, -1))
        
    return seq_TCR_array, seq_epitope_array, maps

# remove the padding elements in string 
def depad_seq(seq_all):
    seq_clear = [seq.replace('.', '') for seq in seq_all]
    
    return seq_clear  

# inverse_transform the OneHotEncoding into character        
def decode(maps, seq):
    seq_decode = []
    for i in range(seq.shape[0]):
        # seq_ori = np.reshape(seq[i,:], (-1, len(maps)))
        seq_ori = seq[i,:,:]
        seq_decode_line = []
        for j in range(seq_ori.shape[0]):
            [seq_decode_line.append(key) for key, value in maps.items() if np.array_equal(value, seq_ori[j,:])]
        seq_decode.append(''.join(seq_decode_line))
    
    return seq_decode        
                 
    
# train the paired dataset
def train(args):
    # train 
    seq_TCR, seq_epitope = load_data('train_data.csv')
    seq_TCR, seq_epitope = pad_seq(seq_TCR, seq_epitope)
    # seq_TCR, seq_epitope, train_maps = one_hot_encode(seq_TCR, seq_epitope)
    seq_TCR, seq_epitope, train_maps = label_encode(seq_TCR, seq_epitope)
    
    with open('train_maps.pkl', 'wb') as f:
        pickle.dump(train_maps, f)
    
    fold_percent = 0.9
    train_seq_TCR = seq_TCR[:round(seq_TCR.shape[0]*fold_percent), :, :]
    train_seq_epitope = seq_epitope[:round(seq_epitope.shape[0]*fold_percent), :, :]
   
    # test
    test_seq_TCR = seq_TCR[round(seq_TCR.shape[0]*fold_percent):, :, :]
    test_seq_epitope = seq_epitope[round(seq_epitope.shape[0]*fold_percent):, :, :]
    # test_seq_TCR, test_seq_epitope = load_data('test_data.csv')
    # test_seq_TCR, test_seq_epitope = pad_seq(test_seq_TCR, test_seq_epitope)
    # test_seq_TCR, test_seq_epitope, test_maps = one_hot_encode(test_seq_TCR, test_seq_epitope)
    
    # Train the model
    model = lstm.train_model(train_seq_TCR, train_seq_epitope, test_seq_TCR, test_seq_epitope)
    # model, best_auc, best_roc = lstm.train_model(train_batches, test_batches, args.device, arg, params)
    # Save trained model
    if args.model_file:
        model.save(args.model_file + '.h5')
    #     torch.save({
    #                 'model_state_dict': model.state_dict(),
    #                 'params': params
    #                 }, args.model_file)
    # if args.roc_file:
    #     # Save best ROC curve and AUC
    #     np.savez(args.roc_file, fpr=best_roc[0], tpr=best_roc[1], auc=np.array(best_auc))
    # pass


# predict the new dataset 
def predict(args):
    if args.model_file:
        loaded_model = tf.keras.models.load_model(args.model_file + '.h5')
    
    with open('predict_input.csv', 'r', encoding='utf-8') as f:
        predict_seq_TCR = f.read().split('\n')
    predict_seq_TCR = predict_seq_TCR[1:]
    
    max_TCR_seq_length = max([len(seq) for seq in predict_seq_TCR])
    
    predict_seq_TCR = [seq + (max_TCR_seq_length - len(seq)) * '.' for seq in predict_seq_TCR]
    
    # seq_all = []
    # [seq_all.extend(ele) for ele in predict_seq_TCR]

    # have to use the encoder training used
    with open('train_maps.pkl', 'rb') as f:
        maps = pickle.load(f)
    # amino = np.reshape(pd.unique(seq_all), (-1, 1))
    # enc = skp.OneHotEncoder().fit(amino).transform(amino).toarray()
    # maps = {}
    # for a, b in zip(amino.squeeze(1), enc):
    #     maps[a] = b
    
    
    predict_seq_TCR_array = np.zeros((len(predict_seq_TCR), len(predict_seq_TCR[0]), len(maps)))
    for i in range(len(predict_seq_TCR)):
        predict_seq_TCR_array[i, :, :] = np.asarray([*map(maps.get, predict_seq_TCR[i])])
    
    # predict_seq_epitope = loaded_model.predict(predict_seq_TCR_array, batch_size=50, verbose="auto")
    predict_seq_epitope = loaded_model.predict_on_batch(predict_seq_TCR_array)
    indices = np.expand_dims(np.argmax(predict_seq_epitope, axis = 2), axis = 2)
    predict_seq_epitope_bin = np.zeros_like(predict_seq_epitope)
    np.put_along_axis(predict_seq_epitope_bin, indices, 1, axis = 2)
    # predict_seq_epitope = np.round(predict_seq_epitope)
    predict_seq_epitope = decode(maps, predict_seq_epitope_bin)
    predict_seq_TCR     = depad_seq(predict_seq_TCR)
    predict_seq_epitope = depad_seq(predict_seq_epitope)
    
    if args.predict_output:
        df = pd.DataFrame()
        df['CDR3b'] = predict_seq_TCR
        df['Predicted_epitope'] = predict_seq_epitope
        df.to_csv(args.predict_output+'.csv')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("function")    
    parser.add_argument("--model_file")
    parser.add_argument("--predict_output")
    # parser.add_argument("--roc_file")
    # parser.add_argument("--train_data_file")
    # parser.add_argument("--test_data_file")
    args = parser.parse_args()

    if args.function == 'train':
        train(args)
    elif args.function == 'predict':
        predict(args)


# model training command 
# python TCR-epitopes.py train --model_file=model_0807

# model prediction command
# python TCR-epitopes.py predict --model_file=model_0807 --predict_output=predict_output_0807