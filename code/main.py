#This script contains a training process which trains models for each RBP and finds the best set of hyperparameters. 
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Activation, add, Convolution1D, Convolution2D, Dense, Dropout, Embedding
from keras.layers import GRU, Input
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras_self_attention import SeqSelfAttention, ScaledDotProductAttention
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score,\
    accuracy_score, precision_recall_curve
from scipy import interp

import argparse
import time
import math

import sys
import os
import numpy as np
import pickle
import logging
from utility import *
from model import *


batchSize = 20
maxEpochs = 100
gpu_id = '0'
k_mer = 7
seq_len = 101

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf_config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=tf_config)

np.random.seed(4)

def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)

def defineExperimentPaths(basic_path, experimentID):
    experiment_name = experimentID
    MODEL_PATH = basic_path + experiment_name + '/model/'
    LOG_PATH = basic_path + experiment_name + '/logs/'
    CHECKPOINT_PATH = basic_path + experiment_name + '/checkpoints/'
    RESULT_PATH = basic_path + experiment_name + '/results/'
    mk_dir(MODEL_PATH)
    mk_dir(CHECKPOINT_PATH)
    mk_dir(RESULT_PATH)
    mk_dir(LOG_PATH)
    return [MODEL_PATH, CHECKPOINT_PATH, LOG_PATH, RESULT_PATH]


def parse_arguments(parser):
    parser.add_argument('--proteinID', type=str, default='all')
    parser.add_argument('--storage', type=str, default='result/')
    args = parser.parse_args()
    return args

def main(parser):
    file_storage = parser.storage
    basic_path = file_storage + '/'
    protein = parser.proteinID
    # load data
    pos_set, neg_set, bindSiteDict_pos, bindSiteDict_neg = read_fasta_file(protein) # load postive samples and negative samples

    bindsite_all = pos_set + neg_set
    dataY = np.array([1] * len(pos_set) + [0] * len(neg_set))
    dataY = to_categorical(dataY)
    indexes = np.random.choice(len(bindsite_all), len(bindsite_all), replace=False)  # randomly extract elements
    training_idx, test_idx = indexes[:round(((len(bindsite_all)) / 10) * 8)], indexes[
                                                                              round(((len(bindsite_all)) / 10) * 8):]
    train_label = dataY[training_idx, :]

    train_bind_samples = np.array(bindsite_all)[training_idx]

    xSeq_tr, xDotBrack_tr, xLoopType_tr = seqStructMapping(train_bind_samples, bindSiteDict_pos, bindSiteDict_neg)

    vocabDict1, embedding_matrix1 = getVocabIndex_pretrained(k_mer)
    embedding_tr1 = pad_sequences(encodeSeqIndex(xSeq_tr, k_mer, vocabDict1), padding="post", maxlen=seq_len)

    basePair_pos = basePair_Score('PreviousData/positive_profile.out', protein)
    basePair_neg = basePair_Score('PreviousData/negative_profile.out', protein)
    tmpBasePair = np.concatenate((basePair_pos, basePair_neg), axis=0)
    basePair_tr = tmpBasePair[training_idx.tolist()]
    basePair_tr = basePair_tr.astype(np.float)


    rbpchem_dict = getRBPBioChem()
    rbpchem_tr = pad_sequences(np.array([rbpchem_dict[i[1]] for i in train_bind_samples.tolist()]), padding='post',
                               dtype='float', maxlen=168)

    # logging info
    logging.basicConfig(level=logging.DEBUG)
    sys.stdout = sys.stderr
    logging.debug("Loading data...")

   
    kf = KFold(5, True)
    i = 0

    for train_index, eval_index in kf.split(train_label):
        train_X1 = embedding_tr1[train_index]
        train_X2 = basePair_tr[train_index]
        train_X3 = rbpchem_tr[train_index]
        train_y = train_label[train_index]

        eval_X1 = embedding_tr1[eval_index]
        eval_X2 = basePair_tr[eval_index]
        eval_X3 = rbpchem_tr[eval_index]  
        eval_y = train_label[eval_index]

        [MODEL_PATH, CHECKPOINT_PATH, LOG_PATH, RESULT_PATH] = defineExperimentPaths(basic_path, str(i))
        logging.debug("Loading network/training configuration...")
        model = get_model(embedding_matrix1)
        logging.debug("Model summary ... ")
        checkpoint_weight = CHECKPOINT_PATH + "weights.best.hdf5"
        if (os.path.exists(checkpoint_weight)):
            print("load previous best weights")
            model.load_weights(checkpoint_weight)

        model.compile(optimizer='Adam',
                      loss={'ss_output': 'categorical_crossentropy'}, metrics=['accuracy'])
        logging.debug("Running training...")

        def step_decay(epoch):
            initial_lrate = 0.0001
            drop = 0.5
            epochs_drop = 5.0
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            # print(lrate)
            return lrate

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=8, verbose=2, mode='auto'),
            ModelCheckpoint(checkpoint_weight,
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='auto',
                            period=1),
            LearningRateScheduler(step_decay),
            # TensorBoard(log_dir='./log_crd_circrbp', histogram_freq=1, write_grads=True)
        ]
        startTime = time.time()
        history = model.fit(
            {'embedding_input1': train_X1, 'profile_input': train_X2, 'rbp_input': train_X3},
            {'ss_output': train_y},
            epochs=maxEpochs,
            batch_size=batchSize,
            callbacks=callbacks,
            verbose=1,
            validation_data=(
                {'embedding_input1': eval_X1, 'profile_input': eval_X2, 'rbp_input': eval_X3},
                {'ss_output': eval_y}),
            shuffle=True)
        endTime = time.time()
        logging.debug("make prediction")

        i = i + 1
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    start_time = time.time()
    main(args)
    end_time = time.time()
    print("running time is :%.2fs"%(end_time-start_time))
