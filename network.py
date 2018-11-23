# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

import keras
import keras.backend as K
from keras.models import Model
from keras.layers.merge import dot
from keras.layers import Dense, Input
from keras.layers.core import Reshape

from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional

from attention import Attention
from misc import get_logger, Option
opt = Option('./config.json')

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)


def top1_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=1)


class TextOnly:
    def __init__(self):
        self.logger = get_logger('textonly')

    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')

            t_uni = Input((max_len,), name="input_1")
            t_uni_embd = embd(t_uni)  # token

            w_uni = Input((max_len,), name="input_2")
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)

            embd_out = Dropout(rate=0.5)(uni_embd)
            relu = Activation('relu', name='relu1')(embd_out)
            outputs = Dense(num_classes, activation=activation)(relu)
            model = Model(inputs=[t_uni, w_uni], outputs=outputs)
            optm = keras.optimizers.Nadam(opt.lr)
            model.compile(loss='binary_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class CNNLSTM:
    def __init__(self):
        self.logger = get_logger('cnn-lstm')

    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            model_conv = Sequential()
            model_conv.add(Embedding(voca_size, opt.embd_size, input_length=max_len))
            model_conv.add(Dropout(0.5))
            model_conv.add(Conv1D(64, 5, activation='relu'))
            model_conv.add(MaxPooling1D(pool_size=4))
            model_conv.add(LSTM(32))
            model_conv.add(Dense(num_classes, activation=activation))
            model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=[top1_acc])
            model_conv.summary(print_fn=lambda x: self.logger.info(x))

        return model_conv

class BiLSTM:
    def __init__(self):
        self.logger = get_logger('bilstm')

    def get_model(self,num_classes, activation='sigmoid', mode='sum'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            model = Sequential()
            model.add(Embedding(voca_size, opt.embd_size, input_length=max_len))
            model.add(Bidirectional(LSTM(32), merge_mode=mode))
            model.add(Dense(512, activation='relu'))
            model.add(Dense(num_classes, activation=activation))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class AttentionBiLSTM:
    def __init__(self):
        self.logger = get_logger('attn-bilstm')

    def get_model(self,num_classes, activation='sigmoid', mode='sum'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            model = Sequential()
            model.add(Embedding(voca_size, opt.embd_size, input_length=max_len))
            model.add(Dropout(0.5))
            model.add(Bidirectional(LSTM(32, return_sequences=True), merge_mode=mode))
            model.add(Attention())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',fmeasure, recall, precision])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model
