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
from keras.layers.merge import dot, Multiply
from keras.layers import Dense, Input, merge
from keras.layers.core import Reshape

from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, GlobalMaxPool1D, Dropout, Activation, concatenate
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional

from attention import Attention
from keras_self_attention import SeqSelfAttention
from misc import get_logger, Option
import cPickle

import os
import numpy as np
from sklearn.externals import joblib

opt = Option('./config.json')
meta_path = os.path.join('./data/train', 'meta')
meta = cPickle.loads(open(meta_path).read())
vocab_s = meta['y_vocab'][2]
vocab_d = meta['y_vocab'][3]
batchsize = int(opt.batch_size)

print(len(vocab_s), vocab_s['-1>-1>-1>-1'])
print(len(vocab_d), vocab_d['-1>-1>-1>-1'])

mask_value_s = np.zeros(shape=(len(vocab_s)), dtype=np.int)
mask_value_d = np.zeros(shape=(len(vocab_d)), dtype=np.int)
mask_value_s[vocab_s['-1>-1>-1>-1']] = 1
mask_value_d[vocab_d['-1>-1>-1>-1']] = 1

print(np.argmax(mask_value_s), np.argmax(mask_value_d))


def masked_loss_function_s(y_true, y_pred):
    mask = K.max(K.cast(K.not_equal(y_true, mask_value_s), K.floatx()), axis=1)
    loss = K.categorical_crossentropy(y_true, y_pred) * mask * batchsize / K.sum(mask)
    return loss


def masked_loss_function_d(y_true, y_pred):
    mask = K.max(K.cast(K.not_equal(y_true, mask_value_d), K.floatx()), axis=1)
    loss = K.categorical_crossentropy(y_true, y_pred) * mask * batchsize / K.sum(mask)
    return loss


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


class MultiTaskAttnWord2vec:
    def __init__(self, pretrain=False):
        self.logger = get_logger('baseline')
        self.voca_size = opt.word_voca_size + 2
        self.char_voca_size = opt.char_voca_size + 2

        if pretrain:
            self.logger.info('use pretrained embedding matrix')
            word_embed_matrix = joblib.load('../word_embed_matrix.np')
            char_embed_matrix = joblib.load('../char_embed_matrix.np')
            self.word_embd = Embedding(self.voca_size, opt.word_embd_size, weights=[word_embed_matrix],
                                       name='shared_embed', trainable=True)
            self.char_embd = Embedding(self.char_voca_size, opt.char_embd_size, weights=[char_embed_matrix],
                                       name='shared_char_embed', trainable=True)
        else:
            self.logger.info('no use pretrained embedding matrix')
            self.word_embd = Embedding(self.voca_size, opt.word_embd_size,
                                       name='shared_embed', trainable=True)
            self.char_embd = Embedding(self.char_voca_size, opt.char_embd_size,
                                       name='shared_char_embed', trainable=True)

    def get_char2vec_model(self):
        with tf.device('/gpu:0'):
            input_target = Input((1,))
            input_context = Input((1,))

            target = self.char_embd(input_target)
            target = Reshape((opt.char_embd_size, 1))(target)
            context = self.char_embd(input_context)
            context = Reshape((opt.char_embd_size, 1))(context)

            # setup a cosine similarity operation which will be output in a secondary model
            similarity = merge.dot([target, context], axes=0, normalize=True)

            # now perform the dot product operation to get a similarity measure
            dot_product = merge.dot([target, context], axes=1)
            dot_product = Reshape((1,))(dot_product)
            # add the sigmoid output layer
            output = Dense(1, activation='sigmoid')(dot_product)
            # create the primary training model
            model = Model(input=[input_target, input_context], output=output)
            model.compile(loss='binary_crossentropy', optimizer='rmsprop')
            model.summary()

            validation_model = Model(input=[input_target, input_context], output=similarity)

            return model, validation_model

    def get_word2vec_model(self):
        with tf.device('/gpu:0'):
            input_target = Input((1,))
            input_context = Input((1,))

            target = self.word_embd(input_target)
            target = Reshape((opt.word_embd_size, 1))(target)
            context = self.word_embd(input_context)
            context = Reshape((opt.word_embd_size, 1))(context)

            # setup a cosine similarity operation which will be output in a secondary model
            similarity = merge.dot([target, context], axes=0, normalize=True)

            # now perform the dot product operation to get a similarity measure
            dot_product = merge.dot([target, context], axes=1)
            dot_product = Reshape((1,))(dot_product)
            # add the sigmoid output layer
            output = Dense(1, activation='sigmoid')(dot_product)
            # create the primary training model
            model = Model(input=[input_target, input_context], output=output)
            model.compile(loss='binary_crossentropy', optimizer='rmsprop')
            model.summary()

            validation_model = Model(input=[input_target, input_context], output=similarity)

            return model, validation_model

    def get_classification_model(self, num_classes, activation='sigmoid', mode='sum'):
        char_max_len = opt.char_max_len
        word_max_len = opt.word_max_len

        with tf.device('/gpu:0'):
            img_input = Input((2048,))
            big_img = Dense(128, activation='relu')(img_input)
            mid_img = Dense(128, activation='relu')(img_input)
            s_img = Dense(128, activation='relu')(img_input)
            d_img = Dense(128, activation='relu')(img_input)

            char_input = Input(shape=(char_max_len,), name='char_input')
            word_input = Input(shape=(word_max_len,), name='word_input')

            char_embd = self.char_embd(char_input)
            word_embd = self.word_embd(word_input)
            char_seq = Bidirectional(LSTM(char_max_len, return_sequences=True), merge_mode=mode)(char_embd)

            conv1 = Conv1D(filters=48, kernel_size=2, activation='relu')(char_seq)
            drop1 = Dropout(0.5)(conv1)
            pool1 = MaxPooling1D(pool_size=2)(drop1)

            # channel 2
            conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(char_seq)
            drop2 = Dropout(0.5)(conv2)
            pool2 = MaxPooling1D(pool_size=2)(drop2)

            # channel 3
            conv3 = Conv1D(filters=24, kernel_size=4, activation='relu')(char_seq)
            drop3 = Dropout(0.5)(conv3)
            pool3 = MaxPooling1D(pool_size=2)(drop3)

            # channel 4
            conv4 = Conv1D(filters=16, kernel_size=6, activation='relu')(char_seq)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling1D(pool_size=2)(drop4)

            ngram1, ngram2, ngram3, ngram4, ngram6 = char_seq, pool1, pool2, pool3, pool4

            big_char_ngram = concatenate(
                [Attention()(ngram1), Attention()(ngram2), Attention()(ngram3), Attention()(ngram4),
                 Attention()(ngram6)])
            mid_char_ngram = concatenate(
                [Attention()(ngram1), Attention()(ngram2), Attention()(ngram3), Attention()(ngram4),
                 Attention()(ngram6)])
            s_char_ngram = concatenate(
                [Attention()(ngram1), Attention()(ngram2), Attention()(ngram3), Attention()(ngram4),
                 Attention()(ngram6)])
            d_char_ngram = concatenate(
                [Attention()(ngram1), Attention()(ngram2), Attention()(ngram3), Attention()(ngram4),
                 Attention()(ngram6)])

            big_word_layer = SeqSelfAttention(attention_activation='sigmoid')(word_embd)
            big_word_layer = Attention()(big_word_layer)

            big_layer = concatenate([big_word_layer, big_char_ngram, big_img])
            big_layer = Dropout(0.5)(big_layer)
            big_out = Dense(len(num_classes[0]), activation='softmax', name='big')(big_layer)

            mid_word_layer = SeqSelfAttention(attention_activation='sigmoid')(word_embd)
            mid_word_layer = Attention()(mid_word_layer)

            mid_layer = concatenate([mid_word_layer, mid_char_ngram, mid_img])
            mid_layer = Dropout(0.5)(mid_layer)
            mid_out = Dense(len(num_classes[1]), activation='softmax', name='mid')(mid_layer)

            s_word_layer = SeqSelfAttention(attention_activation='sigmoid')(word_embd)
            s_word_layer = Attention()(s_word_layer)

            s_layer = concatenate([s_word_layer, s_char_ngram, s_img])
            s_layer = Dropout(0.5)(s_layer)
            s_out = Dense(len(num_classes[2]), activation='softmax', name='small')(s_layer)

            d_word_layer = SeqSelfAttention(attention_activation='sigmoid')(word_embd)
            d_word_layer = Attention()(d_word_layer)

            d_layer = concatenate([d_word_layer, d_char_ngram, d_img])
            d_layer = Dropout(0.5)(d_layer)
            d_out = Dense(len(num_classes[3]), activation='softmax', name='detail')(d_layer)

            model = Model(inputs=[char_input, word_input, img_input],
                          outputs=[big_out, mid_out, s_out, d_out])

            model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy',
                                masked_loss_function_s, masked_loss_function_d],
                          optimizer='rmsprop', metrics=['accuracy', fmeasure, recall, precision])

            model.summary(print_fn=lambda x: self.logger.info(x))
        return model