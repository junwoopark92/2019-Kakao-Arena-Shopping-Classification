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
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, concatenate
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional

from attention import Attention
from keras_self_attention import SeqSelfAttention
from misc import get_logger, Option
import os
import cPickle
import numpy as np
from sklearn.externals import joblib

opt = Option('./config.json')
meta_path = os.path.join('./data/train', 'meta')
meta = cPickle.loads(open(meta_path).read())
vocab_s = meta['y_vocab'][2]
vocab_d = meta['y_vocab'][3]
batchsize = int(opt.batch_size)

print len(vocab_s), vocab_s['-1>-1>-1>-1']
print len(vocab_d), vocab_d['-1>-1>-1>-1']

mask_value_s = np.zeros(shape=(len(vocab_s)), dtype=np.int)
mask_value_d = np.zeros(shape=(len(vocab_d)), dtype=np.int)
mask_value_s[vocab_s['-1>-1>-1>-1']] = 1
mask_value_d[vocab_d['-1>-1>-1>-1']] = 1

print np.argmax(mask_value_s), np.argmax(mask_value_d)


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
        voca_size = opt.unigram_hash_size + 2

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


class AttentionBiLSTMCls:
    def __init__(self):
        self.logger = get_logger('attn-bilstm-cls')

    def get_model(self,num_classes, activation='sigmoid', mode='sum'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            textmodel = Sequential()
            textmodel.add(Embedding(voca_size, opt.embd_size, input_length=max_len))
            textmodel.add(Dropout(0.5))
            textmodel.add(Bidirectional(LSTM(32, return_sequences=True), merge_mode=mode))
            textmodel.add(Attention())

            img_model = Sequential()
            img_model.add(Dense(128, input_shape=(2048,),activation='relu'))
            img_model.add(Dropout(0.5))

            merged_layers = concatenate([textmodel.output, img_model.output])
            out = Dense(num_classes, activation='sigmoid')(merged_layers)

            model = Model([textmodel.input, img_model.input], out)

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[top1_acc, fmeasure, recall, precision])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class MultiTaskAttnImg:
    def __init__(self):
        self.logger = get_logger('attn-bilstm-cls')
        self.embed_matrix = joblib.load('../embed_matrix.np')

    def get_model(self, num_classes, activation='sigmoid', mode='sum'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size# + 1
        with tf.device('/gpu:0'):

            img_input = Input((2048,))
            #img_model = Sequential() 
            #img_model.add(Dense(128, input_shape=(2048,), activation='relu'))
            #big_attn_prob = Dense(2048, activation='softmax')(img_input)
            #big_attn = Multiply()([img_input, big_attn_prob])
            big_attn = Dense(128, activation='relu')(img_input)
           
            #mid_attn_prob = Dense(2048, activation='softmax')(img_input)
            #mid_attn = Multiply()([img_input, mid_attn_prob])
            mid_attn = Dense(128, activation='relu')(img_input)

            
            #s_attn_prob = Dense(2048, activation='softmax')(img_input)
            #s_attn = Multiply()([img_input, s_attn_prob])
            s_attn = Dense(128, activation='relu')(img_input)
            
            #d_attn_prob = Dense(2048, activation='softmax')(img_input)
            #d_attn = Multiply()([img_input, d_attn_prob])
            d_attn = Dense(128, activation='relu')(img_input)
            
            big_input = Input(shape=(max_len,), name='big_input')
            big_embd = Embedding(voca_size, opt.embd_size,
                                 weights=[self.embed_matrix], name='big_embd', trainable=True)

            big_layer = big_embd(big_input)
            big_layer = SeqSelfAttention(attention_activation='sigmoid')(big_layer)
            big_layer = SeqSelfAttention(attention_activation='sigmoid')(big_layer)
            big_layer = Attention()(big_layer)
            big_layer = concatenate([big_layer, big_attn])
            big_layer = Dropout(0.5)(big_layer)
            big_out = Dense(len(num_classes[0]), activation='softmax', name='big')(big_layer)

            mid_layer = big_embd(big_input)
            mid_layer = SeqSelfAttention(attention_activation='sigmoid')(mid_layer)
            mid_layer = SeqSelfAttention(attention_activation='sigmoid')(mid_layer)
            mid_layer = Attention()(mid_layer)
            mid_layer = concatenate([mid_layer, big_out, mid_attn])
            mid_layer = Dropout(0.5)(mid_layer)
            mid_out = Dense(len(num_classes[1]), activation='softmax', name='mid')(mid_layer)

            s_layer = big_embd(big_input)
            s_layer = SeqSelfAttention(attention_activation='sigmoid')(s_layer)
            s_layer = SeqSelfAttention(attention_activation='sigmoid')(s_layer)
            s_layer = Attention()(s_layer)
            s_layer = concatenate([s_layer, mid_out, s_attn])
            s_layer = Dropout(0.5)(s_layer)
            s_out = Dense(len(num_classes[2]), activation='softmax', name='small')(s_layer)

            d_layer = big_embd(big_input)
            d_layer = SeqSelfAttention(attention_activation='sigmoid')(d_layer)
            d_layer = SeqSelfAttention(attention_activation='sigmoid')(d_layer)
            d_layer = Attention()(d_layer)
            d_layer = concatenate([d_layer, s_out, d_attn])
            d_layer = Dropout(0.5)(d_layer)
            d_out = Dense(len(num_classes[3]), activation='softmax', name='detail')(d_layer)

            model = Model(inputs=[big_input, img_input],
                          outputs=[big_out, mid_out, s_out, d_out])

            # model.compile(loss='categorical_crossentropy',
            #               optimizer='adam', metrics=['accuracy', fmeasure, recall, precision])

            model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy',
                                masked_loss_function_s, masked_loss_function_d],
                          optimizer='adam', metrics=['accuracy', fmeasure, recall, precision])

            model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class MultiTaskAttnWord2vec:
    def __init__(self):
        self.logger = get_logger('attn-bilstm-cls')
        self.embed_matrix = joblib.load('../embed_matrix.np')
        self.voca_size = opt.word_voca_size + 2
        self.char_voca_size = opt.char_voca_size + 2
        self.word_embd = Embedding(self.voca_size, opt.word_embd_size, name='shared_embed', trainable=True)

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

            char_embd = Embedding(self.char_voca_size, opt.char_embd_size, name='char_shared_embed', trainable=True)(char_input)
            word_embd = self.word_embd(word_input)
            char_seq = Bidirectional(LSTM(char_max_len, return_sequences=True), merge_mode=mode)(char_embd)

            big_word_layer = SeqSelfAttention(attention_activation='sigmoid')(word_embd)
            big_word_layer = Attention()(big_word_layer)

            big_char_layer = Attention()(char_seq)

            big_layer = concatenate([big_word_layer, big_char_layer, big_img])
            big_layer = Dropout(0.5)(big_layer)
            big_out = Dense(len(num_classes[0]), activation='softmax', name='big')(big_layer)

            mid_word_layer = SeqSelfAttention(attention_activation='sigmoid')(word_embd)
            mid_word_layer = Attention()(mid_word_layer)

            mid_char_layer = Attention()(char_seq)

            mid_layer = concatenate([mid_word_layer, mid_char_layer, mid_img])
            mid_layer = Dropout(0.5)(mid_layer)
            mid_out = Dense(len(num_classes[1]), activation='softmax', name='mid')(mid_layer)

            s_word_layer = SeqSelfAttention(attention_activation='sigmoid')(word_embd)
            s_word_layer = Attention()(s_word_layer)

            s_char_layer = Attention()(char_seq)

            s_layer = concatenate([s_word_layer, s_char_layer, s_img])
            s_layer = Dropout(0.5)(s_layer)
            s_out = Dense(len(num_classes[2]), activation='softmax', name='small')(s_layer)

            d_word_layer = SeqSelfAttention(attention_activation='sigmoid')(word_embd)
            d_word_layer = Attention()(d_word_layer)

            d_char_layer = Attention()(char_seq)

            d_layer = concatenate([d_word_layer, d_char_layer, d_img])
            d_layer = Dropout(0.5)(d_layer)
            d_out = Dense(len(num_classes[3]), activation='softmax', name='detail')(d_layer)

            model = Model(inputs=[char_input, word_input, img_input],
                          outputs=[big_out, mid_out, s_out, d_out])

            model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy',
                                masked_loss_function_s, masked_loss_function_d],
                          optimizer='adam', metrics=['accuracy', fmeasure, recall, precision])

            model.summary(print_fn=lambda x: self.logger.info(x))
        return model