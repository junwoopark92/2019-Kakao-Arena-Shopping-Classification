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

import os
import fire
import h5py
import numpy as np
import cPickle

from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

from misc import get_logger, Option
from network import MultiTaskAttnWord2vec

from sklearn.externals import joblib
opt = Option('./config.json')

char_tfidf_dict = joblib.load(opt.char_indexer)
char_tfidf_size = len(char_tfidf_dict)

word_tfidf_dict = joblib.load(opt.word_indexer)
word_tfidf_size = len(word_tfidf_dict)


class Word2Vec:
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.num_classes = 0
        self.word_sampling_table = sequence.make_sampling_table(opt.word_voca_size + 2)
        self.char_sampling_table = sequence.make_sampling_table(opt.char_voca_size + 2)

    def get_word2vec_generator(self, ds, batch_size):
        left, limit = 0, ds['wuni'].shape[0]

        while True:
            right = min(left + batch_size, limit)
            product_names = ds['wuni'][left:right, :]
            couples = []
            labels = []
            for name in product_names:
                length = np.where(name < opt.word_voca_size)[0].shape[0] + 3
                couple, label = skipgrams(name[:length], opt.word_voca_size + 2,
                                          window_size=3, sampling_table=self.word_sampling_table)
                couples.extend(couple)
                labels.extend(label)

            word_target, word_context = zip(*couples)
            word_target = np.array(word_target, dtype="int32")
            word_context = np.array(word_context, dtype="int32")
            yield [word_target, word_context], labels
            left = right
            if right == limit:
                left = 0

    def get_char2vec_generator(self, ds, batch_size):
        left, limit = 0, ds['wuni'].shape[0]

        while True:
            right = min(left + batch_size, limit)
            product_names = ds['cuni'][left:right, :]
            couples = []
            labels = []
            for name in product_names:
                length = np.where(name < opt.char_voca_size)[0].shape[0] + 3
                couple, label = skipgrams(name[:length], opt.char_voca_size + 2,
                                          window_size=3, sampling_table=self.char_sampling_table)
                couples.extend(couple)
                labels.extend(label)

            char_target, char_context = zip(*couples)
            char_target = np.array(char_target, dtype="int32")
            char_context = np.array(char_context, dtype="int32")
            yield [char_target, char_context], labels
            left = right
            if right == limit:
                left = 0

    def train(self, data_root, out_dir, resume=False):
        data_path = os.path.join(data_root, 'data.h5py')
        dev_data_path = os.path.join('./data/dev', 'data.h5py')
        meta_path = os.path.join(data_root, 'meta')
        data = h5py.File(data_path, 'r')
        meta = cPickle.loads(open(meta_path).read())

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.logger.info('# of classes: %s' % len(meta['y_vocab']))
        self.num_classes = meta['y_vocab']

        train = data['train']
        train_dev = data['dev']

        dev_data = h5py.File(dev_data_path, 'r')
        dev = dev_data['dev']

        self.logger.info('# of train samples: %s' % train['bcate'].shape[0])
        self.logger.info('# of train dev samples: %s' % train_dev['bcate'].shape[0])
        self.logger.info('# of dev samples: %s' % dev['bcate'].shape[0])

        model = MultiTaskAttnWord2vec(pretrain=resume)
        w2v_model, val_w2v_model = model.get_word2vec_model()
        c2v_model, val_c2v_model = model.get_char2vec_model()

        char_dictionary = char_tfidf_dict
        char_reversed_dictionary = dict(zip(char_dictionary.values(), char_dictionary.keys()))

        word_dictionary = word_tfidf_dict
        word_reversed_dictionary = dict(zip(word_dictionary.values(), word_dictionary.keys()))

        word_sim_cb = SimilarityCallback(val_w2v_model, 100000, 5000, 5, word_reversed_dictionary)
        char_sim_cb = SimilarityCallback(val_c2v_model, char_tfidf_size, char_tfidf_size, 5, char_reversed_dictionary)

        total_train_samples = train['wuni'].shape[0]
        total_train_dev_samples = train_dev['wuni'].shape[0]
        total_dev_samples = dev['wuni'].shape[0]

        w2v_dev_gen = self.get_word2vec_generator(dev, batch_size=opt.batch_size)
        self.dev_steps_per_epoch = int(np.ceil(total_dev_samples / float(opt.batch_size)))
        c2v_dev_gen = self.get_char2vec_generator(dev, batch_size=opt.batch_size)

        w2v_train_gen = self.get_word2vec_generator(train, batch_size=opt.batch_size)
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))
        c2v_train_gen = self.get_char2vec_generator(train, batch_size=opt.batch_size)

        w2v_train_dev_gen = self.get_word2vec_generator(train_dev, batch_size=opt.batch_size)
        self.train_dev_steps_per_epoch = int(np.ceil(total_train_dev_samples / float(opt.batch_size)))
        c2v_train_dev_gen = self.get_char2vec_generator(train_dev, batch_size=opt.batch_size)

        for i in range(5):
            w2v_model.fit_generator(w2v_train_gen,
                                    epochs=1,
                                    steps_per_epoch=self.steps_per_epoch,
                                    shuffle=True)
            w2v_model.fit_generator(w2v_train_dev_gen,
                                    epochs=1,
                                    steps_per_epoch=self.train_dev_steps_per_epoch,
                                    shuffle=True)
            w2v_model.fit_generator(w2v_dev_gen,
                                    epochs=1,
                                    steps_per_epoch=self.dev_steps_per_epoch,
                                    shuffle=True)
            word_sim_cb.run_sim()

        self.logger.info('word2vec model pretrain done')
        word_embed_matrix = w2v_model.layers[2].get_weights()[0]
        joblib.dump(word_embed_matrix, '../word_embed_matrix.np')

        for i in range(5):
            c2v_model.fit_generator(c2v_train_gen,
                                    epochs=1,
                                    steps_per_epoch=self.steps_per_epoch,
                                    shuffle=True)
            c2v_model.fit_generator(c2v_train_dev_gen,
                                    epochs=1,
                                    steps_per_epoch=self.train_dev_steps_per_epoch,
                                    shuffle=True)
            c2v_model.fit_generator(c2v_dev_gen,
                                    epochs=1,
                                    steps_per_epoch=self.dev_steps_per_epoch,
                                    shuffle=True)
            char_sim_cb.run_sim()

        self.logger.info('char2vec model pretrain done')

        char_embed_matrix = c2v_model.layers[2].get_weights()[0]
        joblib.dump(char_embed_matrix, '../char_embed_matrix.np')


class SimilarityCallback:
    def __init__(self, val_model, vocab_size, valid_window, valid_size, reverse_dictionary):
        self.vocab_size = vocab_size
        self.valid_size = valid_size
        self.valid_window = valid_window
        self.reverse_dictionary = reverse_dictionary
        self.val_model = val_model

    def run_sim(self):
        for i in range(self.valid_size):
            valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
            valid_word = self.reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = self.reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    def _get_sim(self, valid_word_idx):
        sim = np.zeros((self.vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(self.vocab_size):
            in_arr2[0,] = i
            out = self.val_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim


if __name__ == '__main__':
    w2v = Word2Vec()
    fire.Fire({'train': w2v.train})
