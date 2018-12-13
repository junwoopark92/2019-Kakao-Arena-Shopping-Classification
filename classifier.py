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
import json
import cPickle
from itertools import izip

import fire
import h5py
import numpy as np

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

from attention import Attention
from keras_self_attention import SeqSelfAttention
from misc import get_logger, Option
from network import TextOnly, CNNLSTM, BiLSTM, AttentionBiLSTM, AttentionBiLSTMCls, MultiTaskAttnImg, \
    top1_acc, fmeasure, precision, recall, masked_loss_function_d, masked_loss_function_s, MultiTaskAttnWord2vec

from sklearn.externals import joblib
opt = Option('./config.json')
cate1 = json.loads(open('../cate1.json').read())
DEV_DATA_LIST = opt.dev_data_list#['/ssd2/dataset/dev.chunk.01']
TRAIN_DATA_LIST = ['./data/train/data.h5py']

tfidfvec = joblib.load('../tfidf20.vec')
tfdif_size = len(tfidfvec.vocabulary_)

class Classifier():
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.num_classes = 0
        self.sampling_table = sequence.make_sampling_table(opt.unigram_hash_size + 2)

    def get_sample_generator(self, ds, batch_size):
        left, limit = 0, ds['uni'].shape[0]
        while True:
            right = min(left + batch_size, limit)
            X = [ds[t][left:right, :] for t in ['uni', 'img']]
            Y = [ds[hirachi+'cate'][left:right] for hirachi in ['b','m','s','d']]
            yield X, Y
            left = right
            if right == limit:
                left = 0

    def get_word2vec_generator(self, ds, batch_size):
        left, limit = 0, ds['uni'].shape[0]

        while True:
            right = min(left + batch_size, limit)
            product_names = ds['uni'][left:right, :]
            couples = []
            labels = []
            for name in product_names:
                length = np.where(name != opt.unigram_hash_size+1)[0].shape[0] + 1
                couple, label = skipgrams(name[:length], opt.unigram_hash_size + 2,
                                          window_size=3, sampling_table=self.sampling_table)
                couples.extend(couple)
                labels.extend(label)

            word_target, word_context = zip(*couples)
            word_target = np.array(word_target, dtype="int32")
            word_context = np.array(word_context, dtype="int32")
            yield [word_target, word_context], labels
            left = right
            if right == limit:
                left = 0

    def get_inverted_cate1(self, cate1):
        inv_cate1 = {}
        for d in ['b', 'm', 's', 'd']:
            inv_cate1[d] = {v: k for k, v in cate1[d].iteritems()}
        return inv_cate1

    def write_prediction_result(self, data, pred_y, meta, out_path, readable, istrain=False):
        pid_order = []
        dev_data_list = DEV_DATA_LIST
        if istrain:
            dev_data_list = TRAIN_DATA_LIST
        for data_path in dev_data_list:
            h = h5py.File(data_path, 'r')['dev']
            pid_order.extend(h['pid'][::])

        y2l_b = {i: s for s, i in meta['y_vocab'][0].iteritems()}
        y2l_b = map(lambda x: x[1], sorted(y2l_b.items(), key=lambda x: x[0]))

        y2l_m = {i: s for s, i in meta['y_vocab'][1].iteritems()}
        y2l_m = map(lambda x: x[1], sorted(y2l_m.items(), key=lambda x: x[0]))

        y2l_s = {i: s for s, i in meta['y_vocab'][2].iteritems()}
        y2l_s= map(lambda x: x[1], sorted(y2l_s.items(), key=lambda x: x[0]))

        y2l_d = {i: s for s, i in meta['y_vocab'][3].iteritems()}
        y2l_d = map(lambda x: x[1], sorted(y2l_d.items(), key=lambda x: x[0]))

        pred_b = pred_y[0]
        pred_m = pred_y[1]
        pred_s = pred_y[2]
        pred_d = pred_y[3]

        inv_cate1 = self.get_inverted_cate1(cate1)
        rets = {}
        for pid, p_b, p_m, p_s, p_d in izip(data['pid'], pred_b, pred_m, pred_s, pred_d):
            y_b = np.argmax(p_b)
            y_m = np.argmax(p_m)
            y_s = np.argmax(p_s)
            y_d = np.argmax(p_d)

            label_b = y2l_b[y_b]
            label_m = y2l_m[y_m]
            label_s = y2l_s[y_s]
            label_d = y2l_d[y_d]

            #print(label_b, label_m, label_s, label_d)

            b = label_b.split('>')[0]
            m = label_m.split('>')[1]
            s = label_s.split('>')[2]
            d = label_d.split('>')[3]
            # assert b in inv_cate1['b']
            # assert m in inv_cate1['m']
            # assert s in inv_cate1['s']
            # assert d in inv_cate1['d']
            tpl = '{pid}\t{b}\t{m}\t{s}\t{d}'
            if readable:
                b = inv_cate1['b'][b]
                m = inv_cate1['m'][m]
                s = inv_cate1['s'][s]
                d = inv_cate1['d'][d]
            rets[pid] = tpl.format(pid=pid, b=b, m=m, s=s, d=d)
        no_answer = '{pid}\t-1\t-1\t-1\t-1'
        with open(out_path, 'w') as fout:
            for pid in pid_order:
                ans = rets.get(pid, no_answer.format(pid=pid))
                print >> fout, ans

    def predict(self, data_root, model_root, test_root, test_div, out_path, readable=False):
        meta_path = os.path.join(data_root, 'meta')
        meta = cPickle.loads(open(meta_path).read())

        model_fname = os.path.join(model_root, 'model.h5')
        self.logger.info('# of classes(train): %s' % len(meta['y_vocab']))
        model = load_model(model_fname,
                           custom_objects={'top1_acc': top1_acc,
                                           'Attention':Attention,
                                           'SeqSelfAttention':SeqSelfAttention,
                                           'fmeasure':fmeasure,
                                           'precision':precision,
                                           'recall':recall,
                                           'masked_loss_function_d':masked_loss_function_d,
                                           'masked_loss_function_s':masked_loss_function_s})
        istrain = test_root.split('/')[-2] == 'train'
        print(istrain, test_root)
        test_path = os.path.join(test_root, 'data.h5py')
        test_data = h5py.File(test_path, 'r')

        test = test_data[test_div]
        test_gen = self.get_sample_generator(test, opt.batch_size)
        total_test_samples = test['uni'].shape[0]
        steps = int(np.ceil(total_test_samples / float(opt.batch_size)))
        pred_y = model.predict_generator(test_gen,
                                         steps=steps,
                                         workers=opt.num_predict_workers,
                                         verbose=1,)
        self.write_prediction_result(test, pred_y, meta, out_path, readable=readable, istrain=istrain)

    def train(self, data_root, out_dir, resume=False):
        data_path = os.path.join(data_root, 'data.h5py')
        meta_path = os.path.join(data_root, 'meta')
        data = h5py.File(data_path, 'r')
        meta = cPickle.loads(open(meta_path).read())
        self.weight_fname = os.path.join(out_dir, 'weights')
        self.model_fname = os.path.join(out_dir, 'model')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.logger.info('# of classes: %s' % len(meta['y_vocab']))
        self.num_classes = meta['y_vocab']

        train = data['train']
        dev = data['dev']

        self.logger.info('# of train samples: %s' % train['bcate'].shape[0])
        self.logger.info('# of dev samples: %s' % dev['bcate'].shape[0])

        checkpoint = ModelCheckpoint(self.weight_fname, monitor='val_loss',
                                     save_best_only=True, mode='min', period=1)

        classification_model = None
        w2v_model = None
        val_w2v_model = None
        if not resume:
            #textonly = TextOnly()
            #textonly = CNNLSTM()
            #textonly = BiLSTM()
            #textonly = AttentionBiLSTM()
            #textonly = AttentionBiLSTMCls()
            #textonly = MultiTaskAttnImg()
            textonly = MultiTaskAttnWord2vec()
            classification_model = textonly.get_classification_model(self.num_classes, mode='sum')
            w2v_model, val_w2v_model = textonly.get_word2vec_model()

        else:
            model_fname = os.path.join(out_dir, 'model.h5')
            model = load_model(model_fname, custom_objects={'top1_acc':top1_acc,
                                                            'Attention':Attention,
                                                            'SeqSelfAttention':SeqSelfAttention,
                                                            'fmeasure':fmeasure,
                                                            'precision':precision,
                                                            'recall':recall,
                                                            'masked_loss_function_d':masked_loss_function_d,
                                                            'masked_loss_function_s':masked_loss_function_s})

        dictionary = tfidfvec.vocabulary_
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        sim_cb = SimilarityCallback(val_w2v_model, tfdif_size, tfdif_size, 5, reversed_dictionary)

        total_train_samples = train['uni'].shape[0]
        train_gen = self.get_sample_generator(train,
                                              batch_size=opt.batch_size)
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = dev['uni'].shape[0]
        dev_gen = self.get_sample_generator(dev,
                                            batch_size=opt.batch_size)
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        w2v_train_gen = self.get_word2vec_generator(train,
                                              batch_size=opt.batch_size)
        #self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        w2v_dev_gen = self.get_word2vec_generator(dev,
                                            batch_size=opt.batch_size)
        #self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        for i in range(10):
            w2v_model.fit_generator(w2v_train_gen,
                                epochs=1,
                                steps_per_epoch=self.steps_per_epoch,
                                shuffle=True)
            sim_cb.run_sim()

        self.logger.info('word2vec model pretrain done')

        classification_model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=[checkpoint])

        classification_model.load_weights(self.weight_fname) # loads from checkout point if exists
        open(self.model_fname + '.json', 'w').write(classification_model.to_json())
        classification_model.save(self.model_fname + '.h5')


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
    clsf = Classifier()
    fire.Fire({'train': clsf.train,
               'predict': clsf.predict})
