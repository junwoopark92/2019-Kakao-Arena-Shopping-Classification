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
os.environ['OMP_NUM_THREADS'] = '1'
import re
import sys
import cPickle
import traceback
from collections import Counter
from multiprocessing import Pool

import tqdm
import fire
import h5py
import numpy as np
from keras.utils.np_utils import to_categorical

from sklearn.externals import joblib
from misc import get_logger, Option
opt = Option('./config.json')

re_sc = re.compile('[\!@#$%\^&\*\(\)=\[\]\{\}\.,/\?~\+\'"|\_\-]')

tfidfvec = joblib.load('../tfidf.vec')
tfdif_size = len(tfidfvec.vocabulary_)

if tfdif_size != int(opt.unigram_hash_size):
    raise Exception

def word2index(word):
    try:
        return tfidfvec.vocabulary_[word.decode('utf8')]
    except Exception as e:
        #print type(e)
        return tfdif_size

useless_token = ['상세', '설명', '참조', '없음', '상품상세']
def remove_token(name):
    for token in useless_token:
        if token in name:
            return ''
    return name

class Reader(object):
    def __init__(self, data_path_list, div, begin_offset, end_offset):
        self.div = div
        self.data_path_list = data_path_list
        self.begin_offset = begin_offset
        self.end_offset = end_offset

    def is_range(self, i):
        if self.begin_offset != None and i < self.begin_offset:
            return False
        if self.end_offset != None and self.end_offset <= i:
            return False
        return True

    def get_size(self):
        offset = 0
        count = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[self.div]['pid'].shape[0]
            if not self.begin_offset and not self.end_offset:
                offset += sz
                count += sz
                continue
            if self.begin_offset and offset + sz < self.begin_offset:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                count += 1
            offset += sz
        return count

    def get_class(self, h, i):
        b = h['bcateid'][i]
        m = -1 #h['mcateid'][i]
        s = -1 #h['scateid'][i]
        d = -1 #h['dcateid'][i]
        return '%s>%s>%s>%s' % (b, m, s, d)

    def generate(self):
        offset = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')[self.div]
            sz = h['pid'].shape[0]
            if self.begin_offset and offset + sz < self.begin_offset:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                class_name = self.get_class(h, i)
                yield h['pid'][i], class_name, h, i
            offset += sz

    def get_y_vocab(self, data_path):
        y_vocab = {}
        h = h5py.File(data_path, 'r')[self.div]
        sz = h['pid'].shape[0]
        for i in tqdm.tqdm(range(sz), mininterval=1):
            class_name = self.get_class(h, i)
            if class_name not in y_vocab:
                y_vocab[class_name] = len(y_vocab)
        return y_vocab


def preprocessing(data):
    try:
        cls, data_path_list, div, out_path, begin_offset, end_offset = data
        data = cls()
        data.load_y_vocab()
        data.preprocessing(data_path_list, div, begin_offset, end_offset, out_path)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def build_y_vocab(data):
    try:
        data_path, div = data
        reader = Reader([], div, None, None)
        y_vocab = reader.get_y_vocab(data_path)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
    return y_vocab


class Data:
    y_vocab_path = './data/y_vocab.cPickle'
    tmp_chunk_tpl = 'tmp/base.chunk.%s'

    def __init__(self):
        self.logger = get_logger('data')

    def load_y_vocab(self):
        self.y_vocab = cPickle.loads(open(self.y_vocab_path).read())

    def build_y_vocab(self):
        pool = Pool(opt.num_workers)
        try:
            rets = pool.map_async(build_y_vocab,
                                  [(data_path, 'train')
                                   for data_path in opt.train_data_list]).get(99999999)
            pool.close()
            pool.join()
            y_vocab = set()
            for _y_vocab in rets:
                for k in _y_vocab.iterkeys():
                    y_vocab.add(k)
            self.y_vocab = {y: idx for idx, y in enumerate(y_vocab)}
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        self.logger.info('size of y vocab: %s' % len(self.y_vocab))
        cPickle.dump(self.y_vocab, open(self.y_vocab_path, 'wb'), 2)

    def _split_data(self, data_path_list, div, chunk_size):
        total = 0
        for data_path in data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[div]['pid'].shape[0]
            total += sz
        chunks = [(i, min(i + chunk_size, total))
                  for i in range(0, total, chunk_size)]
        return chunks

    def preprocessing(self, data_path_list, div, begin_offset, end_offset, out_path):
        self.div = div
        reader = Reader(data_path_list, div, begin_offset, end_offset)
        rets = []
        for pid, label, h, i in reader.generate():
            y, x = self.parse_data(label, h, i)
            if y is None:
                continue
            rets.append((pid, y, x))
        self.logger.info('sz=%s' % (len(rets)))
        open(out_path, 'w').write(cPickle.dumps(rets, 2))
        self.logger.info('%s ~ %s done. (size: %s)' % (begin_offset, end_offset, end_offset - begin_offset))

    def _preprocessing(self, cls, data_path_list, div, chunk_size):
        chunk_offsets = self._split_data(data_path_list, div, chunk_size)
        num_chunks = len(chunk_offsets)
        self.logger.info('split data into %d chunks, # of classes=%s' % (num_chunks, len(self.y_vocab)))
        pool = Pool(opt.num_workers)
        try:
            pool.map_async(preprocessing, [(cls,
                                            data_path_list,
                                            div,
                                            self.tmp_chunk_tpl % cidx,
                                            begin,
                                            end)
                                           for cidx, (begin, end) in enumerate(chunk_offsets)]).get(99999999999)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        return num_chunks

    def parse_data(self, label, h, i):
        Y = self.y_vocab.get(label)
        if Y is None and self.div in ['dev', 'test']:
            Y = 0
        if Y is None and self.div != 'test':
            return [None] * 2
        Y = to_categorical(Y, len(self.y_vocab))

        ori_product = h['product'][i]
        ori_product = re_sc.sub(' ', ori_product).strip()

        brand = h['brand'][i]
        brand = re_sc.sub(' ', brand).strip()

        maker = h['maker'][i]
        maker = re_sc.sub(' ', maker).strip()

        model = h['model'][i]
        model = re_sc.sub(' ', model).strip()

        def merge_brand_maker(maker, brand, model, product):
            maker = remove_token(maker)
            brand = remove_token(brand)
            model = remove_token(model)

            product_tokens = product.split()
            add_model_token = [token for token in model.split() if token not in product_tokens]
            product_tokens = add_model_token + product_tokens
            add_brand_token = [token for token in brand.split() if token not in product_tokens]
            product_tokens = add_brand_token + product_tokens
            add_maker_token = [token for token in maker.split() if token not in product_tokens]
            product_tokens = add_maker_token + product_tokens
            return ' '.join(product_tokens)

        product = merge_brand_maker(maker, brand, model, ori_product).lower()
        if (i+1) % 2000 == 0:
            self.logger.info('[ %s ] -> [ %s ]' % (ori_product, product))

        words = [w.strip() for w in product.split()]
        words = [w for w in words
                 if len(w) >= opt.min_word_length and len(w) < opt.max_word_length]
        if not words:
            return [None] * 2

        wx = [word2index(w) for w in words][:opt.max_len]

        x = np.array([int(opt.unigram_hash_size)]*opt.max_len, dtype=np.float32)
        for i in range(len(wx)):
            x[i] = wx[i]
        return Y, x

    def create_dataset(self, g, size, num_classes):
        shape = (size, opt.max_len)
        g.create_dataset('uni', shape, chunks=True, dtype=np.int32)
        g.create_dataset('cate', (size, num_classes), chunks=True, dtype=np.int32)
        g.create_dataset('pid', (size,), chunks=True, dtype='S12')

    def init_chunk(self, chunk_size, num_classes):
        chunk_shape = (chunk_size, opt.max_len)
        chunk = {}
        chunk['uni'] = np.zeros(shape=chunk_shape, dtype=np.int32)
        chunk['cate'] = np.zeros(shape=(chunk_size, num_classes), dtype=np.int32)
        chunk['pid'] = []
        chunk['num'] = 0
        return chunk

    def copy_chunk(self, dataset, chunk, offset, with_pid_field=False):
        num = chunk['num']
        dataset['uni'][offset:offset + num, :] = chunk['uni'][:num]
        dataset['cate'][offset:offset + num] = chunk['cate'][:num]
        if with_pid_field:
            dataset['pid'][offset:offset + num] = chunk['pid'][:num]

    def copy_bulk(self, A, B, offset, y_offset, with_pid_field=False):
        num = B['cate'].shape[0]
        y_num = B['cate'].shape[1]
        A['uni'][offset:offset + num, :] = B['uni'][:num]
        A['cate'][offset:offset + num, y_offset:y_offset + y_num] = B['cate'][:num]
        if with_pid_field:
            A['pid'][offset:offset + num] = B['pid'][:num]

    def get_train_indices(self, size, train_ratio):
        train_indices = np.random.rand(size) < train_ratio
        train_size = int(np.count_nonzero(train_indices))
        return train_indices, train_size

    def make_db(self, data_name, output_dir='data/train', train_ratio=0.8):
        if data_name == 'train':
            div = 'train'
            data_path_list = opt.train_data_list 
        elif data_name == 'dev':
            div = 'dev'
            data_path_list = opt.dev_data_list 
        elif data_name == 'test':
            div = 'test'
            data_path_list = opt.test_data_list
        else:
            assert False, '%s is not valid data name' % data_name

        all_train = train_ratio >= 1.0
        all_dev = train_ratio == 0.0

        np.random.seed(17)
        self.logger.info('make database from data(%s) with train_ratio(%s)' % (data_name, train_ratio))

        self.load_y_vocab()
        num_input_chunks = self._preprocessing(Data,
                                               data_path_list,
                                               div,
                                               chunk_size=opt.chunk_size)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        data_fout = h5py.File(os.path.join(output_dir, 'data.h5py'), 'w')
        meta_fout = open(os.path.join(output_dir, 'meta'), 'w')

        reader = Reader(data_path_list, div, None, None)
        tmp_size = reader.get_size()
        train_indices, train_size = self.get_train_indices(tmp_size, train_ratio)

        dev_size = tmp_size - train_size
        if all_dev:
            train_size = 1
            dev_size = tmp_size
        if all_train:
            dev_size = 1
            train_size = tmp_size

        train = data_fout.create_group('train')
        dev = data_fout.create_group('dev')
        self.create_dataset(train, train_size, len(self.y_vocab))
        self.create_dataset(dev, dev_size, len(self.y_vocab))
        self.logger.info('train_size ~ %s, dev_size ~ %s' % (train_size, dev_size))

        sample_idx = 0
        dataset = {'train': train, 'dev': dev}
        num_samples = {'train': 0, 'dev': 0}
        chunk_size = opt.db_chunk_size
        chunk = {'train': self.init_chunk(chunk_size, len(self.y_vocab)),
                 'dev': self.init_chunk(chunk_size, len(self.y_vocab))}
        chunk_order = range(num_input_chunks)
        np.random.shuffle(chunk_order)
        for input_chunk_idx in chunk_order:
            path = os.path.join(self.tmp_chunk_tpl % input_chunk_idx)
            self.logger.info('prcessing %s ...' % path)
            data = list(enumerate(cPickle.loads(open(path).read())))
            np.random.shuffle(data)
            for data_idx, (pid, y, x) in data:
                if y is None:
                    continue
                is_train = train_indices[sample_idx + data_idx]
                if all_dev:
                    is_train = False
                if all_train:
                    is_train = True
                if x is None:
                    continue
                c = chunk['train'] if is_train else chunk['dev']
                idx = c['num']
                c['uni'][idx] = x
                c['cate'][idx] = y
                c['num'] += 1
                if not is_train:
                    c['pid'].append(np.string_(pid))
                for t in ['train', 'dev']:
                    if chunk[t]['num'] >= chunk_size:
                        self.copy_chunk(dataset[t], chunk[t], num_samples[t],
                                        with_pid_field=t == 'dev')
                        num_samples[t] += chunk[t]['num']
                        chunk[t] = self.init_chunk(chunk_size, len(self.y_vocab))
            sample_idx += len(data)
        for t in ['train', 'dev']:
            if chunk[t]['num'] > 0:
                self.copy_chunk(dataset[t], chunk[t], num_samples[t],
                                with_pid_field=t == 'dev')
                num_samples[t] += chunk[t]['num']

        for div in ['train', 'dev']:
            ds = dataset[div]
            size = num_samples[div]
            shape = (size, opt.max_len)
            ds['uni'].resize(shape)
            ds['cate'].resize((size, len(self.y_vocab)))

        data_fout.close()
        meta = {'y_vocab': self.y_vocab}
        meta_fout.write(cPickle.dumps(meta, 2))
        meta_fout.close()

        self.logger.info('# of classes: %s' % len(meta['y_vocab']))
        self.logger.info('# of samples on train: %s' % num_samples['train'])
        self.logger.info('# of samples on dev: %s' % num_samples['dev'])
        self.logger.info('data: %s' % os.path.join(output_dir, 'data.h5py'))
        self.logger.info('meta: %s' % os.path.join(output_dir, 'meta'))

if __name__ == '__main__':
    data = Data()
    fire.Fire({'make_db': data.make_db,
               'build_y_vocab': data.build_y_vocab})
