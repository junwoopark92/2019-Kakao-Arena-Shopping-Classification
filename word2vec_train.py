# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

from sklearn.externals import joblib

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences


from multiprocessing import Pool

import urllib
import collections
import os
import zipfile

import numpy as np
import tensorflow as tf

import re
import h5py

from misc import Option
opt = Option('./config.json')

def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

re_sc = re.compile('[\!@#$%\^&\*\(\)=\[\]\{\}\.,/\?~\+\'"|\_\-:]')

useless_token = ['상세', '설명', '참조', '없음', '상품상세']
def remove_token(name):
    for token in useless_token:
        if token in name:
            return ''
    return name


def merge_text(h, i):
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
    if (i + 1) % 2000 == 0:
        print '[ %s ] -> [ %s ]' % (ori_product, product)

    return product


def read_all_data():
    pool = Pool(opt.num_workers)
    sents = []
    try:
        rets = pool.map_async(read_h5py, opt.train_data_list).get(99999999)
        pool.close()
        pool.join()
        [sents.extend(ret) for ret in rets]
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise
    print 'size of y vocab: %s' % len(sents)

    return sents


def read_h5py(path):
    print "Reading file: %s", path
    sequence = []
    h = h5py.File(path, 'r')['train']
    sz = h['pid'].shape[0]
    for i in range(sz):
        sequence.append(merge_text(h, i))

    #words = ' '.join(sequence).split()
    words = sequence
    print "Number of sents ", len(words)
    return words


def build_dataset(sents, n_words, max_len=32):
    """Process raw inputs into a dataset."""

    count = [[u'UNK', -1], [u'<pad>', opt.unigram_hash_size]]
    words = ' '.join(sents).split()

    count.extend(collections.Counter(words).most_common(n_words - 2))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for sent in sents:
        indexs = []
        sent + ' ' + ' '.join(['<pad>']*(max_len - len(words)))
        for word in sent.decode('utf-8').split():
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            indexs.append(index)
        data.append(indexs)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

# def build_dataset(sents, n_words):
#     """Process raw inputs into a dataset."""
#
#     count = [['UNK', -1]]
#     words = ' '.join(sents).split()
#     count.extend(collections.Counter(words).most_common(n_words - 1))
#     dictionary = dict()
#     for word, _ in count:
#         dictionary[word] = len(dictionary)
#     data = list()
#     unk_count = 0
#     for sent in sents:
#         indexs = []
#         for word in sent.split():
#             if word in dictionary:
#                 index = dictionary[word]
#             else:
#                 index = 0  # dictionary['UNK']
#                 unk_count += 1
#             indexs.append(index)
#         data.append(indexs)
#     count[0][1] = unk_count
#     reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#     return data, count, dictionary, reversed_dictionary


def collect_data(vocabulary_size=10000):
    vocabulary = read_all_data()
    print(vocabulary[:7])
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary


vocab_size = opt.unigram_hash_size
data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocab_size)
print(data[:7])

window_size = 3
vector_dim = opt.embd_size
epochs = 50000

valid_size = 4     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

sampling_table = sequence.make_sampling_table(vocab_size)
couples = []
labels = []

for d in data:
    couple, label = skipgrams(d, vocab_size, window_size=window_size, sampling_table=sampling_table)
    couples.extend(couple)
    labels.extend(label)

print(couples[:3], labels[:3])

word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

# create some input variables
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

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

# create a secondary validation model to run our similarity checks during training
validation_model = Model(input=[input_target, input_context], output=similarity)

class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
sim_cb = SimilarityCallback()

arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 100 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))
    if cnt % 10000 == 0:
        sim_cb.run_sim()

embed_matrix = model.layers[2].get_weights()[0]
joblib.dump(embed_matrix, '../embed_matrix.np')
joblib.dump(dictionary, '../word_dict.dict')
print embed_matrix.shape