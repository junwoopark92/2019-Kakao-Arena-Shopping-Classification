# -*- coding: utf-8 -*-

import re
import fire
import h5py
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.externals import joblib
from glob import glob
from misc import get_logger, Option


re_sc = re.compile('[\!@#$%\^&\*\(\)=\[\]\{\}\.,/\?~\+\'"|\_\-:]')
useless_token = [u'상세', u'설명', u'참조', u'없음', u'상품상세', u'기타']


def remove_token(name):
    for token in useless_token:
        if token in name:
            return u''
    return name


def merge_brand_maker(maker, brand, model, product, space):
    maker = remove_token(maker)
    brand = remove_token(brand)
    model = remove_token(model)
    product = product

    product_tokens = product.split()
    add_model_token = [token for token in model.split() if token not in product_tokens]
    product_tokens = add_model_token + product_tokens
    add_brand_token = [token for token in brand.split() if token not in product_tokens]
    product_tokens = add_brand_token + product_tokens
    add_maker_token = [token for token in maker.split() if token not in product_tokens]
    product_tokens = add_maker_token + product_tokens

    return space.join(product_tokens)


def refine_rawtext(row, unit='word'):
    space = None
    if unit != 'word' and unit != 'char':
        raise Exception()

    if unit == 'word':
        space = u' '
    if unit == 'char':
        space = u'ⓢ'

    ori_product = re_sc.sub(' ', row['product'].decode('utf-8')).strip()
    brand = re_sc.sub(' ', row['brand'].decode('utf-8')).strip()
    maker = re_sc.sub(' ', row['maker'].decode('utf-8')).strip()
    model = re_sc.sub(' ', row['model'].decode('utf-8')).strip()

    product = merge_brand_maker(maker, brand, model, ori_product, space).lower()
    return product


class RawData:
    def __init__(self):
        self.logger = get_logger('data')

    def get_filenames(self, dirpath):
        filenames = [filename for filename in glob(dirpath+'*') if 'test' not in filename]
        return filenames

    def merge_chunks(self, dirpath):
        sub_list = []
        for filename in self.get_filenames(dirpath):
            if 'train' in filename:
                div = 'train'
            elif 'dev' in filename:
                div = 'dev'
            else:
                continue

            origin = h5py.File(filename, 'r')[div]
            prd = origin['product']
            model = origin['model']
            brand = origin['brand']
            maker = origin['maker']
            bcate = origin['bcateid']
            mcate = origin['mcateid']
            scate = origin['scateid']
            dcate = origin['dcateid']
            sub_list.append(pd.DataFrame({'product': prd, 'model': model, 'maker': maker, 'brand': brand,
                                          'bcate': bcate, 'mcate': mcate, 'scate': scate, 'dcate': dcate}))
            self.logger.info(filename + ' loaded size: %s' % len(sub_list[-1]))

        merged_df = pd.concat(sub_list).reset_index(drop=True)
        return merged_df

    def make_tfidf_vec(self, dirpath, sep='word', max_features=200000, min_df=3):
        df = self.merge_chunks(dirpath)
        doc_list = df.apply(lambda row:refine_rawtext(row, sep), axis=1)
        if sep == 'char':
            doc_list = [' '.join(list(doc)) for doc in doc_list]

        self.logger.info('merge size: %s' % len(df))
        for i in range(5):
            self.logger.info(doc_list[i])

        vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r'\S+', min_df=min_df).fit(doc_list)

        self.logger.info('tfidf vocab size: %s' % len(vectorizer.vocabulary_))
        for i, (key, value) in enumerate(vectorizer.vocabulary_.items()):
            self.logger.info('%s : %d' % (key, value))
            if i > 5:
                break

        joblib.dump(vectorizer.vocabulary_, '../{}_tfidf_{}_min1.dict'.format(sep, str(max_features)[:3]))

    def make_count_vec(self, dirpath, sep='word', max_features=200000, min_df=3):
        df = self.merge_chunks(dirpath)
        doc_list = df.apply(lambda row: refine_rawtext(row, sep), axis=1)
        if sep == 'char':
            doc_list = [' '.join(list(doc)) for doc in doc_list]

        self.logger.info('merge size: %s' % len(df))
        for i in range(5):
            self.logger.info(doc_list[i])

        vectorizer = CountVectorizer(max_features=max_features, token_pattern=r'\S+', min_df=min_df).fit(doc_list)

        self.logger.info('count vocab size: %s' % len(vectorizer.vocabulary_))
        for i, (key, value) in enumerate(vectorizer.vocabulary_.items()):
            self.logger.info('%s : %d' % (key, value))
            if i > 5:
                break

        joblib.dump(vectorizer.vocabulary_, '../{}_count_{}.dict'.format(sep, str(max_features)[:3]))


if __name__ == '__main__':
    data = RawData()
    fire.Fire({'make_tfidf_vec': data.make_tfidf_vec,
               'make_count_vec': data.make_count_vec})
