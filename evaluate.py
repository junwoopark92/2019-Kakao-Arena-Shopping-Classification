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
import cPickle
from itertools import izip

import fire
import h5py
import numpy as np
import pandas as pd

def evaluate(predict_path, data_path, div, y_vocab_path):
    h = h5py.File(data_path, 'r')[div]
    inv_y_vocab_b = {v: k
                   for k, v in cPickle.loads(open(y_vocab_path+'.b').read()).iteritems()}
    inv_y_vocab_m = {v: k
                     for k, v in cPickle.loads(open(y_vocab_path+'.m').read()).iteritems()}
    inv_y_vocab_s = {v: k
                     for k, v in cPickle.loads(open(y_vocab_path+'.s').read()).iteritems()}
    inv_y_vocab_d = {v: k
                     for k, v in cPickle.loads(open(y_vocab_path+'.d').read()).iteritems()}

    fin = open(predict_path)
    hit, n = {}, {'b': 0, 'm': 0, 's': 0, 'd': 0}
    print 'loading ground-truth...'
    print(h.keys())
    CATE_b = np.argmax(h['bcate'], axis=1)
    CATE_m = np.argmax(h['mcate'], axis=1)
    CATE_s = np.argmax(h['scate'], axis=1)
    CATE_d = np.argmax(h['dcate'], axis=1)
    results =[]
    for p, gt_b, gt_m, gt_s, gt_d in izip(fin, CATE_b, CATE_m, CATE_s, CATE_d):
        pid, b, m, s, d = p.split('\t')
        b, m, s, d = map(int, [b, m, s, d])
        gt = map(int,[inv_y_vocab_b[gt_b].split('>')[0],
              inv_y_vocab_m[gt_m].split('>')[1],
              inv_y_vocab_s[gt_s].split('>')[2],
              inv_y_vocab_d[gt_d].split('>')[3]])
        results.append({'pid':pid,'pred':[b,m,s,d],'gt':gt})
        for depth, _p, _g in zip(['b', 'm', 's', 'd'],
                                 [b, m, s, d],
                                 gt):
            if _g == -1:
                continue
            n[depth] = n.get(depth, 0) + 1
            if _p == _g:
                hit[depth] = hit.get(depth, 0) + 1
    pd.DataFrame(results).to_csv('./comp_predict.tsv',sep='\t',index=False)
    for d in ['b', 'm', 's', 'd']:
        if n[d] > 0:
            print '%s-Accuracy: %.3f(%s/%s)' % (d, hit[d] / float(n[d]), hit[d], n[d])
    score = sum([hit[d] / float(n[d]) * w
                 for d, w in zip(['b', 'm', 's', 'd'],
                                 [1.0, 1.2, 1.3, 1.4])]) / 4.0
    print 'score: %.3f' % score

if __name__ == '__main__':
    fire.Fire({'evaluate': evaluate})
