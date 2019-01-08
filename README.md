# shopping-classification

`쇼핑몰 상품 카테고리 분류` 대회 참가자에게 제공되는 코드 베이스입니다. 전처리와 간단한 분류기 그리고 평가 코드로 구성되어 있습니다. (코드는 python2.7/3.5+, keras, tensorflow 기준으로 작성되었습니다.)

## UPDATED
  - 2018.11.22 python3가 호환됩니다. [PR](https://github.com/kakao-arena/shopping-classification/pull/3)
  - py3환경> python data.py make_db multiprocessing deadlock 이슈
  - py2환경으로 제출

## 실행 방법

0. 데이터의 위치
    - {/datadir/} 아래에 chunk들이 모두 있다고 가정합니다. ex) `/dataset/train.chunk.01 ~ 09, dev.chunk.01, test.chunk.01 ~ 02`
1. `python sent2index.py make_tfidf_vec {/datadir/} word`
    - 문장에서 단어단위로 인덱싱을 합니다. ex) 나이키 -> 1, 신발 -> 2
2. `python sent2index.py make_tfidf_vec {/datadir/} char --min_df=1`
    - 문장에서 char단위로 인덱싱을 합니다. ex) 나 -> 1, 이 -> 2, 키 -> 3
3. `python data.py build_y_vocab b`
    - 대카테고리 기준으로 y_vocab을 생성합니다. 대카테고리 -> 57
4. `python data.py build_y_vocab m`
    - 중카테고리 기준으로 y_vocab을 생성합니다. 중카테고리 -> 552 
5. `python data.py build_y_vocab s`
    - 소카테고리 기준으로 y_vocab을 생성합니다. 소카테고리 -> 3190 
6. `python data.py build_y_vocab d`
    - 세부카테고리 기준으로 y_vocab을 생성합니다. 세부카테고리 -> 404
7. `python data.py make_db train --train_ratio=1.0`
    - 학습에 필요한 데이터셋을 생성합니다. (h5py 포맷) dev, test도 동일한 방식으로 생성할 수 있습니다.
    - 위 명령어를 수행하면 `train` 데이터의 100%를 학습합니다.
    - 이 명령어를 실행하기 전에 `python data.py build_y_vocab {d,m,s,d}`으로 데이터 생성이 필요합니다
    - Python 3은 `y_vocab.py3.cPickle.{d,m,s,d}` 파일을 사용합니다.
    - `config.json` 파일에 동시에 처리할 프로세스 수를 `num_workers`로 조절할 수 있습니다.
8. `python data.py make_db dev ./data/dev --train_ratio=0.0`
    - 학습에 필요한 dev 데이터셋을 생성합니다. (h5py 포맷) test도 동일한 방식으로 생성할 수 있습니다.
9. `python pretrain_word2vec.py train ./data/train ./model/train`
    - word2vec, char2vec을 pretrain합니다. model의 embedding layer의 초기값으로 사용됩니다.
    - 상위 디렉토리에 `word_embed_matrix.np, char_embed_matrix.np` 파일이 생성되며 inference시에는 필요하지 않습니다.
    - ./data/train/data.h5py에 dev가 없을시 ./data/dev/data.h5py의 dev를 사용합니다.
10. `python classifier.py train ./data/train ./model/train True True`
    - `./data/train`에 생성한 데이터셋으로 학습을 진행합니다.
    - 완성된 모델은 `./model/train`에 위치합니다.
    - `True`의 옵션의 경우 pretrain된 word2vec, char2vec initial 값으로 시작합니다.
11. `python classifier.py predict ./data/train ./model/train ./data/test/ test predict.tsv`
    - `test`로 생성한 평가 데이터에 대해서 예측한 결과를 `predict.tsv`에 저장합니다.

## pretrain 이용하기
0. 필요한 파일을 다운받습니다.
    - word_tfidf_200.dict : https://drive.google.com/open?id=1TWzSmqSAECebJgGc4o5yI3ph5e70--p7
    - char_tfidf_4830.dict : https://drive.google.com/open?id=10AtKAhXq3Bz2mj8NQQ4agSaKZgFMgSJq
    - final_model : https://drive.google.com/open?id=1xxfjWgEQHZb2PcDSPhWWnxzAP5kXHF0p
    
1. 위 파일들이 다운받으면 실행방법 중 (2, 3, 10, 11) 과정을 생략할수 있습니다.
2. `python classifier.py predict ./data/train ./model/final_model ./data/test/ test predict.tsv`


## 제출하기
0. 제출
    - baseline.predict.tsv 파일을 zip으로 압축한 후 카카오 아레나 홈페이지에 제출합니다.


## 로직 설명
카테고리를 계층 구분하여 "대>-1>-1>-1", "-1>중>-1>-1", "-1>-1>소>-1" ,"-1>-1>-1>세"로 표현하여 데이터를 구성했습니다. 하나의 모델로 shared_layer와 각각의 카테고리 계층에 대한 output이 존재하며 예측 뒤 그 결과를 "대>중>소>세"로  조합합니다. word단위의 embedding과 char단위의 embedding 그리고 resnet feature를 input으로 받습니다.  

0. Shared layer
	- word2vec과 char2vec을 학습하여 intitial값으로 사용합니다. 
	- word의 경우 embedding layer만을 share합니다.
	- char의 경우 bidirectional-lstm이후 cnn-ngram-block까지를 share합니다.

1. Multioutput layer
	- img 각 카테고리마다 dense_layer로 구성됩니다.
	- word의경우 각 카테고리마다 2개의 attention layer로 구성됩니다.
	- char의경우 각 카테고리마다 ngram cnn chanel이후 attention layer로 구성됩니다.
	- 위 3개의 layer들은 concat되고 각 카테고리 아웃풋은 fully-connected layer로 구성됩니다.


## Open Source license
kakao baseline이외에 추가적인 library
	
	- pip install keras-self-attention # MIT License (MIT)
	- pip install pandas # BSD 3-Clause License
	- pip install scikit-learn # BSD license

위의 3 library가 추가적으로 필요합니다. (python 2.7 설치시 기본 lib제외)
## Model Size
inference 기준(학습시 중간에 생성되었다가 삭제가능은 제외)

	- word indexing: word_tfidf_200.dict (9.7M)
	- char indexing: char_tfidf_4830.dict (207K)
	- Model: ./model/train/ (473M)

## 기타
- 코드 베이스를 실행하기 위해서는 데이터셋을 포함해 최소 450G 가량의 공간이 필요합니다.

## 테스트 가이드라인
학습데이터의 크기가 100GB 이상이므로 사용하는 장비에 따라서 설정 변경이 필요합니다. `config.json`에서 수정 가능한 설정 중에서 아래 항목들이 장비의 사양에 민감하게 영향을 받습니다.

    - train_data_list
    - chunk_size
    - num_workers
    - num_predict_workers


`train_data_list`는 학습에 사용할 데이터 목록입니다. 전체 9개의 파일이며, 만약 9개의 파일을 모두 사용하여 학습하기 어려운 경우는 이 파일 수를 줄일 경우 시간을 상당히 단축시킬 수 있습니다. 

`chunk_size`는 전처리 단계에서 저장하는 중간 파일의 사이즈에 영향을 줍니다. Out of Memory와 같은 에러가 날 경우 이 옵션을 줄일 경우 해소될 수 있습니다.

`num_workers`는 전처리 수행 시간과 관련이 있습니다. 장비의 코어수에 적합하게 수정하면 수행시간을 줄이는데 도움이 됩니다.

`num_predict_workers`는 예측 수행 시간과 관련이 있습니다. `num_workers`와 마찬가지로 장비의 코어수에 맞춰 적절히 수정하면 수행시간을 단축하는데 도움이 됩니다.


### Benchmark

GPU vram 12G가 필요합니다. OOM이슈 있음

## 라이선스

This software is licensed under the Apache 2 license, quoted below.

Copyright 2018 Kakao Corp. http://www.kakaocorp.com

Licensed under the Apache License, Version 2.0 (the “License”); you may not use this project except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
