import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# gpuConfig = tf.ConfigProto()
# gpuConfig.allow_soft_placement = config.getboolean('gpu', 'allow_soft_placement')
# gpuConfig.gpu_options.allow_growth = config.getboolean('gpu', 'allow_growth')

import pandas as pd
import csv
pos_path = "Data/jd_xiaomi9_pos.csv"
neg_path = "Data/jd_xiaomi9_neg.csv"
pos_file = open(pos_path)
neg_file = open(neg_path)
pos_reader_lines = csv.reader(pos_file)
neg_reader_lines = csv.reader(neg_file)
# print(pos_reader_lines)
# print(neg_reader_lines)


# 现在我们将所有的评价内容放置到一个list里
train_texts_orig = []

# 文本所对应的labels, 也就是标记
train_target = []
for line in pos_reader_lines:
    #print(line)
    if line[0] != '':
        train_texts_orig.append(line[1])
        train_target.append('1')
for line in neg_reader_lines:
    #print(line)
    if line[0] != '':
        train_texts_orig.append(line[1])
        train_target.append('0')

# 首先加载必用的库
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import re
import jieba # 结巴分词
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")

# 用来解压
import bz2

# 请将下载的词向量压缩包放置在根目录 embeddings 文件夹里
# 解压词向量, 有可能需要等待1-2分钟

with open("Data/sgns.zhihu.bigram", 'wb') as new_file, open("Data/sgns.zhihu.bigram.bz2", 'rb') as file:
    decompressor = bz2.BZ2Decompressor()
    for data in iter(lambda : file.read(100 * 1024), b''):
        new_file.write(decompressor.decompress(data))

# 使用gensim加载预训练中文分词embedding, 有可能需要等待1-2分钟
cn_model = KeyedVectors.load_word2vec_format('Data/sgns.zhihu.bigram', binary=False, unicode_errors="ignore")

# 由此可见每一个词都对应一个长度为300的向量
embedding_dim = cn_model['深圳'].shape[0]
#print('词向量的长度为{}'.format(embedding_dim))
#print(cn_model['深圳'])


# 我们使用tensorflow的keras接口来建模
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# 进行分词和tokenize
# train_tokens是一个长长的list，其中含有4000个小list，对应每一条评价
train_tokens = []
for text in train_texts_orig:
    # 去掉标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    # 结巴分词
    cut = jieba.cut(text)
    # 结巴分词的输出结果为一个生成器
    # 把生成器转换为list
    cut_list = [ i for i in cut ]
    for i, word in enumerate(cut_list):
        try:
            # 将词转换为索引index
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            # 如果词不在字典中，则输出0
            cut_list[i] = 0
    train_tokens.append(cut_list)

num_tokens = [len(token) for token in train_tokens]
num_tokens = np.array(num_tokens)
#print(np.mean(num_tokens))

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
#print(max_tokens)

# 取tokens的长度为98时，大约95%的样本被涵盖
# 我们对长度不足的进行padding，超长的进行修剪
#print(np.sum( num_tokens < max_tokens ) / len(num_tokens))


# 用来将tokens转换为文本
def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_model.index2word[i]
        else:
            text = text + ' '
    return text
reverse = reverse_tokens(train_tokens[1000])
#print(reverse)
#print(embedding_dim)

# 只使用前20000个词
num_words = 60000

# 初始化embedding_matrix，之后在keras上进行应用
embedding_matrix = np.zeros((num_words, embedding_dim))

# embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
# 维度为 50000 * 300
for i in range(num_words):
    embedding_matrix[i,:] = cn_model[cn_model.index2word[i]]
embedding_matrix = embedding_matrix.astype('float32')

# 检查index是否对应，
# 输出300意义为长度为300的embedding向量一一对应
#print(np.sum(cn_model[cn_model.index2word[333]] == embedding_matrix[333] ))


# embedding_matrix的维度，
# 这个维度为keras的要求，后续会在模型中用到
#print(embedding_matrix.shape)

# 进行padding和truncating， 输入的train_tokens是一个list
# 返回的train_pad是一个numpy array
train_pad = pad_sequences(train_tokens, maxlen=max_tokens, padding='pre', truncating='pre')

# 超出五万个词向量的词用0代替
train_pad[ train_pad>=num_words ] = 0

# 可见padding之后前面的tokens全变成0，文本在最后面
#print(train_pad[31])

train_target = np.array(train_target)
#print(train_target)
train_target = train_target.astype('int')


# 进行训练和测试样本的分割
from sklearn.model_selection import train_test_split

# 90%的样本用来训练，剩余10%用来测试
X_train, X_test, y_train, y_test = train_test_split(train_pad,
                                                    train_target,
                                                    test_size=0.1,
                                                    random_state=12)

# # 查看训练样本，确认无误
# print(reverse_tokens(X_train[10]))
# print('class: ',y_train[10])

model = Sequential()
model.add(Embedding(num_words,
                   embedding_dim,
                   weights=[embedding_matrix],
                   input_length = max_tokens,
                   trainable = False))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(LSTM(units=16, return_sequences=False))

model.add(Dense(1, activation='sigmoid'))
# 我们使用adam以0.001的learning rate进行优化
optimizer = Adam(lr=1e-3)

#model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# # 建立一个权重的存储点
# path_checkpoint = 'sentiment_checkpoint.keras'
# checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
#                                       verbose=1, save_weights_only=True,
#                                       save_best_only=True)
#
# # 尝试加载已训练模型
# try:
#     model.load_weights(path_checkpoint)
# except Exception as e:
#     print(e)

# 定义early stoping如果3个epoch内validation loss没有改善则停止训练
earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# 自动降低learning rate
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-8, patience=0, verbose=1)

# 定义callback函数
callbacks = [
    earlystopping,
    # checkpoint,
    lr_reduction
]

# 开始训练
model.fit(X_train, y_train,
          validation_split=0.1,
          epochs=10,
          batch_size=256,
          callbacks=callbacks)

result = model.evaluate(X_test, y_test)
print('Accuracy:{0:.2%}'.format(result[1]))

def predict_sentiment(text):
    print(text)
    # 去标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    # 分词
    cut = jieba.cut(text)
    cut_list = [ i for i in cut ]
    # tokenize
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
            if cut_list[i] >= 30000:
                cut_list[i] = 0
        except KeyError:
            cut_list[i] = 0
    # padding
    tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,
                           padding='pre', truncating='pre')
    # 预测
    result = model.predict(x=tokens_pad)
    #print(result)
    coef = result[0][0]
    if coef >= 0.5:
        print('是一例正面评价','output=%.2f'%coef)
    else:
        print('是一例负面评价','output=%.2f'%coef)

#predict_sentiment('品控不好，还没到一个月就坏了')

# y_pred = model.predict(X_test)
# y_pred = y_pred.T[0]
# y_pred = [1 if p>= 0.5 else 0 for p in y_pred]
# y_pred = np.array(y_pred)
# y_actual = np.array(y_test)
# misclassified = np.where( y_pred != y_actual )[0]
#
# # 输出所有错误分类的索引
# len(misclassified)
# print(len(X_test))
#
# idx=101
# print(reverse_tokens(X_test[idx]))
# print('预测的分类', y_pred[idx])
# print('实际的分类', y_actual[idx])
#
# print(misclassified)
#
# predict_sentiment('手机用起来很舒服，外观造型漂亮，很流畅，屏幕很清晰')

# jqnr = input()
# while jqnr != '' :
#     predict_sentiment(jqnr)
#     jqnr = input()