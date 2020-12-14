from urllib import request, parse
import json
import numpy as np
import re
import jieba
from gensim.models import KeyedVectors
import bz2
from itertools import chain

def get_dict_Ch(tokens):
    words = []
    ch = re.compile(r'[\u4e00-\u9fa5]+')
    for i in tokens:
        for j in i:
            if ch.findall(j) != []:
                words.append(ch.findall(j)[0])
    return words

def get_trans(words):
    trans = []
    for i in words:
        trans.append(translate(i))
    return trans

def get_cn_model():
    with open("weights/sgns.zhihu.bigram", 'wb') as new_file, open("weights/sgns.zhihu.bigram.bz2", 'rb') as file:
        decompressor = bz2.BZ2Decompressor()
        for data in iter(lambda: file.read(100 * 1024), b''):
            new_file.write(decompressor.decompress(data))

    cn_model = KeyedVectors.load_word2vec_format('Data/sgns.zhihu.bigram', binary=False, unicode_errors="ignore")
    return cn_model

def vector(tokens,cn_model,en_model,w):
    vectors = []
    ch = re.compile(r'[\u4e00-\u9fa5]+')
    en = re.compile('[a-zA-Z]+')
    for i in tokens:
        for j in i:
            if ch.findall(j) != []:
                vectors.append(np.multiply(w, cn_model[i]))
            if en.findall(j) != []:
                vectors.append(en_model[i])
    return vectors

def tokenizer(texts):
    tokens = []
    for text in texts:
        text = re.sub("[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）()]+", "", text)
        cut = jieba.cut(text)
        cut_list = []
        for i in cut:
            if i != ' ':
                cut_list.append(i)
        tokens.append(cut_list)
    return list(chain(*tokens))

def translate(content):
    req_url = 'http://fanyi.youdao.com/translate'
    head_data = {}
    head_data['Referer'] = 'http://fanyi.youdao.com/'
    head_data[
        'User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36X-Requested-With: XMLHttpRequest'
    form_data = {}
    form_data['i'] = content
    form_data['doctype'] = 'json'
    data = parse.urlencode(form_data).encode('utf-8')
    req = request.Request(req_url, data, head_data)
    # try:
    response = request.urlopen(req)
    html = response.read().decode('utf-8')
    translate_results = json.loads(html)
    print(translate_results)
    translate_results = translate_results['translateResult'][0][0]['tgt']
    return (str(translate_results))
    # except:
    #     return np.NaN

def transform_w(words,cn_model,en_model):
    w = cn_model['你好']
    length = len(words)
    for i in words:
        if ' ' in i:
            length -= 1
            for j in i.split():
                try:
                    w += np.multiply(en_model[translate(j)], np.linalg.inv(cn_model[j]))
                    length += 1
                except(KeyError):pass
        else:
            try:
                w += np.multiply(en_model[translate(i)], np.linalg.inv(cn_model[i]))
            except(KeyError):pass
    w = w / length
    return w

