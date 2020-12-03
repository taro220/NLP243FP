from urllib import request, parse
import json
import numpy as np

def translate(content):
    req_url = 'http://fanyi.youdao.com/translate'
    head_data = {}
    head_data['Referer'] = 'http://fanyi.youdao.com/'
    head_data[
        'User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36X-Requested-With: XMLHttpRequest'
    form_data = {}
    form_data['i'] = content
    form_data['doctype'] = 'json'
    data = parse.urlencode(form_data).encode('utf-8')
    req = request.Request(req_url, data, head_data)
    response = request.urlopen(req)
    html = response.read().decode('utf-8')
    translate_results = json.loads(html)
    translate_results = translate_results['translateResult'][0][0]['tgt']
    return (translate_results)

def transform(words,cn_model,en_model):
    w = cn_model['你好']
    for i in words:
        w += np.multiply(en_model[translate(i)], np.linalg.inv(cn_model[i]))
    w = w / len(words)
    return w
