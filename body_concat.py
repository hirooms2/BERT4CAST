import os
import requests
import zipfile
import json
import random
import shutil
import collections
import numpy as np

MIND_small_dataset_root = '../MIND_small'
MIND_large_dataset_root = './datasets/MIND/large'


def body_concat(dataset_root):
    for data in ['train', 'dev', 'test']:
        with open(dataset_root + '/%s/msn.json' % data, encoding="utf8") as f:
            lines = f.readlines()
            
        news_body = dict()
        for i in lines:
            try:
                tmp = eval(i)
            except:
                continue
            if type(tmp) == tuple:
                tmp = tmp[0]
            news_body[tmp['nid']] = ' '.join(' '.join(tmp['body']).split()).encode('utf8', 'ignore').decode('utf8')

        with open(dataset_root + '/%s/news.tsv' % data, 'r', encoding='utf-8') as f:
            with open(dataset_root + '/%s/news_with_body.tsv' % data, 'w', encoding='utf-8') as body_f:
                for line in f:
                    news_ID, category, subCategory, title, abstract, url, title_entities, abstract_entities = line.strip().split('\t')
                    line = line.strip() + '\t' + news_body[url.split('/')[-1].split('.')[0]] + '\n'
                    body_f.write(line)

if __name__ == '__main__':
    # body_concat(MIND_small_dataset_root)
    body_concat(MIND_large_dataset_root)