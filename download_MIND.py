import os
import requests
import zipfile
import json
import random
import shutil
import collections
import numpy as np

MIND_small_dataset_root = '../MIND-small'
MIND_large_dataset_root = '../MIND-large'


def download_extract_MIND_small():
    if not os.path.exists(MIND_small_dataset_root):
        os.mkdir(MIND_small_dataset_root)
        
    if not os.path.exists(MIND_small_dataset_root + '/download'):
        os.mkdir(MIND_small_dataset_root + '/download')
    if not os.path.exists(MIND_small_dataset_root + '/download/train'):
        if not os.path.exists(MIND_small_dataset_root + '/download/MINDsmall_train.zip'):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip -P %s/download' % MIND_small_dataset_root)
        assert os.path.exists(MIND_small_dataset_root + '/download/MINDsmall_train.zip'), 'Train set zip not found'
        os.mkdir(MIND_small_dataset_root + '/download/train')
        os.system('unzip %s/download/MINDsmall_train.zip -d %s/download/train' % (MIND_small_dataset_root, MIND_small_dataset_root))
    if not os.path.exists(MIND_small_dataset_root + '/download/dev'):
        if not os.path.exists(MIND_small_dataset_root + '/download/MINDsmall_dev.zip'):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip -P %s/download' % MIND_small_dataset_root)
        assert os.path.exists(MIND_small_dataset_root + '/download/MINDsmall_dev.zip'), 'Dev set zip not found'
        os.mkdir(MIND_small_dataset_root + '/download/dev')
        os.system('unzip %s/download/MINDsmall_dev.zip -d %s/download/dev' % (MIND_small_dataset_root, MIND_small_dataset_root))

def download_extract_MIND_large():
    if not os.path.exists(MIND_large_dataset_root):
        os.mkdir(MIND_large_dataset_root)

    if not os.path.exists(MIND_large_dataset_root + '/train'):
        os.mkdir(MIND_large_dataset_root + '/train')
    if not os.path.exists(MIND_large_dataset_root + '/dev'):
        os.mkdir(MIND_large_dataset_root + '/dev')
    if not os.path.exists(MIND_large_dataset_root + '/test'):
        os.mkdir(MIND_large_dataset_root + '/test')

    if not os.path.exists(MIND_large_dataset_root + '/download'):
        os.mkdir(MIND_large_dataset_root + '/download')
    if not os.path.exists(MIND_large_dataset_root + '/download/train'):
        if not os.path.exists(MIND_large_dataset_root + '/download/MINDlarge_train.zip'):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip -P %s/download' % MIND_large_dataset_root)
        assert os.path.exists(MIND_large_dataset_root + '/download/MINDlarge_train.zip'), 'Train set zip not found'
        os.mkdir(MIND_large_dataset_root + '/download/train')
        os.system('unzip %s/download/MINDlarge_train.zip -d %s/download/train' % (MIND_large_dataset_root, MIND_large_dataset_root))
    if not os.path.exists(MIND_large_dataset_root + '/download/dev'):
        if not os.path.exists(MIND_large_dataset_root + '/download/MINDlarge_dev.zip'):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip -P %s/download' % MIND_large_dataset_root)
        assert os.path.exists(MIND_large_dataset_root + '/download/MINDlarge_dev.zip'), 'Dev set zip not found'
        os.mkdir(MIND_large_dataset_root + '/download/dev')
        os.system('unzip %s/download/MINDlarge_dev.zip -d %s/download/dev' % (MIND_large_dataset_root, MIND_large_dataset_root))
    if not os.path.exists(MIND_large_dataset_root + '/download/test'):
        if not os.path.exists(MIND_large_dataset_root + '/download/MINDlarge_test.zip'):
            os.system('wget https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip -P %s/download' % MIND_large_dataset_root)
        assert os.path.exists(MIND_large_dataset_root + '/download/MINDlarge_test.zip'), 'Test set zip not found'
        os.mkdir(MIND_large_dataset_root + '/download/test')
        os.system('unzip %s/download/MINDlarge_test.zip -d %s/download/test' % (MIND_large_dataset_root, MIND_large_dataset_root))

    for data in ['train', 'dev', 'test']:
        shutil.copyfile('../MIND-large/download/%s/news.tsv' % data, '../MIND-large/%s/news.tsv' % data)
        shutil.copyfile('../MIND-large/download/%s/behaviors.tsv' % data, '../MIND-large/%s/behaviors.tsv' % data)


def split_training_behaviors():
    MIND_small_train_ratio = 0.95
    train_behavior_lines = []
    dev_behavior_lines = []
    behavior_lines = []
    with open(MIND_small_dataset_root + '/download/train/behaviors.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 0:
                behavior_lines.append(line)
    random.shuffle(behavior_lines)

    behavior_num = len(behavior_lines)
    behavior_id = [i for i in range(behavior_num)]
    random.shuffle(behavior_id)
    train_num = int(behavior_num * MIND_small_train_ratio)
    train_behavior_id = random.sample(behavior_id, train_num)
    train_behavior_id = set(train_behavior_id)
    for i, line in enumerate(behavior_lines):
        if i in train_behavior_id:
            train_behavior_lines.append(line)
        else:
            dev_behavior_lines.append(line)
    return train_behavior_lines, dev_behavior_lines


def preprocess_MIND_small():
    train_behavior_lines, dev_behavior_lines = split_training_behaviors()

    # train set
    train_set_root = MIND_small_dataset_root + '/train'
    if not os.path.exists(train_set_root):
        os.mkdir(train_set_root)
    with open(train_set_root + '/behaviors.tsv', 'w', encoding='utf-8') as f:
        for line in train_behavior_lines:
            f.write(line)
    if not os.path.exists(train_set_root + '/news.tsv'):
        shutil.copyfile(MIND_small_dataset_root + '/download/train/news.tsv', train_set_root + '/news.tsv')

    # dev set
    dev_set_root = MIND_small_dataset_root + '/dev'
    if not os.path.exists(dev_set_root):
        os.mkdir(dev_set_root)
    with open(dev_set_root + '/behaviors.tsv', 'w', encoding='utf-8') as f:
        for line in dev_behavior_lines:
            f.write(line)
    if not os.path.exists(dev_set_root + '/news.tsv'):
        shutil.copyfile(MIND_small_dataset_root + '/download/train/news.tsv', dev_set_root + '/news.tsv')

    # test set
    test_set_root = MIND_small_dataset_root + '/test'
    if not os.path.exists(test_set_root):
        os.mkdir(test_set_root)
    if not os.path.exists(test_set_root + '/behaviors.tsv'):
        shutil.copyfile(MIND_small_dataset_root + '/download/dev/behaviors.tsv', test_set_root + '/behaviors.tsv')
    if not os.path.exists(test_set_root + '/news.tsv'):
        shutil.copyfile(MIND_small_dataset_root + '/download/dev/news.tsv', test_set_root + '/news.tsv')

if __name__ == '__main__':
    download_extract_MIND_small()
    preprocess_MIND_small()
    download_extract_MIND_large()