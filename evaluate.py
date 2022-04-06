#!/usr/bin/env python
import sys, os, os.path
import numpy as np
import json
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm


def evaluation(model, test_dataloader):
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    results = [[] for _ in range(len(test_dataloader))]

    with torch.no_grad():
        for idx, (user_features, log_mask, news_features, label) in enumerate(tqdm(test_dataloader)):
            scores = model(user_features, log_mask, news_features, label, compute_loss=False)
            scores = scores.view(-1).cpu().numpy()
            sub_scores = []
            for e, val in enumerate(scores):
                sub_scores.append([val, e])
            sub_scores.sort(key=lambda x: x[0], reverse=True)
            # result = [0 for _ in range(len(sub_scores))]
            for j in range(len(sub_scores)):
                results[idx][sub_scores[j][1]] = j + 1

            label = label.view(-1).cpu().numpy()
            auc, mrr, ndcg5, ndcg10 = scoring(label, results[idx])
            aucs.append(auc)
            mrrs.append(mrr)
            ndcg5s.append(ndcg5)
            ndcg10s.append(ndcg10)
    auc = np.mean(aucs)
    mrr = np.mean(mrrs)
    ndcg5 = np.mean(ndcg5s)
    ndcg10 = np.mean(ndcg10s)

    return (auc, mrr, ndcg5, ndcg10), results


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def parse_line(l):
    impid, ranks = l.strip('\n').split()
    ranks = json.loads(ranks)
    return impid, ranks


def scoring(truth_f, sub_f):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    line_index = 1
    for lt in truth_f:
        ls = sub_f.readline()
        impid, labels = parse_line(lt)

        # ignore masked impressions
        if labels == []:
            continue

        if ls == '':
            # empty line: filled with 0 ranks
            sub_impid = impid
            sub_ranks = [1] * len(labels)
        else:
            try:
                sub_impid, sub_ranks = parse_line(ls)
            except:
                raise ValueError("line-{}: Invalid Input Format!".format(line_index))

        if sub_impid != impid:
            raise ValueError("line-{}: Inconsistent Impression Id {} and {}".format(
                line_index,
                sub_impid,
                impid
            ))

        lt_len = float(len(labels))

        y_true = np.array(labels, dtype='float32')
        y_score = []
        for rank in sub_ranks:
            score_rslt = 1. / rank
            if score_rslt < 0 or score_rslt > 1:
                raise ValueError("Line-{}: score_rslt should be int from 0 to {}".format(
                    line_index,
                    lt_len
                ))
            y_score.append(score_rslt)

        auc = roc_auc_score(y_true, y_score)
        mrr = mrr_score(y_true, y_score)
        ndcg5 = ndcg_score(y_true, y_score, 5)
        ndcg10 = ndcg_score(y_true, y_score, 10)

        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)

        line_index += 1

    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)
