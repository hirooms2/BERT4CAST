import logging
import os
from datetime import datetime

import numpy as np
import torch
from pytz import timezone
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from data_loader import DatasetTest, DatasetTrain
from model import Model
from parameters import parse_args
from preprocess import get_doc_input, glove, load_news, read_news, save_news
from utils import scoring


##TODO : GIT Contributor test

def get_time_kst(): return datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')


def save_best_results(path, args, best_epoch, best_auc, best_mrr, best_ndcg5, best_ndcg10):
    with open(path, 'a', encoding='utf-8') as b_result_f:
        for i, v in vars(args).items():
            b_result_f.write(f'{i}:{v} || ')
        b_result_f.write(
            '\nBEST_SCORE : epoch: {:.0f}\tAUC = {:.4f}\tMRR = {:.4f}\tnDCG@5 = {:.4f}\tnDCG@10 = {:.4f}\n'.format(
                best_epoch, best_auc, best_mrr, best_ndcg5, best_ndcg10))
        b_result_f.write(f'THE END : {get_time_kst()} \n')


def train(args, model, train_dataloader, dev_dataloader):
    # Only support title Turing now
    logging.info('Training...')
    if not os.path.exists('./model'): os.mkdir('./model')

    # results
    if not os.path.exists('./results'): os.mkdir('./results')
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M'))  # MonthDailyHourMinute .....e.g., 05091040
    results_file_path = f"./results/train_device_{args.device_id}_{mdhm}_{args.name}.txt"
    # results_file_path = './results/train_device_%d.txt' % args.device_id

    # Only For Best Result
    best_results_file_path = './results/train_best.txt'  # only Best result 파일

    # parameters
    with open(results_file_path, 'a', encoding='utf-8') as result_f:
        result_f.write(
            '\n=================================================\n==================== train =====================\n')
        result_f.write(get_time_kst())
        result_f.write('\n')
        for i, v in vars(args).items():
            result_f.write(f'{i}:{v} || ')
        result_f.write('\n')

    best_auc, best_epoch = 0, 0
    best_mrr, best_ndcg5, best_ndcg10 = 0, 0, 0

    for ep in range(args.epoch):
        # if ep<2:
        #     args.reg_term=1 # LM 만 학습
        # else:
        #     args.reg_term=0 # CTR 만 학습
        model.train()
        total_loss, total_loss_lm = 0.0, 0.0
        for (user_features, log_mask, news_features, label) in tqdm(train_dataloader,
                                                                    bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            loss = model(user_features, log_mask, news_features, label)
            total_loss += loss.data.float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        print('Loss:\t%.4f' % total_loss)
        print('Loss_LM:\t%.4f' % total_loss_lm)

        # best_auc, best_epoch = 0, 0
        # best_mrr, best_ndcg5, best_ngcg10 = 0, 0, 0
        model.eval()
        aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
        hits = []
        with torch.no_grad():
            for (user_features, log_mask, news_features, label) in tqdm(dev_dataloader,
                                                                        bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
                scores = model(user_features, log_mask, news_features, label, compute_loss=False)

                scores = scores.view(-1).cpu().numpy()
                sub_scores = []
                for e, val in enumerate(scores):
                    sub_scores.append([val, e])
                sub_scores.sort(key=lambda x: x[0], reverse=True)
                result = [0 for _ in range(len(sub_scores))]
                for j in range(len(sub_scores)):
                    result[sub_scores[j][1]] = j + 1

                label = label.view(-1).cpu().numpy()
                auc, mrr, ndcg5, ndcg10 = scoring(label, result)
                aucs.append(auc)
                mrrs.append(mrr)
                ndcg5s.append(ndcg5)
                ndcg10s.append(ndcg10)

        auc = np.mean(aucs)
        mrr = np.mean(mrrs)
        ndcg5 = np.mean(ndcg5s)
        ndcg10 = np.mean(ndcg10s)
        # hit = np.mean(hits)

        print('Epoch %d : dev done\nDev criterions' % (ep + 1))
        print(
            'AUC = {:.4f}\tMRR = {:.4f}\tnDCG@5 = {:.4f}\tnDCG@10 = {:.4f}'.format(auc, mrr, ndcg5, ndcg10))

        # result 파일에 기록 추가
        with open(results_file_path, 'a', encoding='utf-8') as result_f:
            if ep == 0:
                device_ep0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                result_f.write('Using device: {device_ep0}\t')
                # Additional Info when using cuda
                if device_ep0.type == 'cuda':
                    result_f.write(torch.cuda.get_device_name(0))
                    result_f.write('\n <Memory Usage> \n')
                    result_f.write(f'Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB\t||\t')
                    result_f.write(f'Cached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB\n')
            result_f.write('Epoch %d : dev done \t Dev criterions \t' % (ep + 1))
            # LM Loss 기록
            result_f.write(
                'AUC\t{:.4f}\tMRR:\t{:.4f}\tnDCG@5\t{:.4f}\tnDCG@10\t{:.4f}\t'.format(auc, mrr, ndcg5, ndcg10))
            result_f.write(get_time_kst())
            result_f.write('\n')

        if best_auc < auc:
            best_auc = auc
            best_epoch = ep
            best_mrr = mrr
            best_ndcg5 = ndcg5
            best_ndcg10 = ndcg10

            print('save the model')
            # torch.save({model.name: model.state_dict()}, './model/' + model.name) # original save
            torch.save({model.name: model.state_dict()}, f'./model/{mdhm}_{model.name}')  # TIME_MODELNAME 형식

        print('Best Epoch:\t%f\tBest auc:\t%f' % (best_epoch, best_auc))

    with open(results_file_path, 'a', encoding='utf-8') as result_f:
        # result_f.write(f'\nBEST_SCORE epoch: {int(best_epoch)}\tAUC: {best_auc}\tMRR: {best_mrr}\tNDCG@5: {best_ndcg5}\tNDCG@10: {best_ndcg10}\n')
        result_f.write(
            '\nBEST_SCORE : epoch: {:.0f}\tAUC = {:.4f}\tMRR = {:.4f}\tnDCG@5 = {:.4f}\tnDCG@10 = {:.4f}\t'.format(
                best_epoch, best_auc, best_mrr, best_ndcg5, best_ndcg10))
        result_f.write(f'THE END : {get_time_kst()} \n')
    save_best_results(best_results_file_path, args, best_epoch, best_auc, best_mrr, best_ndcg5, best_ndcg10)


def test(args, model, test_dataloader, tokenizer):
    print('test mode start')
    # test_model_path = './model/' + model.name + str(args.reg_term)
    temp_model_name = '2stage_body_nlayer2'
    test_model_path = './model/2stage_body_nlayer20'  # debug hard coding--1
    # model.load_state_dict(torch.load(test_model_path, map_location=torch.device('cpu'))[model.name]) # debug hard coding--2
    model.load_state_dict(torch.load(test_model_path, map_location=torch.device('cpu'))[temp_model_name])
    model.cuda(args.device_id)

    if not os.path.exists('./results'):
        os.mkdir('./results')
    result_file = './results/prediction-%.1f.txt' % args.reg_term
    results = [[] for _ in range(len(test_dataloader))]

    result_lm_file = './results/lm-%.1f.txt' % args.reg_term
    results_lm = []
    hit = []
    with torch.no_grad():
        for idx, (user_features, log_mask, news_features, label) in enumerate(
                tqdm(test_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}')):
            scores, mlm = model(user_features, log_mask, news_features, label, compute_loss=False)
            score_lm, masked_index = mlm

            scores = scores.view(-1).cpu().numpy()

            sub_scores = []

            for e, val in enumerate(scores):
                sub_scores.append([val, e])
            sub_scores.sort(key=lambda x: x[0], reverse=True)
            results[idx] = [0 for _ in range(len(sub_scores))]
            for j in range(len(sub_scores)):
                results[idx][sub_scores[j][1]] = j + 1

            # for Analyzing Language Model
            title_text = news_features[0].squeeze(0).cpu().numpy()
            masked_index = masked_index.cpu().numpy()

            title_mask = news_features[1].squeeze(0)
            title_lens = torch.sum(title_mask, dim=1)
            title_lens = title_lens.cpu().numpy()

            sub_scores = score_lm.topk(10)[1]
            sub_scores = sub_scores.cpu().numpy()

            for (title, midx, s_score, l) in zip(title_text, masked_index, sub_scores, title_lens):
                hit.append(np.isin(title[midx], s_score))
                text = tokenizer.convert_ids_to_tokens(title[:l])
                target_word = text[midx]
                predicted_word = tokenizer.decode(s_score, skip_special_tokens=True)
                result_str = "%s\t[%s]\t%s\n" % (' '.join(text), target_word, predicted_word)
                results_lm.append(result_str)

    print('LM_HIT@10:\t%f' % np.mean(hit))
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, result in enumerate(results):
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))

    with open(result_lm_file, 'w', encoding='utf-8') as result_f:
        for i, result in enumerate(results_lm):
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + '.\t' + result)


def print_num_param(model):
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'total:\t{pytorch_total_params:,}\ttrainable:\t{pytorch_total_trainable_params:,}')


if __name__ == '__main__':
    args = parse_args()
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

    # albert-base-v2
    # bert-base-uncased
    # roberta-base
    # 'textattack/bert-base-uncased-ag-news'
    bert_name = args.bert_name
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    word_dict = tokenizer.get_vocab()
    # bert_config = AutoConfig.from_pretrained(bert_name, output_hidden_states=True)
    bert_config = AutoConfig.from_pretrained(bert_name)
    bert_model = AutoModel.from_pretrained(bert_name, config=bert_config)

    modules = [bert_model.embeddings, bert_model.encoder.layer[:bert_config.num_hidden_layers - args.n_layer]]  # 2개 남기기
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

    data_path = os.path.join('./datasets/', args.dataset)
    text_path = os.path.join(data_path, 'text')
    if not os.path.exists(text_path):
        os.mkdir(text_path)

    word_embedding_path = os.path.join('./datasets', f'glove_d{args.word_embedding_dim}.pkl')
    word_embedding_path = os.path.join('./datasets', f'glove_d{args.glove_dim}.pkl')
    if not os.path.exists(word_embedding_path):
        glove(word_dict, args.word_embedding_dim, word_embedding_path)
        # glove(word_dict, args.glove_dim, word_embedding_path)

    news_file = f'news_t{args.max_title_len}_b{args.max_body_len}_{args.body_type}.txt'
    if not os.path.exists(os.path.join(text_path, news_file)):
        news, news_index, category_dict, subcategory_dict = read_news(data_path, args, tokenizer)
        save_news(news_file, text_path, news, news_index, category_dict, subcategory_dict)
    else:
        news, news_index, category_dict, subcategory_dict = load_news(news_file, text_path)

    args.category_num = len(category_dict) + 1
    args.subcategory_num = len(subcategory_dict) + 1
    args.vocab_size = len(word_dict)

    news_combined = get_doc_input(news, news_index, category_dict, subcategory_dict, args)

    model = Model(args, bert_model, tokenizer, word_embedding_path)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    train_dataset = DatasetTrain(
        news_index=news_index,
        news_combined=news_combined,
        word_dict=word_dict,
        data_dir=os.path.join(data_path, 'train'),
        args=args
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    model = model.cuda(args.device_id)

    if 'train' in args.mode:
        dev_dataset = DatasetTest(
            news_index=news_index,
            news_combined=news_combined,
            word_dict=word_dict,
            data_dir=os.path.join(data_path, 'dev'),
            args=args
        )
        dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
        train(args, model, train_dataloader, dev_dataloader)
    if 'test' in args.mode:
        test_dataset = DatasetTest(
            news_index=news_index,
            news_combined=news_combined,
            word_dict=word_dict,
            data_dir=os.path.join(data_path, 'test'),
            args=args,
            mode='test'
        )
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test(args, model, test_dataloader, tokenizer)
