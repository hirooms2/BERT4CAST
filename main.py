import os
from torch import nn, optim
from torch.utils.data import DataLoader

import torch

from data_loader import DatasetTrain, DatasetTest
from model import Model
from parameters import parse_args
from preprocess import read_news, get_doc_input, save_news, load_news, glove
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import logging
from tqdm.auto import tqdm

from utils import scoring


def train(args, model, train_dataloader, dev_dataloader):
    # Only support title Turing now

    logging.info('Training...')
    if not os.path.exists('./model'):
        os.mkdir('./model')

    for ep in range(args.epoch):
        total_loss = 0.0
        for (user_features, log_mask, news_features, label) in tqdm(train_dataloader):
            loss, _ = model(user_features, log_mask, news_features, label)
            total_loss += loss.data.float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss /= len(train_dataloader)
        print(ep + 1, total_loss)

        best_auc, best_epoch = 0, 0

        aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []

        with torch.no_grad():
            for (user_features, log_mask, news_features, label) in tqdm(dev_dataloader):
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

        print('Epoch %d : dev done\nDev criterions' % (ep + 1))
        print('AUC = {:.4f}\tMRR = {:.4f}\tnDCG@5 = {:.4f}\tnDCG@10 = {:.4f}'.format(auc, mrr, ndcg5, ndcg10))

        if best_auc < auc:
            best_auc = auc
            best_epoch = ep
            print('save the model')
            torch.save({model.name: model.state_dict()}, './model/' + model.name)

        print('Best Epoch:\t%f\tBest auc:\t%f' % (best_epoch, best_auc))


def test(args, model, test_dataloader):
    print('test mode start')
    test_model_path = './model/' + model.name
    model.load_state_dict(torch.load(test_model_path, map_location=torch.device('cpu'))[model.name])
    model.cuda(args.device_id)

    if not os.path.exists('./results'):
        os.mkdir('./results')
    result_file = './results/prediction.txt'
    results = [[] for _ in range(len(test_dataloader))]

    with torch.no_grad():
        for idx, (log_ids, log_mask, input_ids, labels) in enumerate(tqdm(test_dataloader)):
            scores = model(input_ids, log_ids, log_mask, labels, compute_loss=False)
            scores = scores.view(-1).cpu().numpy()
            sub_scores = []

            for e, val in enumerate(scores):
                sub_scores.append([val, e])
            sub_scores.sort(key=lambda x: x[0], reverse=True)
            results[idx] = [0 for _ in range(len(sub_scores))]
            for j in range(len(sub_scores)):
                results[idx][sub_scores[j][1]] = j + 1

    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, result in enumerate(results):
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))


def print_num_param(model):
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'total:\t{pytorch_total_params:,}\ttrainable:\t{pytorch_total_trainable_params:,}')


if __name__ == '__main__':
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    word_dict = tokenizer.get_vocab()
    bert_config = AutoConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    bert_model = AutoModel.from_pretrained("bert-base-uncased", config=bert_config)

    if args.n_layer > 2:
        modules = [bert_model.embeddings, bert_model.encoder.layer[:args.n_layer - 2]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    data_path = os.path.join('./datasets/', args.dataset)
    text_path = os.path.join(data_path, 'text')
    if not os.path.exists(text_path):
        os.mkdir(text_path)

    word_embedding_path = os.path.join('./datasets', f'glove_d{args.word_embedding_dim}.pkl')

    if not os.path.exists(word_embedding_path):
        glove(word_dict, args.word_embedding_dim, word_embedding_path)

    news_file = f'news_t{args.max_title_len}_b{args.max_body_len}.txt'
    if not os.path.exists(os.path.join(text_path, news_file)):
        news, news_index, category_dict, subcategory_dict = read_news(data_path, args, tokenizer)
        save_news(news_file, text_path, news, news_index, category_dict, subcategory_dict)
    else:
        news, news_index, category_dict, subcategory_dict = load_news(news_file, text_path)

    news_combined = get_doc_input(news, news_index, category_dict, subcategory_dict, args)

    model = Model(args, bert_model, word_embedding_path)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
        test(args, model, test_dataloader)
