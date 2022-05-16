import os
import random
import torch
from torch.utils.data import Dataset


def news_sample(news, ratio):
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


class MindDataset(Dataset):
    def __init__(self,
                 data_dir,
                 args,
                 news_index,
                 news_combined,
                 word_dict):
        super(MindDataset, self).__init__()
        self.data_dir = data_dir
        self.device = args.device_id
        self.npratio = args.npratio
        self.user_log_length = args.max_hist_len
        self.batch_size = args.batch_size
        # data loader only cares about the config after tokenization.
        self.sampler = None
        self.epoch = -1

        self.news_combined = news_combined
        self.news_index = news_index
        self.word_dict = word_dict
        self.behaviors = self._process()

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length):
        # pad_x = x[-fix_length:] + [0] * (fix_length - len(x))
        # for reversed positional embedding
        pad_x = list(reversed(x[-fix_length:])) + [0] * (fix_length - len(x))

        mask = [1] * min(fix_length, len(x)) + [0] * max(0, fix_length - len(x))
        return pad_x, mask

    def parser_text(self, indices):
        title_text = self.news_combined['title_text'][indices]
        title_mask = self.news_combined['title_mask'][indices]
        body_text = self.news_combined['body_text'][indices]
        body_mask = self.news_combined['body_mask'][indices]
        category = self.news_combined['category'][indices]
        sub_category = self.news_combined['sub_category'][indices]

        title_text = torch.LongTensor(title_text).cuda(self.device)
        title_mask = torch.LongTensor(title_mask).cuda(self.device)
        body_text = torch.LongTensor(body_text).cuda(self.device)
        body_mask = torch.LongTensor(body_mask).cuda(self.device)
        category = torch.LongTensor(category).cuda(self.device)
        sub_category = torch.LongTensor(sub_category).cuda(self.device)

        return title_text, title_mask, body_text, body_mask, category, sub_category

    def _process(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.behaviors)


class DatasetTrain(MindDataset):
    def __init__(self,
                 data_dir,
                 args,
                 news_index,
                 news_combined,
                 word_dict):
        super().__init__(data_dir, args, news_index, news_combined, word_dict)

    def _process(self):
        examples = []

        with open(os.path.join(self.data_dir, 'behaviors.tsv'), 'r', encoding='utf-8') as behavior_file:
            for behavior_index, line in enumerate(behavior_file):
                impression_ID, user_ID, time, history, impressions = line.split('\t')

                click_impressions = []
                non_click_impressions = []
                for impression in impressions.strip().split(' '):
                    if impression[-2:] == '-1':
                        click_impressions.append(impression[:-2])
                    else:
                        non_click_impressions.append(impression[:-2])

                if len(history) != 0:
                    history = history.strip().split(' ')
                    user_history = list((history[-self.user_log_length:]))
                    for clicked_news in click_impressions:
                        examples.append(
                            [user_ID, user_history, [clicked_news], non_click_impressions])
            return examples

    def __getitem__(self, item):

        user_idx, user_history, click_impressions, non_click_impressions = self.behaviors[item]

        click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(user_history),
                                                   self.user_log_length)

        sess_pos = self.trans_to_nindex(click_impressions)
        sess_neg = self.trans_to_nindex(non_click_impressions)

        if len(sess_pos) > 0:
            sess_pos = random.choices(sess_pos, k=1)

        if len(sess_neg) > 0:
            neg_index = news_sample(list(range(len(sess_neg))), self.npratio)
            sam_negs = [sess_neg[i] for i in neg_index]
        else:
            sam_negs = [0] * self.npratio
        sample_news = sess_pos + sam_negs

        user_features = self.parser_text(click_docs)
        news_features = self.parser_text(sample_news)

        log_mask = torch.FloatTensor(log_mask).cuda(self.device)
        label = torch.tensor(0).cuda(self.device)
        # label = torch.zeros(1)
        return user_features, log_mask, news_features, label


class DatasetTest(MindDataset):
    def __init__(self,
                 data_dir,
                 args,
                 news_index,
                 news_combined,
                 word_dict,
                 mode='dev'):
        super().__init__(data_dir, args, news_index, news_combined, word_dict)
        self.mode = mode

    def _process(self):
        examples = []

        with open(os.path.join(self.data_dir, 'behaviors.tsv'), 'r', encoding='utf-8') as behavior_file:
            for behavior_index, line in enumerate(behavior_file):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                candidates = [article for article in impressions.strip().split(' ')]
                history = history.strip().split(' ')

                if len(history) != 0:
                    user_history = list((history[-self.user_log_length:]))
                    examples.append([user_ID, user_history, candidates])
                else:
                    user_history = [0] * self.user_log_length
                    examples.append([user_ID, user_history, candidates])

            return examples

    def __getitem__(self, item):

        user_idx, user_history, impressions = self.behaviors[item]

        sess = [i.split('-')[0] for i in impressions]
        if self.mode == 'dev':
            labels = [int(i.split('-')[1]) for i in impressions]
        else:
            labels = [0] * len(impressions)

        click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(user_history),
                                                   self.user_log_length)
        sample_news = self.trans_to_nindex(sess)

        user_features = self.parser_text(click_docs)
        news_features = self.parser_text(sample_news)

        log_mask = torch.FloatTensor(log_mask).cuda(self.device)

        label = torch.tensor(labels).cuda(self.device)
        # label = torch.zeros(1)
        return user_features, log_mask, news_features, label
