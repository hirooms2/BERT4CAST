import math
import pickle

import torch.nn as nn
import torch
from layers import MultiHeadAttention
from newsEncoders import NewsEncoder
from userEncoders import UserEncoder


class Model(nn.Module):
    def __init__(self,
                 args,
                 bert_model,
                 tokenizer,
                 word_embedding_path):
        super(Model, self).__init__()
        self.args = args
        self.name = args.name
        self.word_embedding_path = word_embedding_path
        self.news_encoder = NewsEncoder(args,
                                        bert_model,
                                        tokenizer,
                                        word_embedding_path)
        self.user_encoder = UserEncoder(args)

        self.criterion = nn.CrossEntropyLoss()
        self.initialize()

    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()

    def forward(self,
                user_features, log_mask, news_features, label,
                compute_loss=True, stage='CTR'):
        """
        Returns:
          click_probability: batch_size, 1 + K
        """

        # input_ids: batch, history, num_words
        news_vec = self.news_encoder(news_features)  # [batch_size, news_num, hidden_size+c]
        # score_lm, masked_index, masked_voca_id = self.news_encoder.forward_lm(news_features)

        # batch_size, news_dim
        # random_mask = (torch.randn(log_mask.size()) < 0.8).cuda(self.args.device_id)
        # random_mask[: 0] = 1
        # if self.training:
        #     log_mask = log_mask * random_mask
        log_vec = self.news_encoder(user_features)  # [batch_size, hist_len, hidden_size+c]
        # loss_lm2 = self.news_encoder.forward_lm(user_features)
        user_vector = self.user_encoder(log_vec, log_mask, news_vec)

        score = (news_vec * user_vector).sum(dim=2)  # dot-product

        if compute_loss:
            loss = self.criterion(score, label)
            return loss
        else:
            return score
