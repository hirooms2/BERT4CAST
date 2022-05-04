import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import AdditiveAttention


class UserEncoder(torch.nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args

        # self.news_dim = args.word_embedding_dim
        self.news_dim = args.n_heads * args.n_dim + args.category_dim + args.subcategory_dim
        # self.news_dim = args.n_heads * args.n_dim
        # self.news_dim = args.news_dim

        self.affine1 = nn.Linear(2 * self.news_dim, args.attention_dim)
        self.affine2 = nn.Linear(args.attention_dim, 1)

        self.news_additive_attention = AdditiveAttention(
            args.news_dim, args.attention_dim)
        self.gru = nn.GRU(self.news_dim, self.news_dim, batch_first=True)

    def forward(self, log_vec, log_mask, news_vec):
        """
        Inputs:
            (log_vec) batch_size, hist_len, news_dim
            (log_mask) batch_size, hist_len
            (news_vec) batch_size, news_num


        Returns:
            (shape) batch_size,  news_dim
        """

        # # batch_size, news_dim
        # log_vec = self._process_news(log_vec, log_mask, self.pad_doc,
        #                              self.news_additive_attention, self.args.user_log_mask,
        #                              self.args.use_padded_news_embedding)
        #
        # user_log_vecs = log_vec

        news_num = news_vec.size(1)
        hist_len = log_mask.size(1)

        log_vec, _ = self.gru(log_vec)

        log_mask = log_mask.unsqueeze(dim=1).expand(-1, news_num, -1)  # [batch_size, news_num, hist_len]
        news_vec = news_vec.unsqueeze(dim=2).expand(-1, -1, hist_len, -1)  # [batch_size, news_num, hist_len, news_dim]
        log_vec = log_vec.unsqueeze(dim=1).expand(-1, news_num, -1, -1)  # [batch_size, news_num, hist_len, news_dim]
        _con = torch.cat([news_vec, log_vec], dim=3)  # [batch_size, news_num, hist_len, 2 * news_dim]
        logits = self.affine2(torch.tanh(self.affine1(_con))).squeeze(dim=3)  # [batch, news_num, hist_len]
        attention = F.softmax(logits.masked_fill(log_mask == 0, -1e9), dim=2)  # [batch, news_num, hist_len]
        user_log_vecs = (attention.unsqueeze(dim=3) * log_vec).sum(dim=2)  # [batch, news_num, news_dim]

        return user_log_vecs
