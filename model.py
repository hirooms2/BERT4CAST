import torch.nn as nn

from newsEncoders import NewsEncoder
from userEncoders import UserEncoder


class Model(nn.Module):
    def __init__(self,
                 args,
                 bert_model,
                 word_embedding_path):
        super(Model, self).__init__()
        self.args = args
        self.name = args.name
        self.news_encoder = NewsEncoder(args,
                                        bert_model,
                                        word_embedding_path)
        self.user_encoder = UserEncoder(args)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self,
                user_features, log_mask, news_features, label,
                compute_loss=True):
        """
        Returns:
          click_probability: batch_size, 1 + K
        """
        # input_ids: batch, history, num_words
        news_vec = self.news_encoder(news_features)  # [batch_size, news_num, hidden_size+c]
        loss_lm = self.news_encoder.forward_lm(news_features)

        # batch_size, news_dim
        log_vec = self.news_encoder(user_features)  # [batch_size, hist_len, hidden_size+c]
        # loss_lm2 = self.news_encoder.forward_lm(user_features)

        user_vector = self.user_encoder(log_vec, log_mask, news_vec)

        score = (news_vec * user_vector).sum(dim=2)  # dot-product

        if compute_loss:

            loss = self.criterion(score, label)
            loss += self.args.reg_term * loss_lm ## lm loss , Regularization Term
            # loss = loss_ctr + args.lambda * loss_lm
            return loss, score
        else:
            return score
