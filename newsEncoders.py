import torch
import torch.nn as nn
import pickle
from layers import MultiHeadAttention, AdditiveAttention, Context_Aware_Att


class NewsEncoder(nn.Module):
    def __init__(self, args, bert_model, word_embedding_path):
        super(NewsEncoder, self).__init__()
        self.args = args
        self.device = args.device_id
        self.max_title_len = args.max_title_len
        self.max_body_len = args.max_body_len
        self.word_embedding_dim = args.word_embedding_dim
        self.word_embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=self.word_embedding_dim)

        self.bert_model = bert_model
        self.word_embedding_path = word_embedding_path
        self.multihead_attention = MultiHeadAttention(args.word_embedding_dim, args.n_heads, args.n_dim, args.n_dim)
        # self.attention = AdditiveAttention(args.n_heads * args.n_dim, args.attention_dim)
        self.attention = AdditiveAttention(args.word_embedding_dim, args.attention_dim)

        self.reduce_dim_linear = nn.Linear(args.n_heads * args.n_dim, args.news_dim)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.cast = Context_Aware_Att(args.n_heads, args.n_dim, args.word_embedding_dim, args.max_title_len,
                                      args.max_body_len)

        if args.pretrain == 'glove':
            with open(self.word_embedding_path, 'rb') as word_embedding_f:
                self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))

    # Input
    # title_text          : [batch_size, news_num, max_title_length]
    # title_mask          : [batch_size, news_num, max_title_length]
    # body_text           : [batch_size, news_num, max_body_length]
    # body_mask           : [batch_size, news_num, max_body_length]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # mask                : [batch_size, news_num, max_title_length + max_body_length]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def forward(self, news_features):
        title_text = news_features[0]
        title_mask = news_features[1]
        body_text = news_features[2]
        body_mask = news_features[3]
        category = news_features[4]
        sub_category = news_features[5]

        batch_size = category.size(0)
        news_num = category.size(1)

        title_mask = title_mask.view(
            [batch_size * news_num, self.max_title_len])  # [B * L, N]
        body_mask = body_mask.view(
            [batch_size * news_num, self.max_body_len])  # [B * L, M]

        title_text = title_text.view([batch_size * news_num, self.max_title_len])  # [B * L, N]
        body_text = body_text.view([batch_size * news_num, self.max_body_len])  # [B * L, M]

        # title_output = self.bert_model(input_ids=title_text, attention_mask=title_mask)
        # body_output = self.bert_model(input_ids=body_text, attention_mask=body_mask)
        #
        # title_emb = title_output.last_hidden_state
        # body_emb = body_output.last_hidden_state

        input_text = torch.cat([title_text, body_text], dim=1)
        input_mask = torch.cat([title_mask, body_mask], dim=1)
        bert_output = self.bert_model(input_ids=input_text, attention_mask=input_mask)
        word_emb = bert_output.last_hidden_state[:, :self.max_title_len, :]  # [B * L, N, d]

        # worb_emb = self.dropout(self.word_embedding(input_text))
        # c = self.dropout(self.multihead_attention(word_emb, word_emb, word_emb,
        #                                           input_mask))  # [batch_size * news_num, max_sentence_length, news_embedding_dim]
        # c = c[:, :self.max_title_len, :]
        # title_emb = self.dropout(self.word_embedding(title_text))
        # body_emb = self.dropout(self.word_embedding(body_text))

        # c = self.cast(title_emb, body_emb, body_emb, title_mask, body_mask)

        title_rep = self.attention(word_emb, title_mask).view(batch_size, news_num,
                                                              -1)  # [batch_size, news_num, hidden_size]
        # title_rep = self.reduce_dim_linear(title_rep)

        # title_rep = self.feature_fusion(title_rep, category, sub_category)  # [batch_size, news_num, hidden_size+a]
        return title_rep

    # Input
    # news_representation : [batch_size, news_num, unfused_news_embedding_dim]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def feature_fusion(self, news_representation, category, subCategory):
        category_representation = self.category_embedding(category)  # [batch_size, news_num, category_embedding_dim]
        subCategory_representation = self.subCategory_embedding(
            subCategory)  # [batch_size, news_num, subCategory_embedding_dim]
        news_representation = torch.cat(
            [news_representation, self.dropout(category_representation), self.dropout(subCategory_representation)],
            dim=2)  # [batch_size, news_num, news_embedding_dim]
        return news_representation
