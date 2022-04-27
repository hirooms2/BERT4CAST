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

        self.masked_token_emb = nn.Parameter(torch.zeros(self.word_embedding_dim), requires_grad=True)
        torch.nn.init.normal_(self.masked_token_emb)
        self.linear_output = nn.Linear(args.n_heads * args.n_dim, self.word_embedding_dim)
        # self.linear_output = nn.Linear(args.n_heads * args.n_dim, args.vocab_size, bias=False)

        self.bert_model = bert_model
        self.word_embedding_path = word_embedding_path
        self.multihead_attention = MultiHeadAttention(args.word_embedding_dim, args.n_heads, args.n_dim, args.n_dim)
        self.attention = AdditiveAttention(args.n_heads * args.n_dim, args.attention_dim)
        # self.attention = AdditiveAttention(args.word_embedding_dim, args.attention_dim)

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

        title_emb = self.dropout(self.word_embedding(title_text))  # [B * L, N, d]
        body_emb = self.dropout(self.word_embedding(body_text))  # [B * L, M, d]

        # title_emb = self.dropout(self.multihead_attention(title_emb, title_emb, title_emb, title_mask))
        # body_emb = self.dropout(self.multihead_attention(body_emb, body_emb, body_emb, body_mask))

        # all_emb = torch.cat([title_emb, body_emb], dim=1)  # [B * L, N + M, d]
        # all_mask = torch.cat([title_mask, body_mask], dim=1)  # [B * L, N + M]

        # masked_word_emb = torch.cat([title_emb, body_emb], dim=1)  # [B * L, N + M, d]

        # title_output = self.bert_model(input_ids=title_text, attention_mask=title_mask)
        # body_output = self.bert_model(input_ids=body_text, attention_mask=body_mask)
        # title_emb = title_output.last_hidden_state
        # body_emb = body_output.last_hidden_state
        # input_emb = torch.cat([title_text, body_text], dim=1)
        # input_mask = torch.cat([title_mask, body_mask], dim=1)
        # bert_output = self.bert_model(input_ids=input_text, attention_mask=input_mask)
        # word_emb = bert_output.last_hidden_state[:, :self.max_title_len, :]  # [B * L, N, d]
        # word_emb = self.word_embedding(input_text)

        # worb_emb = self.dropout(self.word_embedding(input_text))
        # c = self.dropout(self.multihead_attention(all_emb, all_emb, all_emb,
        #                                           all_mask))  # [batch_size * news_num, max_sentence_length, news_embedding_dim]
        # c = c[:, :self.max_title_len, :]

        # title_emb = self.dropout(self.word_embedding(title_text))
        # body_emb = self.dropout(self.word_embedding(body_text))
        c = self.cast(title_emb, body_emb, body_emb, title_mask, body_mask)  # [B * L, N, d]

        title_rep = self.attention(c, title_mask).view(batch_size, news_num,
                                                       -1)  # [batch_size, news_num, hidden_size]
        # title_rep = self.reduce_dim_linear(title_rep)
        # title_rep = self.feature_fusion(title_rep, category, sub_category)  # [batch_size, news_num, hidden_size+a]
        return title_rep

    def forward_lm(self, news_features):
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
        all_mask = torch.cat([title_mask, body_mask], dim=1)  # [B * L, N + M]

        title_text = title_text.view([batch_size * news_num, self.max_title_len])  # [B * L, N]
        body_text = body_text.view([batch_size * news_num, self.max_body_len])  # [B * L, M]g

        # only for stopwords???
        lens = torch.sum(title_mask, dim=1, keepdim=True)
        sampling_prob = title_mask / (lens + 1e-10)
        masked_index = sampling_prob.multinomial(num_samples=1, replacement=True)
        masked_index = masked_index.squeeze(1)
        masked_voca_id = title_text.clone().detach()[torch.arange(batch_size * news_num), masked_index]

        title_masked_emb = self.dropout(self.word_embedding(title_text))
        title_masked_emb[torch.arange(batch_size * news_num), masked_index] = self.masked_token_emb
        # title_masked_emb[:, 0] = self.masked_token_emb
        body_emb = self.dropout(self.word_embedding(body_text))  # [B * L, M, d]

        # masked_emb = torch.cat([title_masked_emb, body_emb], dim=1)  # [B * L, N + M, d]
        # c_masked = self.dropout(self.multihead_attention(masked_emb, masked_emb, masked_emb,
        #                                                  all_mask))  # [batch_size * news_num, max_sentence_length, news_embedding_dim]
        # c_masked = c_masked[torch.arange(batch_size * news_num), masked_index]

        c_masked = self.cast(title_masked_emb, body_emb, body_emb, title_mask, body_mask)  # [B * L, N, d]
        c_masked = c_masked[torch.arange(batch_size * news_num), masked_index]
        # c_masked = c_masked[:, 0]

        # check point::: [d, V]???
        # score_lm = self.linear_output(c_masked)

        # Loss_LM 만드는 부분
        a = self.linear_output(c_masked)
        b = self.word_embedding.weight[:]
        score_lm = torch.matmul(a, b.transpose(1, 0))  # [B, d] x [d, N] = [B, N]

        return score_lm, masked_index, masked_voca_id

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
