import torch
import torch.nn as nn
import pickle
from layers import MultiHeadAttention, AdditiveAttention, Context_Aware_Att, Conv1D


# TH
class NewsEncoder(nn.Module):
    def __init__(self, args, bert_model, tokenizer, word_embedding_path):
        super(NewsEncoder, self).__init__()
        self.args = args
        self.device = args.device_id
        self.max_title_len = args.max_title_len
        self.max_body_len = args.max_body_len
        self.word_embedding_dim = args.word_embedding_dim
        self.glove_embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.glove_dim)
        self.hidden_size = args.n_heads * args.n_dim

        self.category_embedding = nn.Embedding(num_embeddings=args.category_num, embedding_dim=args.category_dim,
                                               padding_idx=0)
        self.subCategory_embedding = nn.Embedding(num_embeddings=args.subcategory_num,
                                                  embedding_dim=args.subcategory_dim, padding_idx=0)

        # self.linear_output = nn.Linear(args.n_heads * args.n_dim, self.word_embedding_dim)
        # self.linear_output = nn.Linear(args.word_embedding_dim, args.news_dim)
        self.linear_word = nn.Linear(args.word_embedding_dim + args.glove_dim, args.hidden_size)
        if args.scaling == 'yes':
            self.scalar = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            self.scalar = torch.nn.Parameter(torch.ones(1), requires_grad=False)

        self.reduce_dim_linear = nn.Linear(self.hidden_size + args.category_dim + args.subcategory_dim, args.news_dim)

        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.word_embedding_path = word_embedding_path

        # self.attention = AdditiveAttention(args.n_heads * args.n_dim, args.attention_dim)
        self.attention = AdditiveAttention(self.hidden_size, args.attention_dim)

        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.cast = Context_Aware_Att(args.n_heads, args.n_dim, args.hidden_size, args.max_title_len, args.max_body_len)
        # self.cast = Context_Aware_Att(args.n_heads, args.n_dim, args.glove_dim, args.max_title_len, args.max_body_len)

        with open(self.word_embedding_path, 'rb') as word_embedding_f:
            self.glove_embedding.weight.data.copy_(pickle.load(word_embedding_f))

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        self.cast.initialize()
        self.attention.initialize()

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

        title_bert = self.bert_model(input_ids=title_text, attention_mask=title_mask).last_hidden_state * 1
        body_bert = self.bert_model(input_ids=body_text, attention_mask=body_mask).last_hidden_state * 1
        title_glove = self.glove_embedding(title_text)
        body_glove = self.glove_embedding(body_text)

        title_emb = self.linear_word(torch.cat([title_bert, title_glove], dim=2))
        body_emb = self.linear_word(torch.cat([body_bert, body_glove], dim=2))
        # title_emb = torch.cat([title_bert * self.scalar, title_glove], dim=2)
        # body_emb = torch.cat([body_bert * self.scalar, body_glove], dim=2)

        c = self.dropout(self.cast(title_emb, body_emb, body_emb, title_mask, body_mask))  # [B * L, N, d]
        # c = self.cast(title_emb, body_emb, body_emb, title_mask, body_mask)  # [B * L, N, d]

        title_rep = self.attention(c, title_mask).view(batch_size, news_num, -1)  # [batch_size, news_num, hidden_size]
        news_rep = self.feature_fusion(title_rep, category, sub_category)  # [B, news_num, d+a]

        return news_rep

    # def forward_lm(self, news_features):
    #     title_text = news_features[0]
    #     title_mask = news_features[1]
    #     body_text = news_features[2]
    #     body_mask = news_features[3]
    #     category = news_features[4]
    #     sub_category = news_features[5]
    #
    #     batch_size = category.size(0)
    #     news_num = category.size(1)
    #
    #     title_mask = title_mask.view(
    #         [batch_size * news_num, self.max_title_len])  # [B * L, N]
    #     body_mask = body_mask.view(
    #         [batch_size * news_num, self.max_body_len])  # [B * L, M]
    #     all_mask = torch.cat([title_mask, body_mask], dim=1)  # [B * L, N + M]
    #
    #     title_text = title_text.view([batch_size * news_num, self.max_title_len])  # [B * L, N]
    #     body_text = body_text.view([batch_size * news_num, self.max_body_len])  # [B * L, M]
    #
    #     # only for stopwords???
    #     masked_title_text, masked_index, masked_voca_id = self.mask_tokens(title_text, title_mask)
    #
    #     title_output = self.bert_model(input_ids=title_text, attention_mask=title_mask)
    #     body_output = self.bert_model(input_ids=body_text, attention_mask=body_mask)
    #     title_emb = title_output.last_hidden_state
    #     body_emb = body_output.last_hidden_state
    #
    #     c_masked = self.dropout(self.cast(title_emb, body_emb, body_emb, title_mask, body_mask))  # [B * L, N, d]
    #     c_masked = c_masked[torch.arange(batch_size * news_num), masked_index]
    #     # c_masked = c_masked[:, 0]
    #
    #     # check point::: [d, V]???
    #     # score_lm = self.linear_output(c_masked)
    #
    #     # Loss_LM 만드는 부분
    #     a = self.linear_output(c_masked)
    #     # a = c_masked
    #     b = self.bert_model.embeddings.word_embeddings.weight[:]
    #     # b = self.word_embedding.weight[:]
    #
    #     score_lm = torch.matmul(a, b.transpose(1, 0))  # [B, d] x [d, N] = [B, N]
    #
    #     return score_lm, masked_index, masked_voca_id

    # Input
    # news_representation : [batch_size, news_num, unfused_news_embedding_dim]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def feature_fusion(self, news_representation, category, subCategory):
        category_representation = self.category_embedding(category)  # [batch_size, news_num, category_embedding_dim]
        subCategory_representation = self.subCategory_embedding(subCategory)  # [B, N, s_emb_dim]

        news_representation = torch.cat(
            [news_representation, self.dropout(category_representation), self.dropout(subCategory_representation)],
            dim=2)  # [batch_size, news_num, news_embedding_dim]

        news_representation = self.reduce_dim_linear(news_representation)
        return news_representation

    def mask_tokens(self, title_text: torch.Tensor, title_mask: torch.Tensor, mlm_probability=0.15):
        masked_title_text = title_text.clone()  # [B, N]
        lens = torch.sum(title_mask, dim=1, keepdim=True)  # [B, 1]
        sampling_prob = title_mask / (lens + 1e-10)  # [B, N]
        masked_index = sampling_prob.multinomial(num_samples=1, replacement=True).squeeze(1)  # [B, N]

        masked_voca_id = title_text[torch.arange(title_mask.shape[0]), masked_index]  # [B]
        masked_title_text[torch.arange(title_mask.shape[0]), masked_index] = self.tokenizer.mask_token_id

        return masked_title_text, masked_index, masked_voca_id
