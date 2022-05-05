import os

from six.moves.urllib.parse import urlparse
from tqdm import tqdm
import numpy as np
from torchtext.vocab import GloVe
import torch
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

stop_words = set(stopwords.words('english'))
pat = re.compile(r"[\w]+|[.,!?;|]")


def remove_stopwords(sentence):
    words = pat.findall(sentence)
    filtered_sentence = [w for w in words if not w.lower() in stop_words]
    filtered_sentence = ' '.join(filtered_sentence)
    return filtered_sentence


def get_domain(url):
    domain = urlparse(url).netloc
    return domain


def glove(word_dict, word_embedding_dim, datapath):
    # 4. Glove word embedding
    glove = GloVe(name='840B', dim=word_embedding_dim, cache='../glove', max_vectors=10000000000)

    glove_stoi = glove.stoi
    glove_vectors = glove.vectors
    glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)
    word_embedding_vectors = torch.zeros([len(word_dict), word_embedding_dim])
    for word in word_dict:
        index = word_dict[word]
        if index != 0:
            if word in glove_stoi:
                word_embedding_vectors[index, :] = glove_vectors[glove_stoi[word]]
            else:
                random_vector = torch.zeros(word_embedding_dim)
                random_vector.normal_(mean=0, std=0.1)
                word_embedding_vectors[index, :] = random_vector + glove_mean_vector
    with open(datapath, 'wb') as word_embedding_f:
        pickle.dump(word_embedding_vectors, word_embedding_f)


def load_news(news_file, text_path):
    print('load raw news data')
    news = pickle.load(open(os.path.join(text_path, news_file), 'rb'))
    news_index = pickle.load(open(os.path.join(text_path, 'news_index.txt'), 'rb'))
    category_dict = pickle.load(open(os.path.join(text_path, 'category_dict.txt'), 'rb'))
    subcategory_dict = pickle.load(open(os.path.join(text_path, 'subcategory_dict.txt'), 'rb'))
    return news, news_index, category_dict, subcategory_dict


def save_news(news_file, text_path, news, news_index, category_dict, subcategory_dict):
    print('save the raw news data')
    pickle.dump(news, open(os.path.join(text_path, news_file), 'wb'))
    pickle.dump(news_index, open(os.path.join(text_path, 'news_index.txt'), 'wb'))
    pickle.dump(category_dict, open(os.path.join(text_path, 'category_dict.txt'), 'wb'))
    pickle.dump(subcategory_dict, open(os.path.join(text_path, 'subcategory_dict.txt'), 'wb'))


def read_news(data_path, args, tokenizer):
    news = {}
    categories = []
    subcategories = []
    news_index = {}
    index = 1

    train_path = os.path.join(data_path, 'train')
    dev_path = os.path.join(data_path, 'dev')
    test_path = os.path.join(data_path, 'test')

    for i, path in enumerate([train_path, dev_path, test_path]):
        text_path = os.path.join(path, 'news_with_summarized.tsv')
        with open(text_path, 'r', encoding='utf-8') as f:

            for line in tqdm(f):
                splited = line.strip('\n').split('\t')
                doc_id, category, subcategory, title, abstract, _, title_entities, _, body, sbody = splited
                if doc_id in news_index:
                    continue
                news_index[doc_id] = index
                index += 1

                title = title.lower()
                title = tokenizer(title, max_length=args.max_title_len, padding='max_length', truncation=True,
                                  add_special_tokens=False)

                # body = remove_stopwords(body.lower()[:2000])
                body = body.lower()
                body = tokenizer(body, max_length=args.max_body_len, padding='max_length', truncation=True,
                                 add_special_tokens=False)

                categories.append(category)
                subcategories.append(subcategory)

                news[doc_id] = [title, body, category, subcategory]

    categories = list(set(categories))
    category_dict = {}
    index = 1
    for x in categories:
        category_dict[x] = index
        index += 1

    subcategories = list(set(subcategories))
    subcategory_dict = {}
    index = 1
    for x in subcategories:
        subcategory_dict[x] = index
        index += 1

    return news, news_index, category_dict, subcategory_dict


def get_doc_input(news, news_index, category_dict, subcategory_dict, args):
    news_num = len(news) + 1

    news_title = np.zeros((news_num, args.max_title_len), dtype='int32')
    news_title_type = np.zeros((news_num, args.max_title_len), dtype='int32')
    news_title_attmask = np.zeros((news_num, args.max_title_len), dtype='int32')
    news_body = np.zeros((news_num, args.max_body_len), dtype='int32')
    news_body_type = np.zeros((news_num, args.max_body_len), dtype='int32')
    news_body_attmask = np.zeros((news_num, args.max_body_len), dtype='int32')
    news_category = np.zeros(news_num, dtype='int32')
    news_subcategory = np.zeros(news_num, dtype='int32')

    for key in tqdm(news):
        title, body, category, subcategory = news[key]
        doc_index = news_index[key]

        news_title[doc_index] = title['input_ids']
        news_title_type[doc_index] = title['token_type_ids']
        news_title_attmask[doc_index] = title['attention_mask']
        news_body[doc_index] = body['input_ids']
        news_body_type[doc_index] = body['token_type_ids']
        news_body_attmask[doc_index] = body['attention_mask']
        news_category[doc_index] = category_dict[category] if category in category_dict else 0
        news_subcategory[doc_index] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0

    return {'title_text': news_title, 'title_mask': news_title_attmask, 'body_text': news_body,
            'body_mask': news_body_attmask, 'category': news_category, 'sub_category': news_subcategory}
