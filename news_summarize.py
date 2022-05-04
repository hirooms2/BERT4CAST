from summarizer import Summarizer
import os
from tqdm import tqdm

index = 1

data_path = os.path.join('./datasets/', 'MIND/small')

train_path = os.path.join(data_path, 'train')
dev_path = os.path.join(data_path, 'dev')
test_path = os.path.join(data_path, 'test')

summarizer = Summarizer()

for i, path in enumerate([train_path, dev_path, test_path]):
    text_path = os.path.join(path, 'news_with_body.tsv')
    w_text_path = os.path.join(path, 'news_with_summarized.tsv')

    with open(text_path, 'r', encoding='utf-8') as rf:
        with open(w_text_path, 'w', encoding='utf-8') as wf:
            for line in tqdm(rf):
                splited = line.strip('\n')
                splited2 = splited.split('\t')
                doc_id, category, subcategory, title, abstract, _, title_entities, _, body = splited2

                summarized = summarizer(body, num_sentences=3)

                s = '%s\t%s\n' % (splited, summarized)
                wf.write(s)
