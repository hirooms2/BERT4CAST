source activate <NAME_YOUR_ENV>
python download_MIND.py

git clone https://github.com/msnews/MIND.git

cd MIND/crawler

conda env create -f environment.yaml
source activate scrapy

cd MIND/crawler

export MIND_NEWS_PATH=../../../MIND-small/train/news.tsv
scrapy crawl msn -o msn.json
mv msn.json ../../../MIND-small/train

export MIND_NEWS_PATH=../../../MIND-small/dev/news.tsv
scrapy crawl msn -o msn.json
mv msn.json ../../../MIND-small/dev

export MIND_NEWS_PATH=../../../MIND-small/test/news.tsv
scrapy crawl msn -o msn.json
mv msn.json ../../../MIND-small/test

export MIND_NEWS_PATH=../../../MIND-large/train/news.tsv
scrapy crawl msn -o msn.json
mv msn.json ../../../MIND-large/train

export MIND_NEWS_PATH=../../../MIND-large/dev/news.tsv
scrapy crawl msn -o msn.json
mv msn.json ../../../MIND-large/dev

export MIND_NEWS_PATH=../../../MIND-large/test/news.tsv
scrapy crawl msn -o msn.json
mv msn.json ../../../MIND-large/test

source activate <NAME_YOUR_ENV>

cd ../..

python body_concat.py