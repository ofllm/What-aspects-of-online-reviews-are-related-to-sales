# Aspect-based sentiment analysis 
# step
- download pretrain model:chinese_roberta_wwm_large_ext_pytorch（https://drive.google.com/open?id=1-2vEZfIFCdM1-vJ3GD6DlSyKT4eVXMKq） ,extract to ../model/chinese_roberta_wwm_large_ext_pytorch 
- download fasttext word embedding vector（https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz）, extract to)../model/fasttext
- sample data:data/train.json
- train:python train.py
- predict:pyhton predict_csv.py

- rnn train/predict:python rnn/train_model.py
- svm train/predict:python svm/svm.py
