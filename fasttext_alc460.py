import pandas as pd
import os
import re
import fasttext
import gzip
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def text_features(df):
    df.reviewText = [str(x) for x in df['reviewText'].tolist()]

    df['review_num_words'] = [len(x.split()) for x in df['reviewText'].tolist()]
    df['review_num_characters'] = [len(x) for x in df['reviewText'].tolist()]
    df['review_text_parsed'] = df['reviewText'].str.lower()
    df['review_text_parsed'] = [str(x) for x in df['review_text_parsed'].tolist()]
    df['review_text_parsed'] = [re.sub(r'[^\w\s]','', x) for x in df['review_text_parsed'].tolist()]
    df['review_text_parsed'] = [re.sub(r'\d','', x) for x in df['review_text_parsed'].tolist()]
    df['positive'] = 1
    df.loc[df.overall < 4, 'positive'] = 0
    df = df.fillna('')
    
    return df

def create_fasttext_docs(X_train, y_train):
    
    y_train = y_train.reset_index()['positive']
    X_train = X_train.reset_index()['review_text_parsed']


    train_list=[]
    train_file = open("train_corpus.txt","a")
    for i in range(len(y_train)):
        if y_train[i] == 1:
            train_list.append('__label__positive, {}'.format(X_train[i]))
        else:
            train_list.append('__label__negative, {}'.format(X_train[i]))
    train_file.writelines(train_list)
    train_file.close()

    test_list=[]
    test_file = open("test_corpus.txt","a")
    for i in range(len(y_test)):
        if y_test[i] == 1:
            test_list.append('__label__positive, {}'.format(X_test[i]))
        else:
            test_list.append('__label__negative, {}'.format(X_test[i]))
    test_file.writelines(test_list)
    test_file.close()
    

if os.path.exists('video_game_reviews.csv'):
    df = pd.read_csv('video_game_reviews.csv')
    df = df.fillna('')
else:
    df = getDF('Video_Games.json.gz')
    df = df.iloc[0:510000,:]
    df = text_features(df)
    save_file = df.loc[:,['overall','positive','reviewText','review_text_parsed']]
    save_file.to_csv('video_game_reviews.csv')

X_train, X_test, y_train, y_test = train_test_split(df.review_text_parsed, df.positive, test_size=0.2, random_state=0)

if (os.path.exists('train_corpus.txt')) & (os.path.exists('test_corpus.txt')):
    pass
else:
    create_fasttext_docs(X_train, y_train)

y_test = y_test.reset_index()['positive']
X_test = X_test.reset_index()['review_text_parsed']

ft_model_baseline = fasttext.train_supervised('train_corpus.txt', thread=3)

ft_labels_baseline = [ft_model_baseline.predict(x)[0][0][9:-1] for x in X_test]

y_pred_baseline = list(pd.factorize(ft_labels_baseline)[0])

print(confusion_matrix(y_test, y_pred_baseline))
print("F1-score (baseline model): {}".format(f1_score(y_test, y_pred_baseline)))

ft_model_bigrams = fasttext.train_supervised('train_corpus.txt', thread=3, wordNgrams=2)

ft_labels_bigrams = [ft_model_bigrams.predict(x)[0][0][9:-1] for x in X_test]

y_pred_bigrams = list(pd.factorize(ft_labels_bigrams)[0])

print(confusion_matrix(y_test, y_pred_bigrams))
print("F1-score (with bigrams): {}".format(f1_score(y_test, y_pred_bigrams)))

ft_model_context3 = fasttext.train_supervised('train_corpus.txt', thread=3, ws=3)

ft_labels_context3 = [ft_model_context3.predict(x)[0][0][9:-1] for x in X_test]

y_pred_context3 = list(pd.factorize(ft_labels_context3)[0])

print(confusion_matrix(y_test, y_pred_context3))
print("F1-score (with bigrams): {}".format(f1_score(y_test, y_pred_context3)))

ft_model_epoch10 = fasttext.train_supervised('train_corpus.txt', thread=3, epoch=10)

ft_labels_epoch10 = [ft_model_epoch10.predict(x)[0][0][9:-1] for x in X_test]

y_pred_epoch10 = list(pd.factorize(ft_labels_epoch10)[0])

print(confusion_matrix(y_test, y_pred_epoch10))
print("F1-score (with bigrams): {}".format(f1_score(y_test, y_pred_epoch10)))