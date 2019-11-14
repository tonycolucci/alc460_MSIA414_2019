import pandas as pd
import gzip
import json
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

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


def train(df, bigram=False, sl_tf=False, max_feat=750):
    if bigram:
        n_gram = (1,2)
    else:
        n_gram = (1,1)

    tfidf = TfidfVectorizer(input = 'content', ngram_range = n_gram, max_features = 750, sublinear_tf = sl_tf)
    features = tfidf.fit_transform(df.review_text_parsed)

    X_train, X_test, y_train, y_test = train_test_split(features, df.positive, test_size=0.2, random_state=0)

    clf = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print("F1-score: {}".format(f1_score(y_test, y_pred)))



if os.path.exists('video_game_reviews.csv'):
    df = pd.read_csv('video_game_reviews.csv')
    df = df.fillna('')
else:
    df = getDF('Video_Games.json.gz')
    df = df.iloc[0:510000,:]
    df = text_features(df)
    save_file = df.loc[:,['overall','positive','reviewText','reviewText_parsed']]
    save_file.to_csv('video_game_reviews.csv')

print("Data Collected")

print("Baseline (No bigrams, linear tf, 750 max features):")
train(df, bigram=False, sl_tf=False)
print("With bigrams:")
train(df, bigram=True)
print("Sublinear tf:")
train(df, sl_tf=True)
print("500 max features")
train(df, max_feat=500)
print("1000 max features")
train(df, max_feat=1000)

