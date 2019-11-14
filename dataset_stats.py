import pandas as pd
import gzip
import json
import re
import os

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

    return df

def get_dataset_stats(df):
    output_dict = {}

    output_dict['num_reviews'] = df.shape[0]
    output_dict['average_score'] = df.overall.mean()
    output_dict['average_review_len_words'] = df.review_num_words.mean()
    output_dict['average_review_len_characters'] = df.review_num_characters.mean()
    
    return output_dict

if not os.path.exists('video_game_reviews.csv'):
    df = pd.read_csv('video_game_reviews.csv')
else:
    df = getDF('Video_Games.json.gz')
    df = df.iloc[0:500000,:]
    df = text_features(df)
    save_file = df.loc[:,['overall','positive','reviewText','review_text_parsed']]
    save_file.to_csv('video_game_reviews.csv')

stat_dict = get_dataset_stats(df)
print(stat_dict)



