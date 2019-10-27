import gensim
import os
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

corpus_list = []
for dir_ext in os.listdir('20_newsgroups/'):
    dir_name = '20_newsgroups/{}/'.format(dir_ext)
    for file_ext in os.listdir(dir_name):
        file_name = '{}{}'.format(dir_name,file_ext)
        file = open(file_name, 'r')
        text_list = file.readlines()
        text = ''.join(text_list)
        text = text.lower()
        text = re.sub('\W',' ', text)
        text = re.sub('\d',' ', text)
        text_tokens = nltk.word_tokenize(text)
        corpus_list.append(text_tokens)


## Train baseline model (using default parameters, including CBOW)
model = Word2Vec(corpus_list, size=100, window=5, min_count=1, workers=4)
model.save("word2vec_baseline.model")

## Train skip-gram model
model_sg = Word2Vec(corpus_list, size=100, window=5, min_count=1, workers=4, sg=1)
model_sg.save("word2vec_skipgram.model")

## Train model using two-word phrases
bigram_transformer = Phrases(corpus_list)
model_withphrases = Word2Vec(bigram_transformer[corpus_list], size=100, window=5, min_count=1, workers=4)
model_withphrases.save("word2vec_withphrases.model")

## Embeddings with skip-gram and bigrams
model_sg_withphrases = Word2Vec(bigram_transformer[corpus_list], size=100, window=5, min_count=1, workers=4, sg=1)
model_sg_withphrases.save("word2vec_sg_withphrases.model")

## Compare several words to see how models the most similar words compare across our differently trained models
word_comp_list = ['wisconsin', 'atheism', 'college', 'pizza', 'math', 'halloween', 'water', 'sunday', 'anarchy', 'train']
word_comp_dict = {}

for word in word_comp_list:
    word_comp_dict[word] = {}
    
    baseline_similar = model.wv.similar_by_word(word, topn=5)
    sg_similar = model_sg.wv.similar_by_word(word, topn=5)
    with_phrases_similar = model_withphrases.wv.similar_by_word(word, topn=5)
    sg_with_phrases_similar = model_sg_withphrases.wv.similar_by_word(word, topn=5)
    
    word_comp_dict[word]['baseline'] = baseline_similar
    word_comp_dict[word]['skipgram'] = sg_similar
    word_comp_dict[word]['withphrase'] = with_phrases_similar
    word_comp_dict[word]['sg_withphrase'] = sg_with_phrases_similar

word_comp_dict