#!/usr/bin/env python3
# Source: https://datascienceplus.com/topic-modeling-in-python-with-nltk-and-gensim/
from data_reader import DataReader
import sys
import spacy
from spacy.lang.en import English
parser = English()

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

if __name__ == "__main__":
    text_data = []
    dr = DataReader(sys.argv[1:])
    articles = dr._make_data()
    articles.normalize()
    testing, training = articles.make_sets()
    print(articles)
    print(testing)
    print(training)
    for a in training: 
        text = a.raw_content # The raw text
        tokens = prepare_text_for_lda(text)
        if random.random() > .99:
            print(tokens)
            text_data.append(tokens)
