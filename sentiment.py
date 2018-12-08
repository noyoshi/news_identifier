#!/usr/bin/env python3

# For each article, I want to generate a sentiment score for that article. 
# To do this, I need to
#   1. Do POS tagging and tokenizing of the article?
#   2. Run those tokens and tags into this dictionary and see the results?
# The POS tagged word should be: word#POS
# NOTE I think that words that are the same, but different POS would not have
# much of an effect on our data - so I am going to ignore the POS. This should
# be fine as we are looking at the document as whole and not smaller sentence or
# word level things.

import sys
from data_reader import DataReader

sentiment_data = {}
with open("data/senti_words.txt") as in_f:
    for line in in_f:
        line = line.strip()
        word_tag, score = line.split('\t')
        score = float(score)
        word, tag = word_tag.split('#')
        sentiment_data[word] = score

dr = DataReader(sys.argv[1:])
articles = dr._make_data()
articles.normalize()

for a in articles:
    sentiment = 0
    # For each article, get the tokenized data
    for word in a.tokenized_content:
        if word not in sentiment_data:
            continue
        sentiment += sentiment_data[word]
    print(a.title, sentiment)


