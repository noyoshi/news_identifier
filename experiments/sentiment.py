#!/usr/bin/env python3

# For each article, I want to generate a sentiment score for that article. 
# I want to get the percentage of words they use that are classified as
# "poloarizing" by this dataset that I have. Then, I want to see how many are
# positve, how many are negavtive, the sums of both, and the average scores for
# both the positive and negative ratings?

# Features are for both title and body of article, and they are:
# Number of positive words, number of negative words, average of positive words,
# average of negative words, average for whole thing, sum of positive words, sum
# of negative words

import sys
import collections

import pandas as pd
from sklearn import svm

from data_reader import DataReader

class SentimentFeatureBuilder(object):
    def __init__(self, data_loc="../data/senti_words.txt"):
        self.sentiment_data = collections.defaultdict(int)
        self._initialize_data(data_loc)
        self.feature_names = ["title_num_pos", "title_num_neg", 
                "title_avg_pos", "title_avg_neg",
                "title_avg", "sum_pos_title", "sum_neg_title"]
        self.feature_names += ["body_num_pos", "body_num_neg", 
                "body_avg_pos", "body_avg_neg",
                "body_avg", "sum_pos_body", "sum_neg_body", "label"]

    def _initialize_data(self, data_loc):
        """Initializes the data from the stuff found in data_loc"""
        with open(data_loc) as in_f:
            for line in in_f:
                line = line.strip()
                word_tag, score = line.split('\t')
                score = float(score)
                word, tag = word_tag.split('#')
                self.sentiment_data[word] = score

    def get_article_features(self, a):
        """Extracts the features from the given article"""
        sentiment_data = self.sentiment_data
        tok_content = a.get_tokenized_content()
        tok_title = a.tokenized_title
        body_sentiment = [sentiment_data[x] for x in tok_content if sentiment_data[x] != 0]
        title_sentiment = [sentiment_data[x] for x in tok_title if sentiment_data[x] != 0]
        pos_body, neg_body, pos_title, neg_title = [], [], [], []
        sum_pos_body, sum_pos_title, sum_neg_body, sum_neg_title = 0, 0, 0, 0
        body_num_pos, body_num_neg, title_num_pos, title_num_neg = 0, 0, 0, 0

        # Get the features
        for x in body_sentiment:
            if x > 0:
                pos_body.append(x)
                sum_pos_body += x
                body_num_pos += 1
            elif x < 0:
                neg_body.append(x)
                sum_neg_body += x
                body_num_neg += 1

        for x in title_sentiment:
            if x > 0:
                pos_title.append(x)
                sum_pos_title += x
                title_num_pos += 1
            elif x < 0:
                neg_title.append(x)
                sum_neg_title += x
                title_num_neg += 1
        
        # Average negative and positive scores
        body_avg_pos = sum_pos_body / body_num_pos if body_num_pos != 0 else 0 
        title_avg_pos = sum_pos_title / title_num_pos if title_num_pos != 0 else 0
        body_avg_neg = sum_neg_body / body_num_neg if body_num_neg != 0 else 0 
        title_avg_neg = sum_neg_title / title_num_neg if title_num_neg != 0 else 0

        # The sum of all scores divided by the length of the scored words

        # Title features
        features = [title_num_pos, title_num_neg, 
                title_avg_pos, title_avg_neg * -1,
                sum(pos_title), sum(neg_title) * -1]
        # body features
        # NOTE that i cant have negative numbers, so i got rid of the body and
        # title avg!
        features += [body_num_pos, body_num_neg, 
                body_avg_pos, body_avg_neg * -1,
                sum(pos_body), sum(neg_body) * -1]

        return features

def print_features(features, source):
    print(make_line(features, source))

def save_features(features, source, out_file):
    line = make_line(features, source) 
    out_file.write(line + '\n')

def make_line(features, source):
    return ','.join(map(str, features)) + ',{}'.format(source)

def save_data(articles, feature_builder, file_loc):
    """Saves the data to use for training or testing later"""
    with open(file_loc, 'w+') as out_f:
        out_f.write(','.join(feature_builder.feature_names) + '\n')
        for a in articles:
            features = feature_builder.get_article_features(a)
            save_features(features, a.source, out_f)

def print_data(articles, feature_builder):
    print(','.join(feature_builder.feature_names))
    for a in articles:
        features = feature_builder.get_article_features(a)
        print_features(features, a.source)

if __name__ == '__main__':
    dr = DataReader(sys.argv[1:])
    articles = dr._make_data()
    articles.normalize()
    sent_feature_builder = SentimentFeatureBuilder()
    testing, training = articles.make_sets()

    # Print the features to STDOUT to be used as training data?
    # save_data(training, sent_feature_builder, "training.txt")
    # save_data(testing, sent_feature_builder, "testing.txt")
    
    # Try to use a linear SVC to fit?
    model = svm.LinearSVC()
    
    df_training = pd.read_csv("training.txt")
    df_testing = pd.read_csv("testing.txt")
    training_data = df_training.loc[:, df_training.columns != 'label']
    training_label = df_training['label']

    testing_data = df_testing.loc[:, df_testing.columns != 'label']
    testing_label = df_testing['label']

    model.fit(training_data, training_label)
    predictions = model.predict(testing_data)
    correct = 0
    for i, ans in enumerate(testing_label):
        if ans == predictions[i]:
            correct += 1
    print(correct / len(predictions))

    







