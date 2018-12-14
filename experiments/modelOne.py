import pandas as pd
import sklearn
import numpy as np
import nltk
import re
import gc
import collections
import sys
import random
from tqdm import tqdm

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn import tree
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import chi2

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from scipy.sparse import hstack, coo_matrix
from sentiment import SentimentFeatureBuilder
################################################################################

def loadData():
    print('loading data')

def prep(articles):
    nArticles = len(articles)
    cleanArticles = []
    stops = set(stopwords.words('english'))

    for i in range(0, nArticles):
        words = articles[i].lower().split()
        words = [w.lower() for w in words if not w in stops]
        cleanArticles.append(' '.join(words))
    return cleanArticles

################################################################################
# Feature extranction and Document Term Matrix parsing using TFIDF

def getDTMByTFIDF(article_matrix, sentiment_matrix, nFeatures):
    print(sentiment_matrix.shape)
    tfIdf_vectorizer = TfidfVectorizer(max_features=nFeatures)
    # dtm = tfIdf_vectorizer.fit_transform(article_matrix).toarray()
    dtm = tfIdf_vectorizer.fit_transform(article_matrix).toarray()
    print(dtm.shape)
    features = np.hstack([dtm, sentiment_matrix])
    return features 

def featuresByChiSq(features, labels, nFeature=5000):
    chi2_model = SelectKBest(chi2, k=nFeature)
    dtm = chi2_model.fit_transform(features, labels)
    return dtm, chi2_model

################################################################################

def crossValidate(document_term_matrix, labels, classifier='SVM', nfold=10):
    precision, recall, fscore = [], [], []
    x_train, x_test, y_train, y_test = [], [], [], []
    actual, guessed = None, None
    clf = LinearSVC()
    labels = np.array(labels)
    
    # Create the split between the training and the testing data
    split_index = int(len(document_term_matrix) * 0.80) 
    matricies = [(x, i) for i, x in enumerate(document_term_matrix)]
    random.shuffle(matricies)

    x_train_shuf, x_test_shuf = matricies[0:split_index], matricies[split_index:]

    while x_train_shuf:
        x, i = x_train_shuf.pop(0)
        y_train.append(labels[i])
        x_train.append(x)

    while x_test_shuf:
        x, i = x_test_shuf.pop(0)
        y_test.append(labels[i])
        x_test.append(x)

    del x_train_shuf
    del x_test_shuf

    guessed = clf.fit(x_train, y_train).predict(x_test)
    actual = y_test

    p, r, f, s = precision_recall_fscore_support(actual, guessed, average='weighted')
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    guessDict = collections.defaultdict(collections.Counter)
    for g, a in zip(guessed, actual):
        guessDict[a][g] += 1
    
    make_matrix(guessDict)

    return round(np.mean(precision), 3), round(np.mean(recall), 3), round(np.mean(fscore), 3), guessDict

def make_matrix(guesses):
    import seaborn as sn
    import matplotlib.pyplot as plt
    SOURCES = ["Vox", "CNN", "Talking Points Memo", "Buzzfeed News", "Washington Post",
        "Guardian", "Atlantic", "Business Insider", "New York Times",
        "NPR", "Reuters", "New York Post", "Fox News", "National Review", "Breitbart"]

    array = []
    for s in SOURCES:
        d = []
        tot = sum(guesses[s].values())
        for other_s in SOURCES:
            if tot != 0:
                d.append(round(guesses[s][other_s]/tot, 2))
            else:
                d.append(0)
        array.append(list(d))
    df_cm = pd.DataFrame(array, SOURCES, SOURCES)
    plt.figure(figsize = (16,16))
    sn.set(font_scale=1.4) #for label size
    x = sn.heatmap(df_cm, annot=True,annot_kws={"size": 20})# font size
    fig = x.get_figure()
    fig.savefig('ML_Sentiment_heatmap')

def modelOne(data):
    article_content, sources = [], []
    sentiment_matrix = []
    feature_builder = SentimentFeatureBuilder()
    for a in tqdm(data):
        article_content.append(a.raw_content)
        sources.append(a.source)
        sentiment_matrix.append(feature_builder.get_article_features(a))
    pArticles = prep(article_content)

    # Garbage collection
    del article_content
    gc.collect()

    # Applies information gain / TDFID using TFIDF and Chi Squared
    print("Using tf-idf...")
    dtm = getDTMByTFIDF(pArticles, np.array(sentiment_matrix), 5000)
    print("Getting features with ChiSquared ...")
    chisqDtm, chisqModel = featuresByChiSq(dtm, sources, 5000)
    # Runs the SVM to make predictions
    print("Running the Linear SVM...")
    p, r, f, guessDict = crossValidate(chisqDtm, sources, 'SVM', 10)

    print('Chi^2 Features:', p, r, f)

    guessPercentageMatrix = []
    guessPercentageDict = collections.defaultdict(dict)

    documentCount = 0
    for actual, guessCounts in guessDict.items():
        guessPercentageRow = []
        for source, guessCount in guessCounts.items():
            documentCount += guessCount
            guessPercentageDict[actual][source] = float(guessCount) / sum(guessCounts.values())

    print('Document Count: ', documentCount)
    for source, guesses in guessPercentageDict.items():
        print('Source: ', source)
        for guess, percentage in guesses.items():
            print('\tGuess: ', guess, ', Percentage: ', percentage)

