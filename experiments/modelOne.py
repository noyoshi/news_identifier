import pandas as pd
import sklearn
import numpy as np
import nltk
import re
import gc
import collections
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

from mlxtend.plotting import plot_confusion_matrix

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn import tree
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import chi2

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

################################################################################

def loadData():
    print('loading data')

def prep(articles):
    nArticles = len(articles)
    cleanArticles = []
    cleanWords = []

    stops = set(stopwords.words('english'))

    for i in range(0, nArticles):
        words = articles[i].lower().split()
        words = [w.lower() for w in words if not w in stops]
        cleanWords.append(words)
        cleanArticles.append(' '.join(words))

    return cleanArticles, cleanWords

def getDTMByTFIDF(articles, nFeatures):
    tfIdf_vectorizer = TfidfVectorizer(max_features=nFeatures)
    dtm = tfIdf_vectorizer.fit_transform(articles).toarray()
    return dtm, tfIdf_vectorizer

################################################################################

def featuresByChiSq(features, labels, nFeature=5000):
    chi2_model = SelectKBest(chi2, k=nFeature)
    dtm = chi2_model.fit_transform(features, labels)
    return dtm, chi2_model

def featuresByInformationGain(features, labels):
    treeCL = tree.DecisionTreeClassifier(criterion='entropy')
    treeCL = treeCL.fit(features, labels)
    transformed_features = SelectFromModel(treeCL, prefit=True).transform(features)
    return transformed_features

def featuresByLSA(features, ncomponents=100):
    svd = TruncatedSVD(n_components=ncomponents)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    dtm_lsa = lsa.fit_transform(features)
    return dtm_lsa

################################################################################

def crossValidate(document_term_matrix, labels, classifier='SVM', nfold=10):
    clf = None
    precision = []
    recall = []
    fscore = []

    if classifier is "RF":
        clf = RandomForestClassifier()
    elif classifier is "NB":
        clf = MultinomialNB()
    elif classifier is "SVM":
        clf = LinearSVC()
    else:
        print('error unkown classifier')
        return

    skf = StratifiedKFold(n_splits=2)
    # skf.get_n_splits(labels)

    actual = None
    guessed = None


    labels = np.array(labels)
    for train_index, test_index in skf.split(document_term_matrix, labels):
        X_train, X_test = document_term_matrix[train_index], document_term_matrix[test_index]

        # print(train_index, test_index, labels)
        y_train, y_test = labels[train_index], labels[test_index]
        model = clf.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # print(X_test)
        # print(y_pred)
        # print(y_test)

        actual = y_test
        guessed = y_pred

        p, r, f, s = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        precision.append(p)
        recall.append(r)
        fscore.append(f)

    guessDict = collections.defaultdict(collections.Counter)
    for g, a in zip(guessed, actual):
        guessDict[a][g] += 1

    # print(guessDict)

    # return round(np.mean(precision), 3), round(np.mean(recall), 3), round(np.mean(fscore), 3), confusion_matrix(y_test, y_pred)
    return round(np.mean(precision), 3), round(np.mean(recall), 3), round(np.mean(fscore), 3), guessDict

def modelOne(data):
    articles = []
    sources = []

    print(len(data))
    for a in data:
        articles.append(a.raw_content)
        sources.append(a.source)

    pArticles, pArticlesWords = prep(articles)

    del data
    del articles
    gc.collect()

    dtm, vect = getDTMByTFIDF(pArticles, 5000)

    chisqDtm, chisqModel = featuresByChiSq(dtm, sources, 5000)
    # chisqDtm, chisqModel = featuresByInformationGain(dtm, sources)
    # chisqDtm= featuresByLSA(dtm)

    p, r, f, guessDict = crossValidate(chisqDtm, sources, 'SVM', 10)
    # p, r, f = crossValidate(chisqDtm, sources, 'RF', 10)
    # p, r, f = crossValidate(chisqDtm, sources, 'NB', 10)

    print('Chi^2 Features:', p, r, f)

    guessPercentageMatrix = []
    guessPercentageDict = collections.defaultdict(dict)

    documentCount = 0
    for actual, guessCounts in guessDict.items():
        guessPercentageRow = []
        for source, guessCount in guessCounts.items():
            # print(guessCount)
            documentCount += guessCount
            guessPercentageDict[actual][source] = float(guessCount) / sum(guessCounts.values())

    # print(guessPercentageDict)

    print('Document Count: ', documentCount)
    for source, guesses in guessPercentageDict.items():
        print('Source: ', source)
        for guess, percentage in guesses.items():
            print('\tGuess: ', guess, ', Percentage: ', percentage)


    # fig, ax = plot_confusion_matrix(conf_mat=guessDict, colorbar=True, show_absolute=False, show_normed=True)

    # ax.xaxis.set_major_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(1))

    # print(sources)

    # ax.set_xticklabels([''] + sources)
    # ax.set_yticklabels([''] + sources)

    # plt.show()
