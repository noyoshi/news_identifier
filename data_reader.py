#!/usr/bin/env python3
import collections
import csv
import operator
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

from article import Article, ArticleCollection

class DataReader(object):
    def __init__(self, *args):
        """Takes in a list of csv files to parse through"""
        self.args = args[0]
        self.KEYS = ['title', 'publication', 'author', 'date', 'content']
        self.hack_csv()

    def _yield_data(self):
        """In case we can optimize by creating generators, use this"""
        for f in self.args:
            with open(f) as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    data = self._parse_row(row)
                    yield Article(*data)

    def _make_data(self):
        """Parses through files and returns a list of articles"""
        articles = ArticleCollection()
        for f in self.args:
            with open(f) as csv_file:
                reader = csv.DictReader(csv_file)
                for row in tqdm(reader):
                    data = self._parse_row(row)
                    articles.append(Article(*data))

        return articles

    def _parse_row(self, row):
        """Parses out the imporant fields in the row"""
        for k in self.KEYS:
            yield row[k]

    def hack_csv(self):
        """bad stuff happens to the csv library when you try to open large csvs,
        this applies a quick and dirty fix"""
        maxInt = sys.maxsize
        decrement = True

        while decrement:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.
            decrement = False
            try:
                csv.field_size_limit(maxInt)
            except OverflowError:
                maxInt = int(maxInt/10)
                decrement = True

def get_vectors(*strings):
    text = [t for t in strings]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

def get_cosine_sim(*strings):
    vectors = [t for t in get_vectors(*strings)]
    return cosine_similarity(vectors)

if __name__ == "__main__":
    import sys
    dr = DataReader(sys.argv[1:])
    articles = dr._make_data()
    articles.normalize()
    testing, training = articles.make_sets()
    print(articles)
    print(testing)
    print(training)

    sourceStringCombinedDict = collections.defaultdict(str)
    for a in training:
        sourceStringCombinedDict[a.source] += a.raw_content

    correctGuesses = 0
    totalGuesses = 0

    for a in testing:
        cosSimDict = {}
        for source, sourceString in sourceStringCombinedDict.items():
            cosSimDict[source] = get_cosine_sim(a.raw_content, sourceString)[0][1]

        guess = max(cosSimDict.items(), key=operator.itemgetter(1))[0]
        print('Source: ', a.source, ', Guess: ', guess)

        if guess == a.source:
            correctGuesses += 1
        totalGuesses += 1
        print('Correct Guess Percentage at', totalGuesses, ' articles studied: ', (correctGuesses / totalGuesses))

    print('Correct Guess Percentage: ', (correctGuesses / totalGuesses))
    

    # for source, string in sourceStringCombinedDict.items():
    #     print(get_vector(string))


    # sourceWordCountDict = collections.defaultdict(dict)
    # count = 0
    # for a in training:
    #     for w in a._tokenize(a.raw_content):
    #         if w not in sourceWordCountDict[a.source].keys():
    #             sourceWordCountDict[a.source][w] = 1
    #         else:
    #             sourceWordCountDict[a.source][w] += 1

    #     if count > 4:
    #         break

    # cleanedWordCountDict = collections.defaultdict(dict)
    # for s in sourceWordCountDict.keys():
    #     for w, c in sourceWordCountDict[s].items():
    #         if c > 1:
    #             cleanedWordCountDict[s][w] = c

    # print(cleanedWordCountDict)
