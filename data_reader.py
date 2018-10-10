#!/usr/bin/env python3
import collections
import csv

from tqdm import tqdm

from article import Article, ArticleCollection

class DataReader(object):
    def __init__(self, *args):
        """Takes in a list of csv files to parse through"""
        self.args = args[0]
        self.KEYS = ['author', 'content', 'date', 'publication', 'title']
        self.hack_csv()

    def _yield_data(self):
        """In case we can optimize by creating generators, use this"""
        for f in self.args:
            with open(f) as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    auth, cont, date, src, title = self._parse_row(row)
                    yield Article(title, src, auth, date, cont)

    def _make_data(self):
        """Parses through files and returns a list of articles"""
        articles = ArticleCollection()
        for f in self.args:
            with open(f) as csv_file:
                reader = csv.DictReader(csv_file)
                for row in tqdm(reader):
                    auth, cont, date, src, title = self._parse_row(row)
                    articles.append(Article(title, src, auth, date, cont))

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

if __name__ == "__main__":
    import sys
    dr = DataReader(sys.argv[1:])
    articles = dr._make_data()
    articles.normalize()
    testing, training = articles.make_sets()
    print(articles)
    print(testing)
    print(training)
