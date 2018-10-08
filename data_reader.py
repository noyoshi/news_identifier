#!/usr/bin/env python3
import csv

from article import Article

class DataReader(object):
    def __init__(self, *args):
        """Takes in a list of csv files to parse through"""
        self.args = args[0]
        self.KEYS = ['author', 'content', 'date', 'publication', 'title']

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
        articles = []
        for f in self.args:
            with open(f) as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    auth, cont, date, src, title = self._parse_row(row)
                    articles.append(Article(title, src, auth, date, cont))
        return articles

    def _parse_row(self, row):
        """Parses out the imporant fields in the row"""
        for k in self.KEYS:
            yield row[k]

if __name__ == "__main__":
    import sys
    dr = DataReader(sys.argv[1:])
    for x in dr._make_data():
        print(x)
        print(x.tokenized_content)
        break
