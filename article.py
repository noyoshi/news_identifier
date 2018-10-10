#!/usr/bin/env python3
import collections
import nltk

class Article(object):
    def __init__(self, title, source, author, date, content):
        self.author = author
        self.date = date # YYYY-MM-DD
        self.raw_content = content
        self.source = source
        self.title = title

        self.tokenized_title = self._tokenize(title)
        self.tokenized_content = self._tokenize(content)

        self.aliases = [source]
        self.redacted_string = "[NEWS_SOURCE]"

    def _scrub_names(self, string):
        """Scrubs the name(s) of the news source from the string"""
        for name in self.aliases:
            string = string.replace(name, self.redacted_string)

        return string

    def _tokenize(self, string):
        return nltk.word_tokenize(string)


    def __str__(self):
        return "Author: {}\nSource: {}\nTitle: {}\nBody: {}".format(
                self.author,
                self.source,
                self.title,
                self.raw_content)

class ArticleCollection(object):
    """Iterable collection of Article objects"""
    def __init__(self, data=[]):
        self.data = data
        self.data_index = 0
        self.source_counts = collections.Counter()

    def normalize(self):
        """Normalizes the data. Finds the news source with the least amount of
        articles in here and deletes items itself until all the distribution of
        articles across sources is the same for everything"""
        min_tup = min(self.source_counts.most_common(), key=lambda x: x[1])
        min_count = min_tup[1]
        new_data = []

        # TODO use some random sampling, such that the order that we read in
        # articles has nothing to do with whether they are deleted or not
        for article in self.data:
            if self.source_counts[article.source] > min_count:
                self.source_counts[article.source] -= 1
            else:
                new_data.append(article)

        self.data = new_data

    def copy(self, other_article_collection):
        for article in other_article_collection:
            self.append(article)
        
    # Making it behave like a list, part 1 -----------------------------------
    def append(self, article):
        self.data.append(article)
        self.source_counts[article.source] += 1
    
    # Making it an iterable --------------------------------------------------
    def __iter__(self):
        return self

    def __next__(self):
        if self.data_index >= len(self.data):
            raise StopIteration
        else:
            self.data_index += 1
            return self.data[self.data_index - 1]

    # Lets you print info about it -------------------------------------------
    def __str__(self):
        output = "Total Articles: {}\nSources:\n".format(len(self.data))
        for source in self.source_counts:
            count = self.source_counts[source]
            string = "\tSource: {}\tCount: {}\n".format(source, count)
            output += string
        return output

