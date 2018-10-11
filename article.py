#!/usr/bin/env python3
import collections
import nltk
import random

from tqdm import tqdm

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
    def __init__(self, data={}):
        self.current_indicies = {}
        self.data = data

        self.iter_n = 0
        self.min_count = 10000000
        self.size = 0
        self.sources = set()

    def normalize(self):
        """Normalizes the data. Finds the news source with the least amount of
        articles in here and deletes items itself until all the distribution of
        articles across sources is the same for everything"""
        self._get_min()
        for src in self.data:
            while len(self.data[src]) > self.min_count:
                index = random.randint(0, len(self.data[src]) - 1)
                self.size -= 1
                del self.data[src][index]

    def _get_min(self):
        for src in self.data:
            if len(self.data[src]) < self.min_count:
                self.min_count = len(self.data[src])

    def copy(self, other_article_collection):
        for article in other_article_collection:
            self.append(article)

    def make_sets(self):
        """Returns new lists that are split into testing(20%), and
        developing(20%). The CURRENT COLLECTION will be used as training data.
        This data will also be stores on the current collection"""
        testing_size = int(self.size * 0.2)
        developing_size = testing_size

        testing = self.make_single_set(testing_size)
        developing = self.make_single_set(developing_size)

        return testing, developing

    def make_single_set(self, size):
        """Don't touch this, idk how it works but it is very fragile"""
        # TODO make this more robust... if you remove the empty dict in the line
        # below it doesn't work, it pretty much copies itself into new set or
        # somethinig?
        new_set = self.__class__({})
        while len(new_set) < size:
            for src in self.sources:
                index = random.randint(0, len(self.data[src]) - 1)
                self.size -= 1

                article = self.data[src][index]
                new_set.append(article)
                del self.data[src][index]

        return new_set

    # Making it behave like a list, part 1 -----------------------------------
    def append(self, article):
        self.size += 1
        src = article.source

        if src not in self.data:
            self.data[src] = []
        self.data[src].append(article)
        self.sources.add(src)

        if src not in self.current_indicies:
            self.current_indicies[src] = 0

    # Making it an iterable --------------------------------------------------
    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_n >= self.size:
            # Reset the current_indicies dictionary
            for src in self.current_indicies:
                self.current_indicies[src] = 0
            raise StopIteration
        else:
            news_sources = list(self.sources)
            random.shuffle(news_sources)
            src = news_sources.pop()
            # I have a shuffled list of all news sources
            while self.current_indicies[src] >= len(self.data[src]):
                # Find a news soure I have not already traverse, randomly
                src = news_sources.pop()

            # Gets the index of the source
            n = self.current_indicies[src]

            # Updates the curret index of the source
            self.current_indicies[src] += 1

            # Updates how many things we have iterated through
            self.iter_n += 1

            return self.data[src][n]

    # Lets you print info about it -------------------------------------------
    def __str__(self):
        output = "Total Articles: {}\nSources:\n".format(self.size)
        for src in self.data:
            count = len(self.data[src])
            string = "\tSource: {}\tCount: {}\n".format(src, count)
            output += string
        return output

    def __len__(self):
        return self.size
