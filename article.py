#!/usr/bin/env python3
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
