#!/usr/bin/env python3

class Article(object):
    def __init__(self, title, source, author, date, content):
        self.author = author
        self.date = date # YYYY-MM-DD
        self.raw_content = content
        self.source = source
        self.title = title

        self.listed_content = []

        self.aliases = [source]
        self.redacted_string = "[NEWS_SOURCE]"

    def _scrub_names(self, string):
        """Scrubs the name(s) of the news source from the string"""
        for name in self.aliases:
            string = string.replace(name, self.redacted_string)

        return string

    def _build_listed_content(self):
        for line in self.raw_content.split('\n'):
            for word in line.split(' '):
                word = self._scrub_names(word)
                self.listed_content.append(word)

    def __str__(self):
        return "Author: {}\nSource: {}\nTitle: {}\nBody: {}".format(
                self.author,
                self.source,
                self.title,
                self.raw_content)
