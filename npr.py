#!/usr/bin/env python3
import argparse
import requests
import sys

from bs4 import BeautifulSoup

from article import Article
from utils import clean_html, get_soup
from news_source import NewsSource

"""Note: We need to scrub out the name of the news source, it will be replaced
with [SOURCE_NAME]"""

class NPR(NewsSource):
    def __init__(self):
        NewsSource.__init__(self, "https://text.npr.org", ["NPR", "PBS Newshour"])

    def _get_story_links(self):
        """Goes to the parent URL for NPR and updates links list to contain all
        linked articles"""
        soup = get_soup(self.BASE_URL)
        for tag in soup.find_all("a"):
            url = tag["href"]
            if not url.startswith("http"):
                # The "News" link marks the end of relavent links on this page
                if tag.text == "News": break
                self.links.append(self.BASE_URL + url)

    
    def get_articles(self):
        """Gets the articles from this news source, will be called the same for
        all classes"""
        self._get_story_links()
        
        for link in self.links:
            r = requests.get(link)
            soup = BeautifulSoup(r.text, features="html.parser")

            paragraph_list = soup.find_all("p")
            article_title = clean_html(str(paragraph_list[2]))

            paragraph_list = paragraph_list[5:]
            paragraph_list.pop()


            text = self._parse_text(paragraph_list)

            a = Article("NPR", article_title, text)
            print(a)
            self.articles.append(a)

        return self.articles

if __name__ == "__main__":
    npr = NPR()
    npr.get_articles()

