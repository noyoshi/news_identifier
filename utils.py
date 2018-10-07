import re
import requests

from bs4 import BeautifulSoup

def clean_html(raw_html):
    # https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def get_soup(url):
    r = requests.get(url)
    return BeautifulSoup(r.text, features="html.parser")
