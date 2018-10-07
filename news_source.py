from utils import clean_html, get_soup

class NewsSource(object):
    """Abstract News Source class"""
    def __init__(self, base_url, names):
        self.BASE_URL = base_url
        self.names = names
        self.redacted_string = "[SOURCE_NAME]"
        self.links = []
        self.articles = [] # List of article objects

    def _scrub_names(self, string):
        """Scrubs the name(s) of the news source from the string"""
        for name in self.names:
            string = string.replace(name, self.redacted_string)

        return string

    def _parse_text(self, paragraphs):
        """Parses the paragraph list to return the body of the article"""
        text = ''
        for p in paragraphs:
            cleaned_text = clean_html(str(p))
            cleaned_text = self._scrub_names(cleaned_text)
            text += cleaned_text + '\n'

        return text.strip()

