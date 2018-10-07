class Article(object):
    def __init__(self, source, title, text):
        self.source = source
        self.title = title
        self.text = text

    def get_source(self):
        return self.source
    
    def get_title(self):
        return self.title

    def get_text(self):
        return self.text

    def __str__(self):
        return "Souce: {}\nTitle: {}\nText: {}".format(
                self.source,
                self.title,
                self.text)
