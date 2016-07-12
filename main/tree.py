import utils

class tree(object):

    def __init__(self, phrase):
        self.phrase = phrase.strip()
        self.embeddings = []
        self.representation = None

    def populate_embeddings_characters(self, chars):
        phrase = " " + self.phrase.lower() + " "
        for i in phrase:
            self.embeddings.append(utils.lookup(chars, i))

    def unpopulate_embeddings(self):
        self.embeddings = []