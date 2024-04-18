import torch
import nltk
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(text):
    return nltk.word_tokenize(text)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

a = "How does long shipping take?"
words = ['Organize', 'organizes', 'organizing']
stemmed_words = [stem(w) for w in words]
print(stemmed_words)