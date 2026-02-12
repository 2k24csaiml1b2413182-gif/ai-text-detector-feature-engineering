import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import textstat

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

class FeatureEngine:
    def __init__(self, text):
        # If text is empty/NaN, treat it as empty string
        self.text = str(text) if text else ""
        self.sentences = sent_tokenize(self.text)
        self.words = word_tokenize(self.text)
        #remove punctuation
        self.words_only = [w.lower() for w in self.words if w.isalpha()]

    def get_avg_sentence_length(self):
        # Logic: AI sentences are often uniform in length
        if len(self.sentences) == 0: return 0
        return len(self.words) / len(self.sentences)

    def get_vocab_richness(self):
        # Logic: AI uses repetitive words. Humans use diverse words.
        if len(self.words_only) == 0: return 0
        unique_words = set(self.words_only)
        return len(unique_words) / len(self.words_only)

    def get_burstiness(self):
        # Logic: Variation in sentence length. High variance = Human.
        lengths = [len(word_tokenize(s)) for s in self.sentences]
        if not lengths: return 0
        return np.std(lengths) # Standard Deviation

    def get_readability(self):
        # Logic: AI often writes at a specific grade level
        return textstat.flesch_reading_ease(self.text)

    def extract_all(self):
        return {
            "avg_sent_len": self.get_avg_sentence_length(),
            "vocab_richness": self.get_vocab_richness(),
            "burstiness": self.get_burstiness(),
            "readability": self.get_readability()
        }
