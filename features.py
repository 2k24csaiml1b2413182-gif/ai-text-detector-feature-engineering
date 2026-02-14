import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import textstat
import numpy as np
from textblob import TextBlob
from sentence_transformers import SentenceTransformer # <--- NEW LIBRARY

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

class FeatureEngine:
    # --- THE BRAIN (Loaded once for the whole program) ---
    # We use 'all-MiniLM-L6-v2': Fast, small, and smart.
    model = SentenceTransformer('all-MiniLM-L6-v2') 

    def __init__(self, text):
        self.text = text
        if not text or not isinstance(text, str):
            self.sentences = []
            self.words = []
            self.blob = TextBlob("")
        else:
            self.sentences = sent_tokenize(text)
            self.words = word_tokenize(text)
            self.blob = TextBlob(text)

    # --- EXISTING FEATURES (The "Accountant") ---
    def extract_avg_sentence_length(self):
        if len(self.sentences) == 0: return 0
        return len(self.words) / len(self.sentences)

    def extract_vocab_richness(self):
        if len(self.words) == 0: return 0
        return len(set(word.lower() for word in self.words)) / len(self.words)

    def extract_burstiness(self):
        if len(self.sentences) == 0: return 0
        lengths = [len(word_tokenize(s)) for s in self.sentences]
        return np.std(lengths)

    def extract_sentiment(self):
        return self.blob.sentiment.polarity

    def extract_subjectivity(self):
        return self.blob.sentiment.subjectivity

    def extract_pos_ratios(self):
        if not self.words: return {'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0}
        tags = self.blob.tags
        counts = {'N': 0, 'V': 0, 'J': 0}
        for word, tag in tags:
            if tag.startswith('N'): counts['N'] += 1
            elif tag.startswith('V'): counts['V'] += 1
            elif tag.startswith('J'): counts['J'] += 1
        total = len(tags)
        return {
            'noun_ratio': counts['N'] / total,
            'verb_ratio': counts['V'] / total,
            'adj_ratio': counts['J'] / total
        }

    # --- NEW FEATURE (The "Philosopher") ---
    def extract_embeddings(self):
        # Turns text into a list of 384 numbers (The Vector)
        if not self.text.strip():
            return [0.0] * 384
        
        # .encode() returns a numpy array. We convert to list for the CSV.
        vector = self.model.encode(self.text).tolist()
        return vector

    def extract_all(self):
        # 1. Standard Features
        features = {
            "avg_sent_len": self.extract_avg_sentence_length(),
            "vocab_richness": self.extract_vocab_richness(),
            "burstiness": self.extract_burstiness(),
            "sentiment": self.extract_sentiment(),
            "subjectivity": self.extract_subjectivity()
        }
        
        # 2. Add Grammar
        features.update(self.extract_pos_ratios())
        
        # 3. Add Embeddings (Flatten the list)
        # We create columns: embed_0, embed_1 ... embed_383
        vector = self.extract_embeddings()
        for i, val in enumerate(vector):
            features[f'embed_{i}'] = val
            
        return features
