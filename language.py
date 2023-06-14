"""
language.py - Language processing utilities and exploration
"""

# First, let's explore the Glove word embeddings
# Prereq: download a Glove model, extract the zip file,
# and place same directory as this file.
# I've tested it with the smallest model, glove.6B.50d.txt
# https://nlp.stanford.edu/projects/glove/
# https://keras.io/examples/nlp/pretrained_word_embeddings/

from glob import glob
from os import path
import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant

TRACE_DEBUG = True

def trace (*args, **kwargs):
    if TRACE_DEBUG:
        print(*args, **kwargs)

class glove_embeddings:
    def __init__ (self):
        list_of_files = glob("data/glove/glove.*.txt")
        if len(list_of_files) == 0:
            trace ("Couldn't find any Glove files (glove.*.txt) in the current directory")
            return None
        self.glove_file = max(list_of_files, key=path.getmtime)
        trace (f"Using Glove file: {self.glove_file}")
        self.cosine_similarity = False  # False = use squared distance
        
        # Create a dictionary mapping words to their Glove numpy vectors
        self.embeddings_dict = {}
        with open(self.glove_file, encoding="utf8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                self.embeddings_dict[word] = coefs

        self.vocab_size = len(self.embeddings_dict)
        trace (f"Vocab size: {self.vocab_size}" )
        self.embedding_dim = len(self.embeddings_dict["the"])
        trace (f"Embedding vector size: {self.embedding_dim}")

    def get_distance (self, vec1, vec2) -> float:
        # Return the distance between two vectors
        if self.cosine_similarity:
            dot = np.dot(vec1, vec2)
            norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if np.isclose(norm, 0, atol=1e-10):
                return 0
            return dot/norm
        else:
            return np.dot(vec1 - vec2, vec1 - vec2)
        
    @property
    def sort_order (self) -> bool:
        # Return True if larger distances are better
        if self.cosine_similarity:
            return True
        else:
            return False

    def get_embedding (self, word : str) -> np.array:
        # Return the embedding vector for a word
        if word in self.embeddings_dict:
            return self.embeddings_dict[word]
        else:
            return None
        
    def get_words (self) -> list:
        # Return a list of all words in the vocabulary
        return list(self.embeddings_dict.keys())

    def find_closest_words (self, word : str, n : int = 10) -> dict:
        # Find the closest words to work based on cosine similarity
        d = {} # cosine similarity to work
        words = self.get_words()
        for w in words:
            if w != word:
                d[w] = self.get_distance(self.get_embedding(word), self.get_embedding(w))
        return sorted(d.items(), key=lambda x: x[1], reverse=self.sort_order)[:n]

    def find_analogies (self, word1 : str, word2 : str, word3 : str, n : int = 10) -> dict:
        # Find the closest "word4" words: word1 is to word2 as word3 is to word4
        # Maximize cosine similarity between word4 - word3 and word2 - word1
        d = {}
        words = self.get_words()
        diff1 = self.get_embedding(word1) - self.get_embedding(word2)
        word3_vec = self.get_embedding(word3)
        for word4 in words:
            diff2 = word3_vec - self.get_embedding(word4)
            d[word4] = self.get_distance(diff1, diff2)
        return sorted(d.items(), key=lambda x: x[1], reverse=self.sort_order)[:n]

class embedding_explorer:
    def __init__ (self):
        self.glove = glove_embeddings()
        self.HELP_STRING = """
            s to find words with similar meanings
            a to find analogies
            t to toggle between cosine similarity vs squared distance
            h for help
            q to quit
        """
    def explore(self):
        print ("Type h for help")
        while True:
            cmd = input("What do you want to do? ")
            if cmd == "h":
                print (self.HELP_STRING)
            elif cmd == "t":
                self.glove.cosine_similarity = not self.glove.cosine_similarity
                if self.glove.cosine_similarity:
                    print ("Using cosine similarity")
                else:
                    print ("Using squared distance")
            elif cmd == "a":
                words = input ("Enter three words, a b c, and we'll find d such that a is to b as c is to d: ")
                words = words.split()
                if len(words) != 3:
                    print ("You must enter three words")
                    continue
                vocab = self.glove.get_words()
                for w in words:
                    if w not in vocab:
                        print (f"Sorry, {w} is not in my vocabulary")
                        continue
                word1, word2, word3 = words
                print (f"Closest analogies for {word1} is to {word2} as {word3} is to:")
                for w, sim in self.glove.find_analogies(word1, word2, word3):
                    print (f"{w}: {sim}")
            elif cmd == "s":
                word = input ("What word? ")
                if word not in self.glove.get_words():
                    print (f"Sorry, {word} is not in my vocabulary")
                    continue
                print (f"Closest words to {word}:")
                for w, sim in self.glove.find_closest_words(word):
                    print (f"{w}: {sim}")
            elif cmd == "q":
                break

if __name__ == "__main__":
    explorer = embedding_explorer()
    explorer.explore()