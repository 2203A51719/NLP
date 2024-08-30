import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt', quiet=True)
text = "Hello there! How are you doing today? NLP is fascinating."
sentences = sent_tokenize(text)
print("Sentence Tokens:")
for i, sentence in enumerate(sentences, 1):
    print(f"Sentence {i}: {sentence}")
words = word_tokenize(text)
print("\nWord Tokens:")
print(words)
from nltk.stem import PorterStemmer
words = ["running", "ran", "runs", "easily", "fairly"]
stemmer = PorterStemmer()
print("Original words and their stems:")
for word in words:
    stem = stemmer.stem(word)
    print(f"{word} -> {stem}")
    from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "NLP is fun and interesting.",
    "NLP involves linguistics and computer science."
]
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(texts)
print("Vocabulary:")
print(vectorizer.get_feature_names_out())
print("\nBag of Words representation:")
print(bow_matrix.toarray())
from sklearn.feature_extraction.text import TfidfVectorizer
sentences = [
    "NLP is an interesting field.",
    "It involves processing natural language."
]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF representation:")
print(tfidf_matrix.toarray())
Sentence Tokens:
Sentence 1: Hello there!
Sentence 2: How are you doing today?
Sentence 3: NLP is fascinating.

Word Tokens:
['Hello', 'there', '!', 'How', 'are', 'you', 'doing', 'today', '?', 'NLP', 'is', 'fascinating', '.']
Original words and their stems:
running -> run
ran -> ran
runs -> run
easily -> easili
fairly -> fairli
Vocabulary:
['and' 'computer' 'fun' 'interesting' 'involves' 'is' 'linguistics' 'nlp'
 'science']

Bag of Words representation:
[[1 0 1 1 0 1 0 1 0]
 [1 1 0 0 1 0 1 1 1]]
Vocabulary:
['an' 'field' 'interesting' 'involves' 'is' 'it' 'language' 'natural'
 'nlp' 'processing']

TF-IDF representation:
[[0.4472136 0.4472136 0.4472136 0.        0.4472136 0.        0.
  0.        0.4472136 0.       ]
 [0.        0.        0.        0.4472136 0.        0.4472136 0.4472136
  0.4472136 0.        0.4472136]]
