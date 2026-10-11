import re
from string import punctuation

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


for package in ("stopwords", "punkt", "punkt_tab", "wordnet"):
    nltk.download(package, quiet=True)


def preprocess(complaints):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    cleaned = []

    for complaint in complaints:
        text = str(complaint).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\d+", "", text)
        words = word_tokenize(text)
        cleaned_words = [
            lemmatizer.lemmatize(word)
            for word in words
            if word not in punctuation and word not in stop_words
        ]
        cleaned.append(" ".join(cleaned_words))

    return cleaned