import nltk.tokenize as tkn
import nltk.corpus as co
import nltk.stem.porter as po
import nltk.stem.wordnet as wo
import re as re

def normalize_text(text, num_value='num_value'):
    """
    Normalize a message for analysis (ie. convert to lower case alpha numeric text)
    :param text: The input text
    :param num_value: The default replacement for a numeric value
    :return: The normalized text
    """

    return re.sub(r'[^a-zA-Z ]+', num_value, re.sub(r'[^a-zA-Z0-9 ]', '', text.lower()))

def tokenize_text(text):
    """
    Splits text into an array of tokens
    :param text: The input text
    :return: The tokenized text
    """

    return tkn.word_tokenize(text)

def remove_stopwords(tokenized_text):
    """
    Removes stopwords from a piece of text
    :param tokenized_text: The input text array of tokens
    :return: The output text array of tokens without the stopwords
    """

    return [token for token in tokenized_text if token not in __stop_words__]

def stem_text(tokenized_text):
    """
    Stems all tokens in the input tokenized text
    :param tokenized_text: The tokenized text
    :return: The tokenized text with stemmed words
    """

    return [__stemmer__.stem(token) for token in tokenized_text]

def lemmatize_text(tokenized_text):
    """
    Lemmatizes all tokens in the input tokenized text
    :param tokenized_text: The tokenized text
    :return: The tokenized text with lemmatized words
    """

    return [__lemmatizer__.lemmatize(token) for token in tokenized_text]

#region Private

# Locally initialized stop words (optimization)
__stop_words__ = co.stopwords.words('english')

# Locally initialized stemmer (optimization)
__stemmer__ = po.PorterStemmer()

# Locally initialized lemmatizer (optimization)
__lemmatizer__ = wo.WordNetLemmatizer()

#endregion