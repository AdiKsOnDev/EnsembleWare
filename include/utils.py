import nltk
import logging
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from nltk.tokenize import NLTKWordTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger('include')

def tokenize(string):
    return NLTKWordTokenizer.tokenize(string)

def preprocess_textual(df, tokenizer, feature):
    """
    Preprocesses and divides the dataframe,
    returning X, y for the given feature

    Args:
        df (pd.Dataframe): Data
        tokenizer (nltk.tokenize.NLTKWordTokenizer): Tokenizer to be used
        feature (str): Feature that has to be encoded and vectorized

    Returns:
        X: Vector representations of the Functions
        y: Encoded labels for each Function combination
    """
    df_dlls = df[[feature, "Label"]].copy()
    df_dlls = df_dlls[df_dlls[feature].notna()]

    # Vectorization is done just to get the vocabulary size
    # Not the best solution, but easy to read understand
    logger.debug('Getting the vocabulary size')
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(df_dlls[feature])
    vocabulary = vectorizer.get_feature_names_out()
    vocabulary_size = len(vocabulary)
    logger.debug(f'Got vocabulary size of {vocabulary_size}')

    tokenizer.fit_on_texts(df_dlls[feature])
    logger.debug('Tokenized the texts')

    sequences = tokenizer.texts_to_sequences(df_dlls[feature])
    max_length = max(len(seq) for seq in sequences)
    labels = df_dlls['Label'].astype('category').cat.codes
    logging.debug('Converted tokens to sequences and encoded the labels')

    X = pad_sequences(sequences, maxlen=max_length, padding='post')
    y = labels 

    return X, y, vocabulary_size


def evaluate_predictions(y_hat: list[int], y: list[int]):
    """
    Calculates and prints out the accuracy, precision, recall, f1_score

    Args:
        y_hat (list[int]): Predictions of the model
        y (list[int]): Actual labels from the dataset

    Returns:
        dict: Dictionary with metrics
    """
    accuracy = accuracy_score(y_hat, y)
    precision = precision_score(y_hat, y)
    recall = recall_score(y_hat, y)
    f1 = f1_score(y_hat, y)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.5f}")

    return metrics

def setup_loggers(log_level):
    main_logger = logging.getLogger('main')
    include_logger = logging.getLogger('include')
    models_logger = logging.getLogger('models')

    logging.basicConfig(
        level=logging.WARNING,
        format="| %(name)s | [%(levelname)s] | %(filename)s:%(lineno)d | %(message)s |"
    )

    main_logger.setLevel(
        log_level
    )
    include_logger.setLevel(
        log_level
    )
    models_logger.setLevel(
        log_level
    )

class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, language='english'):
        nltk.download('stopwords')
        nltk.download('punkt_tab')

        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()


    def _case_normalization(self, text):
        """
        A method to change the parameter into a lowercase string.
        Args:
        text (string): Every word in a sentence

        Returns:
        (string): Returns the original string in lowercase
        """
        return str(text).lower()


    def _remove_stopwords(self, text):
        """
        A method to remove the words from the text that are in the stopwords list.
        Args:
        text (string): Every word in a sentence

        Returns:
        (string): Returns the concatenated text without the stopwords
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]

        return ' '.join(filtered_words)


    def _stem_text(self, text):
        """
        A method to remove the prefixes and suffixes from the words(stemming).
        Args:
        text (string): Every word in a sentence

        Returns:
        (string): Returns the concatenated text without the prefixes and suffixes
        """
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]

        return ' '.join(stemmed_words)


    def _preprocess_text(self, text):
        """
        A method to use the above techniques in order to preprocess the parameter
        text (string): Every word in a sentence

        Returns:
        (string): Returns the preprocessed text after case normalization, stopword removal, and stemming.
        """
        text = self._case_normalization(text)
        text = self._remove_stopwords(text)
        text = self._stem_text(text)

        return text


    def fit(self, X, y=None):
        return self


    def transform(self, X):
        for doc in X:
            yield self._preprocess_text(doc)

def create_pipeline(classifier, vectorizer):
    steps = [
        ('normalizer', TextNormalizer()),
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ]
    logger.debug(f'Making a pipeline with the {type(vectorizer).__name__} and {type(classifier).__name__}')

    return Pipeline(steps)
