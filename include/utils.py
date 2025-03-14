import logging
from nltk.tokenize import NLTKWordTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
