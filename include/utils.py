import os
import logging
from keras.models import load_model
from nltk.tokenize import NLTKWordTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.feature_extraction.text import CountVectorizer

from include.models.CNN_BiGRU import compile_model

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

def run_CNN_BiGRU(df, tokenizer, feature='Sections', model_path="./models"):
    df_api = df[[feature, 'Label']].copy()
    df_api = df_api[df_api[feature].notna()]

    train_api, test_api = train_test_split(
        df_api,
        test_size=0.25,
        random_state=42,
        stratify=df_api['Label']
    )
    logger.debug(f'Train set size is {len(train_api)}, while Test set size is {len(test_api)}')

    if not os.path.exists(f'{model_path}/CNN_BiGRU_{feature}.h5'):
        logger.warning(f'Model wasn\'t found in {model_path}/CNN_BiGRU_{feature}.h5, starting model compilation')
        X, y, vocabulary_size = preprocess_textual(train_api, tokenizer, feature)
        CNN_BiGRU = compile_model(vocabulary_size)

        CNN_BiGRU.fit(x=X, y=y, batch_size=32, epochs=16, verbose=2)
        CNN_BiGRU.save(f'{model_path}/CNN_BiGRU_{feature}.h5')
        logger.info(f'Model saved in {model_path}/CNN_BiGRU_{feature}.h5')
    else:
        CNN_BiGRU = load_model(f'{model_path}/CNN_BiGRU_{feature}.h5')
        logger.warning(f'Model loaded from {model_path}/CNN_BiGRU_{feature}.h5')

    X, y, _ = preprocess_textual(test_api, tokenizer, feature)

    y_hat = CNN_BiGRU.predict(X)
    y_hat = (y_hat > 0.5).astype(int)  # Classify the raw probabilities

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
