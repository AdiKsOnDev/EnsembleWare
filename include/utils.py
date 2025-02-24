import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from nltk.tokenize import NLTKWordTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize(string):
    return NLTKWordTokenizer.tokenize(string)

def preprocess_functions_column(df, tokenizer):
    """
    Preprocesses and divides the dataframe,
    returning X, y for Functions

    Args:
        - df (pd.Dataframe): Data
        - tokenizer (nltk.tokenize.NLTKWordTokenizer): Tokenizer to be used

    Returns:
        - X: Vector representations of the Functions
        - y: Encoded labels for each Function combination
    """
    df_dlls = df[["Functions", "Label"]].copy()
    df_dlls = df_dlls[df_dlls["Functions"].notna()]
    api_calls = df_dlls['Functions']

    w2v_model = Word2Vec(sentences=api_calls, vector_size=100,
                         window=5, min_count=1, workers=4)
    tokenizer.fit_on_texts(df_dlls['Functions'])
    word_index = tokenizer.word_index
    embedding_dim = 100

    # +1 to account for padding symbol
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
        else:
            # Random for unknown words
            embedding_matrix[i] = np.random.uniform(-0.01, 0.01, embedding_dim)

    sequences = tokenizer.texts_to_sequences(df_dlls['Functions'])
    max_length = max(len(seq) for seq in sequences)
    labels = df_dlls['Label'].astype('category').cat.codes

    X = pad_sequences(sequences, maxlen=max_length, padding='post')
    y = tf.keras.utils.to_categorical(
        labels, num_classes=len(df_dlls['Label'].unique()))

    return X, y
