import os
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from keras.models import load_model
from nltk.tokenize import NLTKWordTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def tokenize(string):
    return NLTKWordTokenizer.tokenize(string)

def preprocess_for_CNN_BiGRU(df, tokenizer, feature):
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

    tokenizer.fit_on_texts(df_dlls[feature])

    sequences = tokenizer.texts_to_sequences(df_dlls[feature])
    max_length = max(len(seq) for seq in sequences)
    labels = df_dlls['Label'].astype('category').cat.codes

    X = pad_sequences(sequences, maxlen=max_length, padding='post')
    y = tf.keras.utils.to_categorical(
        labels, num_classes=len(df_dlls['Label'].unique()))

    return X, y

def run_CNN_BiGRU(df, tokenizer, feature='Sections', model_path="./models"):
    df_api = df[[feature, 'Label']].copy()
    df_api = df_api[df_api[feature].notna()]

    train_api, test_api = train_test_split(
        df_api,
        test_size=0.25,
        random_state=42,
        stratify=df_api['Label']
    )

    if not os.path.exists(f'{model_path}/CNN_BiGRU_{feature}.h5'):
        CNN_BiGRU = compile()
        X, y = preprocess_for_CNN_BiGRU(train_api, tokenizer, feature)

        CNN_BiGRU.fit(x=X, y=y, batch_size=32, epochs=16, verbose=2)
        CNN_BiGRU.save(f'{model_path}/CNN_BiGRU_Sections.h5')
    else:
        CNN_BiGRU = load_model(f'{model_path}/CNN_BiGRU_Sections.h5')

    X, y = preprocess_for_CNN_BiGRU(test_api, tokenizer, feature)

    y_hat = CNN_BiGRU.predict(X)
    y_hat = (y_hat > 0.5).astype(int)  # Classify the raw probabilities

    accuracy = accuracy_score(y_hat, y)
    precision = precision_score(y_hat, y, average="weighted")
    recall = recall_score(y_hat, y, average="weighted")
    f1 = f1_score(y_hat, y, average="weighted")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.5f}")
