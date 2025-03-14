import logging
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from include.utils import create_pipeline, evaluate_predictions

logger = logging.getLogger('models')


def train_model(train_X, train_y, model):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=sent_tokenize, preprocessor=None)

    pipeline = create_pipeline(model, tfidf_vectorizer)
    logger.debug('Created the pipeline')

    logger.debug('Starting the model fitting')
    pipeline.fit(train_X, train_y)
    logger.debug(f'{type(model).__name__} successfully trained')

    return pipeline


def evaluate_model(test_X, test_y, pipeline):
    y_hat = pipeline.predict(test_X)
    logger.debug(f'Predicted {len(y_hat)} samples, evaluating them against {len(test_y)} samples')

    print(f"Results of {pipeline}")
    evaluate_predictions(y_hat, test_y)
