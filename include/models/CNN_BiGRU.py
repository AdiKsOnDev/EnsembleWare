import os
import logging
from dotenv import load_dotenv
from keras.models import load_model
from keras import layers, models, losses
from sklearn.model_selection import train_test_split

from include.utils import evaluate_predictions, preprocess_textual

load_dotenv()
logger = logging.getLogger('models')


def compile_model(VOCABULARY_SIZE):
    logger.debug('Model Compilation started')
    CNN_BiGRU = models.Sequential()

    CNN_BiGRU.add(layers.Embedding(VOCABULARY_SIZE, 10))
    logger.debug(f'Embedding layer added with vocabulary size of {VOCABULARY_SIZE}')

    # Convolutional Layer
    CNN_BiGRU.add(layers.Conv1D(128, 4, activation='relu'))
    CNN_BiGRU.add(layers.MaxPooling1D(2))
    CNN_BiGRU.add(layers.Dropout(0.2))
    logger.debug('Convolutional Layer added')

    # BiGRU
    CNN_BiGRU.add(layers.Bidirectional(layers.GRU(3)))
    logger.debug('BiGRU layer added')

    # Fully Connected Layer
    CNN_BiGRU.add(layers.Dense(25))
    CNN_BiGRU.add(layers.Dropout(0.5))
    CNN_BiGRU.add(layers.Dense(25))
    CNN_BiGRU.add(layers.Dropout(0.5))
    CNN_BiGRU.add(layers.Dense(1, activation='sigmoid'))
    logger.debug('Fully connected and output layers added')

    CNN_BiGRU.compile(loss=losses.BinaryCrossentropy(), optimizer='adam')
    logger.info('Model compiled, returning')

    return CNN_BiGRU


def run_CNN_BiGRU(df, tokenizer, feature='Sections', model_path="./models"):
    """
    Preprocesses the dataframe, then compiles, trains,
    and evaluates the CNN BiGRU model

    Args:
        df (pd.DataFrame): Dataset
        tokenizer (keras.preprocessing.text.Tokenizer): Keras tokenizer
        feature (str): Column to be used for model fitting
        model_path (str): Path where the function will save and/or look for the model
    """
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
        vocabulary_size += 100
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

    evaluate_predictions(y_hat, y)
