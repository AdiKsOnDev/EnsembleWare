import os
import sys
import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import train_test_split

from include.utils import setup_loggers
from include.models.pipelines import train_model, evaluate_model

LOG_LEVEL = logging.DEBUG
load_dotenv()
setup_loggers(LOG_LEVEL)

DATASET_PATH = os.getenv('DATASET_PATH')
RESULTS_PATH = os.getenv('RESULTS_PATH')

load_dotenv()
df = pd.read_csv(DATASET_PATH)

for feature in ('Sections', 'DLLs', 'Functions'):
    df_api = df[[feature, 'Label']].copy()
    df_api = df_api[df_api[feature].notna()]
    X = df_api[feature]
    y = df_api['Label'].astype('category').cat.codes

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, stratify=y)
    model = MultinomialNB()

    pipeline = train_model(train_X, train_y, model)
    evaluate_model(test_X, test_y, pipeline)
