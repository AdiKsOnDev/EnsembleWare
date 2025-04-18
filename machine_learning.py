import os
import re
import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB, GaussianNB, BernoulliNB

from include.utils import setup_loggers
from include.pipelines import train_model, evaluate_model

LOG_LEVEL = logging.INFO
load_dotenv()
setup_loggers(LOG_LEVEL)
logger = logging.getLogger('main')

DATASET_PATH = os.getenv('DATASET_PATH')
RESULTS_PATH = os.getenv('RESULTS_PATH')

load_dotenv()
df = pd.read_csv(DATASET_PATH)
df = df.drop('Unnamed: 0', axis=1)

"""
Textual Features
"""
for feature in ('Sections', 'DLLs', 'Functions'):
    logger.info(f"Training MultinomialNB using {feature}")
    df_api = df[[feature, 'Label']].copy()
    df_api = df_api[df_api[feature].notna()]
    X = df_api[feature]
    y = df_api['Label'].astype('category').cat.codes

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, stratify=y)
    model = MultinomialNB()

    pipeline = train_model(train_X, train_y, model)
    evaluate_model(test_X, test_y, pipeline)

"""
Numerical Features
"""
X = df.drop(['File_Name', 'Sections', 'DLLs', 'Functions', 'Label'], axis=1)
y = df['Label'].astype('category').cat.codes

for column in X.columns:
    X[column].fillna(0, inplace=True)

def convert_hex(value):
    if isinstance(value, str) and value.startswith("0x"):
        try:
            return int(value, 16)
        except ValueError:
            return value
    return value

X = X.map(convert_hex)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, stratify=y)
for model in (GaussianNB(), MultinomialNB(), BernoulliNB(), ComplementNB(), RandomForestClassifier()):
    model.fit(train_X, train_y)

    evaluate_model(test_X, test_y, model)
