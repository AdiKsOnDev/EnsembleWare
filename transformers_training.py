import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from include.Dataset import Dataset
from include.trainer import fine_tune
from include.utils import setup_loggers
from include.models.transformers.bert import BERTModel
from include.models.transformers.roformer import RoformerModel

df = pd.read_csv('./data/PE_Dataset_Labeled.csv')
label_encoder = LabelEncoder()

setup_loggers(log_level=logging.DEBUG)
main_logger = logging.getLogger('main')

models = [
        RoformerModel(model_name="junnyu/roformer_chinese_base", 
                      num_labels=2),
        BERTModel(model_name="bert-base-uncased", 
                  num_labels=2)
    ]
features = ["DLLs", "Functions", "Sections"]

main_logger.info("Starting the cycle without preprocessing")
for feature in features:
    for model in models:
        main_logger.debug(f"Started the pipeline for {model.model_name} using {feature} Names")

        df["Label"] = label_encoder.fit_transform(df["Label"])

        main_logger.debug(f"DataFrame size before filtering is {len(df)}")
        df_filtered = df[[feature, 'Label']].copy()
        df_filtered = df_filtered[df_filtered[feature].notna()]
        main_logger.debug(f"DataFrame size after filtering is {len(df_filtered)}")

        texts = df_filtered[feature].tolist()
        labels = df_filtered["Label"].tolist()

        train_X, test_X, train_y, test_y = train_test_split(
            texts, labels, test_size=0.25, random_state=42, stratify=labels
        )

        main_logger.info(f"Tokenizing for {model.model_name}")

        train_X = model.tokenize(train_X)
        test_X = model.tokenize(test_X)
        train_dataset = Dataset(train_X, train_y)
        test_dataset = Dataset(test_X, test_y)

        main_logger.debug(f"About to start fine-tuning {model.model_name}")
        fine_tune(model, train_dataset, test_dataset, results_dir=f"./results/transformers/{feature}/")
