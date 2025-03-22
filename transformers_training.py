import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from include.Dataset import Dataset
from include.trainer import fine_tune
from include.utils import setup_loggers
from include.models.transformers.bert import BERTModel
from include.models.transformers.roformer import RoformerModel
from include.models.transformers.modernbert import ModernBERT

df = pd.read_csv('./data/PE_Dataset_Labeled.csv')
label_encoder = LabelEncoder()

setup_loggers(log_level=logging.DEBUG)
main_logger = logging.getLogger('main')

models = [
        ModernBERT(model_name="answerdotai/ModernBERT-base",
                       num_labels=2, max_length=1024),
        RoformerModel(model_name="junnyu/roformer_chinese_base", 
                      num_labels=2),
        BERTModel(model_name="bert-base-uncased", 
                  num_labels=2)
    ]
features = ["DLLs", "Functions", "Sections"]
directory = './results/transformers'

main_logger.info("Starting the cycle without preprocessing")
for feature in features:
    validation_X, validation_y = [], []

    for model in models:
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
        train_X, validation_X, train_y, validation_y = train_test_split(
            texts, labels, test_size=0.25, random_state=42, stratify=labels
        )


        # main_logger.debug(f"Started the pipeline for {model.model_name} using {feature} Names")
        #
        # main_logger.info(f"Tokenizing for {model.model_name}")
        #
        # train_X = model.tokenize(train_X)
        # test_X = model.tokenize(test_X)
        # train_dataset = Dataset(train_X, train_y)
        # test_dataset = Dataset(test_X, test_y)
        #
        # main_logger.debug(f"About to start fine-tuning {model.model_name}")
        # fine_tune(model, train_dataset, test_dataset, results_dir=f"./results/transformers/{feature}/")

    models_validation = [
            ModernBERT(model_name=f"{directory}/{feature}/answerdotai/ModernBERT-base/fine_tuned_answerdotai/ModernBERT-base", 
                       num_labels=2, 
                       max_length=1024),
            RoformerModel(model_name=f"{directory}/{feature}/junnyu/roformer_chinese_base/fine_tuned_junnyu/roformer_chinese_base/",
                       num_labels=2),
            BERTModel(model_name=f"{directory}/{feature}/bert-base-uncased/fine_tuned_bert-base-uncased/",
                       num_labels=2)
        ]

    print(f"Results for Models trained with {feature}:")
    for model in models_validation:
        predictions = model.predict(validation_X)

        accuracy = accuracy_score(validation_y, predictions)
        precision = precision_score(validation_y, predictions)
        recall = recall_score(validation_y, predictions)
        f1= f1_score(validation_y, predictions)

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
        }

        print(f"\tResults for {model.model_name}")
        for metric, value in metrics.items():
            print(f"\t\t{metric}: {value:.4f}")
