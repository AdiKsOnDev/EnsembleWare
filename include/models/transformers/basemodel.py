import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger('models')

class BaseModel:
    def __init__(self, model_name, num_labels=2, max_length=512):
        logger.debug(f"{model_name} initialised with: \
                \n{num_labels} labels \
                \nmax length of {max_length}")
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self._load_model(model_name, num_labels)

    def _load_model(self, model_name, num_labels):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        logger.debug(f"Tokenizer and the model for {self.model_name} are initialised")

    def tokenize(self, texts):
        logger.debug(f"Tokenising {len(texts)} texts")
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    def forward(self, inputs):
        return self.model(**inputs)


    def predict(self, texts, batch_size=4):
        tokenized_data = self.tokenize(texts)

        input_ids = tokenized_data["input_ids"]
        attention_mask = tokenized_data["attention_mask"]

        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            logger.info(f"{self.model.__class__.__name__} is Predicting")

            for batch in tqdm(dataloader, desc="Processing Batches", unit="batch"):
                input_ids, attention_mask = [b.to(self.model.device) for b in batch]

                outputs = self.forward({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })
                logits = outputs.logits

                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(batch_predictions)

        logger.debug("{model.name} successfully predicted the given text")
        return predictions

