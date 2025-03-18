import torch
import logging

logger = logging.getLogger('include')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        logger.debug("Dataset object created")
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

    def __len__(self):
        return len(self.labels)

