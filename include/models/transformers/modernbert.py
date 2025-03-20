from include.models.transformers.basemodel import BaseModel


class ModernBERT(BaseModel):
    def __init__(self, model_name, num_labels=2, max_length=8192):
        super().__init__(model_name, num_labels, max_length)
